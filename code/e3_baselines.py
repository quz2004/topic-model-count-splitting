"""E3 baselines (NMF on log-norm, LDVAE) under the same protocol as e3_lupus.py: count split, HVG + model on train half,
usage on held-out half, donor x cell-type pseudobulk, covariate-adjusted donor tests, 5x5 CV AUROC. K=30.
usage: python code/e3_baselines.py SEED {nmf|ldvae|cnmf|scetm} [null]   (null: permutation-null calibration only, skips CV)"""
import sys, json, numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, scanpy as sc, torch
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import NMF
sys.path.insert(0, 'code')
from nbtm import split_counts, lognorm
seed = int(sys.argv[1]); METHOD = sys.argv[2]; K = 30
a = ad.read_h5ad('data/perez_sub.h5ad'); X = sp.csr_matrix(a.X); X = X.astype(np.int64)
obs = a.obs.reset_index(drop=True); genes = np.array(a.var_names)
donor = obs.donor_id.astype(str).values; ctype = obs.cell_type.astype(str).values
top_ct = pd.Series(ctype).value_counts().index[:8].tolist()
hall = json.load(open('data/hallmark_ifn.json')); ISG = set(hall['Interferon Alpha Response']) | set(hall['Interferon Gamma Response'])
tr, te = split_counts(X, 0.5, seed)
t = ad.AnnData(tr.astype(np.float32)); t.var_names = genes
sc.pp.filter_genes(t, min_cells=50); sc.pp.normalize_total(t, target_sum=1e4); sc.pp.log1p(t)
sc.pp.highly_variable_genes(t, n_top_genes=2000, flavor='seurat')
hv = np.where(np.isin(genes, t.var_names[t.var.highly_variable]))[0]; g = genes[hv]; isg_mask = np.isin(g, list(ISG))
trh, teh = tr[:, hv], te[:, hv]
if METHOD == 'nmf':
    nmf = NMF(K, init='nndsvda', random_state=seed, max_iter=300).fit(lognorm(trh).toarray())
    U = nmf.transform(lognorm(teh).toarray()); U = U / (U.sum(1, keepdims=True) + 1e-12); L = nmf.components_ / (nmf.components_.mean(0) + 1e-12)
elif METHOD == 'cnmf':
    import tempfile, os
    from cnmf import cNMF
    wd = tempfile.mkdtemp(prefix=f'cnmf{seed}_'); fn = os.path.join(wd, 'train.h5ad')
    ta = ad.AnnData(sp.csr_matrix(trh).astype(np.float32)); ta.var_names = g; ta.obs_names = [str(i) for i in range(ta.n_obs)]
    keep = np.asarray(ta.X.sum(1)).ravel() > 0; ta = ta[keep].copy(); ta.write_h5ad(fn)
    open(os.path.join(wd, 'genes.txt'), 'w').write('\n'.join(g))
    cn = cNMF(output_dir=wd, name='r'); cn.prepare(fn, components=[K], n_iter=20, seed=seed, genes_file=os.path.join(wd, 'genes.txt'), init='random')
    cn.factorize(); cn.combine(); cn.consensus(K, density_threshold=2.0, show_clustering=False, build_ref=False, refit_usage=False)
    Sp = pd.read_csv(cn.paths['consensus_spectra__txt'] % (K, '2_0'), sep='\t', index_col=0)
    sd = np.sqrt(np.asarray(((trh - trh.mean(0)) ** 2).mean(0)).ravel() * trh.shape[0] / (trh.shape[0] - 1)) if not sp.issparse(trh) else \
        np.sqrt(np.asarray(trh.power(2).mean(0)).ravel() - np.asarray(trh.mean(0)).ravel() ** 2) * np.sqrt(trh.shape[0] / (trh.shape[0] - 1))
    Xte = (sp.csr_matrix(teh).toarray() / (sd + 1e-12)).astype(np.float64)
    U = cn.refit_usage(Xte, Sp.values); U = U / (U.sum(1, keepdims=True) + 1e-12); L = Sp.values / (Sp.values.mean(0) + 1e-12)
elif METHOD == 'scetm':
    from scETM import scETM as SC, UnsupervisedTrainer
    torch.manual_seed(seed); np.random.seed(seed)
    def mk(M):
        b = ad.AnnData(sp.csr_matrix(M).astype(np.float32)); b.var_names = g; b.obs['batch_indices'] = 0; return b
    atr, ate = mk(trh), mk(teh)
    dev = torch.device('cpu')
    m = SC(atr.n_vars, 1, n_topics=K, trainable_gene_emb_dim=128, device=dev)
    tr_ = UnsupervisedTrainer(m, atr, test_ratio=0., seed=seed, batch_size=2000)
    tr_.train(n_epochs=100, eval=False, save_model_ckpt=False, n_samplers=1)
    m.get_all_embeddings_and_nll(ate, inplace=True); m.get_all_embeddings_and_nll(atr, inplace=True)
    U = np.asarray(ate.obsm['theta']); U = U / U.sum(1, keepdims=True)
    B = atr.uns['alpha'] @ atr.varm['rho'].T; L = B - B.mean(1, keepdims=True)   # topic x gene logits (beta = softmax)
else:
    import scvi
    scvi.settings.seed = seed; torch.manual_seed(seed)
    def mk(M):
        b = ad.AnnData(sp.csr_matrix(M).astype(np.float32)); b.var_names = g; return b
    atr, ate = mk(trh), mk(teh)
    scvi.model.LinearSCVI.setup_anndata(atr)
    m = scvi.model.LinearSCVI(atr, n_latent=K); m.train(max_epochs=60, accelerator='cpu', plan_kwargs=dict(lr=1e-3))
    L = m.get_loadings().values.T; U = m.get_latent_representation(ate)
    L = np.vstack([L, -L]); U = np.hstack([U, -U])   # latent sign arbitrary: both orientations are candidate programs
    # NB: for prediction use signed latents only (U[:, :K])
Kp = U.shape[1]
# ---- donor table (identical to e3_lupus.py)
dd = obs.groupby('donor_id', observed=True).agg(case=('disease', lambda s: int((s.astype(str) != 'normal').iloc[0])),
                                               sex=('sex', 'first'), eth=('self_reported_ethnicity', 'first'), stage=('development_stage', 'first'))
dd['age'] = dd.stage.astype(str).str.extract(r'(\d+)')[0].astype(float); dd['age'] = dd.age.fillna(dd.age.median()); dd['male'] = (dd.sex.astype(str) == 'male').astype(int)
E = pd.get_dummies(dd.eth.astype(str), drop_first=True, dtype=float); E = E.loc[:, E.sum() >= 10]
cov = pd.concat([dd[['age', 'male']].astype(float), E], axis=1); cov = (cov - cov.mean()) / (cov.std() + 1e-9); cov = cov.loc[:, cov.std() > 0]
donors = dd.index.astype(str).tolist(); dd.index = dd.index.astype(str)
def pseudobulk(M, cts, min_cells=10):
    out = {}
    for c in cts:
        mk_ = ctype == c; df = pd.DataFrame(M[mk_]); df['d'] = donor[mk_]
        cnt = df.groupby('d').size(); mean = df.groupby('d').mean(); mean[cnt.reindex(mean.index) < min_cells] = np.nan
        out[c] = mean.reindex(donors)
    return out
def enrich(Hm, top=50):
    out = []
    for k in range(Hm.shape[0]):
        idx = np.argsort(-Hm[k])[:top]; out.append(stats.hypergeom.sf(isg_mask[idx].sum() - 1, len(g), isg_mask.sum(), top))
    return np.array(out)
pe = enrich(L); k_isg = int(np.argmin(pe)); res = dict(seed=seed, method=METHOD, K=K, isg_topic=dict(k=k_isg, enrich_p=float(pe[k_isg])))
Upred = U if METHOD == 'nmf' else U[:, :K]
PB = pseudobulk(Upred, top_ct); PBi = pseudobulk(U[:, [k_isg]], top_ct)
def donor_test(y, case=None):
    case = dd.case if case is None else case
    ok = ~np.isnan(y.values)
    if ok.sum() < 30 or case[ok].nunique() < 2 or case[ok].sum() < 8: return np.nan, np.nan
    Xd = sm.add_constant(pd.concat([case.astype(float).rename('case'), cov], axis=1).loc[ok]); fit = sm.OLS(y.values[ok], Xd).fit(cov_type='HC3')
    return float(fit.params['case'] / np.nanstd(y.values)), float(fit.pvalues['case'])
if len(sys.argv) > 3 and sys.argv[3] == 'null':
    # same 10-permutation donor-label shuffle as e3_lupus.py (covariates kept), all cell type x component tests
    rng = np.random.default_rng(seed); nulls = []
    for _ in range(10):
        pc = pd.Series(rng.permutation(dd.case.values), index=dd.index)
        for c in top_ct:
            for k in range(Upred.shape[1]):
                p_ = donor_test(PB[c][k], pc)[1]
                if not np.isnan(p_): nulls.append(p_)
    nulls = np.array(nulls); np.save(f'results/e3_nullp_seed{seed}_{METHOD}.npy', nulls)
    out = dict(seed=seed, method=METHOD, n=int(len(nulls)), frac_p05=float((nulls < .05).mean()), frac_p01=float((nulls < .01).mean()))
    json.dump(out, open(f'results/e3_basenull_{METHOD}_seed{seed}.json', 'w')); print(out); sys.exit()
res['isg_by_ct'] = {c: dict(zip(['eff_sd', 'p'], donor_test(PBi[c][0]))) for c in top_ct}
def feat_pb(PBd):
    F = pd.concat([PBd[c].add_prefix(c + '|') for c in top_ct], axis=1); return F.fillna(F.mean()).values
y = dd.case.values; feats = {'topics_pb': feat_pb(PB)}; feats['topics+cov'] = np.hstack([feats['topics_pb'], cov.values])
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed); auc = {}
for name, F in feats.items():
    a_ = []
    for tri, tei in cv.split(F, y):
        clf = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=8, cv=3, penalty='l2', max_iter=2000, scoring='roc_auc'))
        clf.fit(F[tri], y[tri]); a_.append(roc_auc_score(y[tei], clf.predict_proba(F[tei])[:, 1]))
    auc[name] = a_
res['auc_mean'] = {k: float(np.mean(v)) for k, v in auc.items()}
json.dump(auc, open(f'results/e3_auc_folds_seed{seed}_{METHOD}.json', 'w'))
json.dump(res, open(f'results/e3_base_{METHOD}_seed{seed}.json', 'w'), indent=1); print(json.dumps(res, indent=1))

"""E3: donor-level inference on topic usage in the Perez et al. lupus PBMC cohort (Census subsample).
usage: python code/e3_lupus.py SEED [K]
Protocol: count split -> HVG + topic model on train half only (labels never used) -> topic usage on held-out half
-> donor x cell-type pseudobulk of topic usage -> (i) covariate-adjusted case/control tests, (ii) cross-validated
donor-level disease prediction (topics vs cell-type proportions vs covariates vs PCA pseudobulk).
"""
import sys, os, re, json, numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, scanpy as sc
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
sys.path.insert(0, 'code')
from nbtm import *

seed = int(sys.argv[1]); K = int(sys.argv[2]) if len(sys.argv) > 2 else 30
MODE = sys.argv[3] if len(sys.argv) > 3 else 'plain'   # 'ct': cell-type intercepts in decoder (identity absorbed, topics = within-type programs)
SPLIT = sys.argv[4] if len(sys.argv) > 4 else 'binom'   # 'bb': beta-binomial split with per-gene NB size from code/nb_size.py (data/perez_sub_nbsize.npy)
DATA = os.environ.get('E3_DATA', 'perez_sub')   # e.g. E3_DATA=covid_sub for the external replication cohort (code/fetch_covid.py)
SUF = ('' if MODE == 'plain' else '_' + MODE) + ('' if K == 30 else f'_K{K}') + ('' if SPLIT == 'binom' else '_bb') + ('' if DATA == 'perez_sub' else '_' + DATA)
a = ad.read_h5ad(f'data/{DATA}.h5ad')
X = sp.csr_matrix(a.X); assert np.allclose(X.data, np.round(X.data)); X = X.astype(np.int64)
obs = a.obs.reset_index(drop=True); genes = np.array(a.var_names)
case = (obs.disease.astype(str) != 'normal').values.astype(int)
donor = obs.donor_id.astype(str).values
ctype = obs.cell_type.astype(str).values
top_ct = [c for c in pd.Series(ctype).value_counts().index if c not in ('other', 'platelet')][:8]   # no-op for the lupus data
hall = json.load(open('data/hallmark_ifn.json')); ISG = set(hall['Interferon Alpha Response']) | set(hall['Interferon Gamma Response'])

tr, te = split_counts(X, 0.5, seed, size=None if SPLIT == 'binom' else np.load('data/perez_sub_nbsize.npy'))
t = ad.AnnData(tr.astype(np.float32)); t.var_names = genes
sc.pp.filter_genes(t, min_cells=50); sc.pp.normalize_total(t, target_sum=1e4); sc.pp.log1p(t)
sc.pp.highly_variable_genes(t, n_top_genes=2000, flavor='seurat')
hv = np.where(np.isin(genes, t.var_names[t.var.highly_variable]))[0]; g = genes[hv]; isg_mask = np.isin(g, list(ISG))
trh, teh = tr[:, hv], te[:, hv]
bt = pd.Categorical(ctype).codes if MODE == 'ct' else None
m = fit_ntm(trh, K, batch=bt, epochs=60, bs=256, seed=seed)                      # NO donor/disease covariates in the model
th = encode_theta(m, teh); beta = topic_gene(m)

# ---- donor table
dd = obs.groupby('donor_id', observed=True).agg(case=('disease', lambda s: int((s.astype(str) != 'normal').iloc[0])),
                                               sex=('sex', 'first'), eth=('self_reported_ethnicity', 'first'),
                                               stage=('development_stage', 'first'))
dd['age'] = dd.stage.astype(str).str.extract(r'(\d+)')[0].astype(float)
DEC = {'second': 15, 'third': 25, 'fourth': 35, 'fifth': 45, 'sixth': 55, 'seventh': 65, 'eighth': 75, 'ninth': 85, 'tenth': 95}   # 'sixth decade stage' (Stephenson); no-op for lupus (numeric ages)
dd['age'] = dd.age.fillna(dd.stage.astype(str).str.extract(r'(\w+) decade')[0].map(DEC))
dd['age'] = dd.age.fillna(dd.age.median()); dd['male'] = (dd.sex.astype(str) == 'male').astype(int)
E = pd.get_dummies(dd.eth.astype(str), drop_first=True, dtype=float); E = E.loc[:, E.sum() >= 10]   # rare categories give leverage-1 donors (degenerate HC3)
cov = pd.concat([dd[['age', 'male']].astype(float), E], axis=1); cov = (cov - cov.mean()) / (cov.std() + 1e-9); cov = cov.loc[:, cov.std() > 0]
donors = dd.index.astype(str).tolist(); dd.index = dd.index.astype(str)

def pseudobulk(M, cts, min_cells=10):
    """donor x (celltype*feature) mean of M (cells x F); NaN-filled -> column mean imputation."""
    out = {}
    for c in cts:
        mk = ctype == c
        df = pd.DataFrame(M[mk]); df['d'] = donor[mk]
        cnt = df.groupby('d').size(); mean = df.groupby('d').mean()
        mean[cnt.reindex(mean.index) < min_cells] = np.nan
        out[c] = mean.reindex(donors)
    return out

res = dict(seed=seed, K=K, n_donors=len(donors), n_cases=int(dd.case.sum()), n_cells=int(X.shape[0]))
PB = pseudobulk(th, top_ct)

# ---- (i) topic enrichment (label-free ISG topic) and covariate-adjusted donor-level tests
H = beta
def enrich(Hm, top=50):
    out = []
    for k in range(Hm.shape[0]):
        w = Hm[k] / (Hm.mean(0) + 1e-12); idx = np.argsort(-w)[:top]
        out.append(stats.hypergeom.sf(isg_mask[idx].sum() - 1, len(g), isg_mask.sum(), top))
    return np.array(out)
pe = enrich(H); k_isg = int(np.argmin(pe)); res['isg_topic'] = dict(k=k_isg, enrich_p=float(pe[k_isg]))

def donor_test(y, case=None):
    case = dd.case if case is None else case
    ok = ~np.isnan(y.values)
    if ok.sum() < 30 or case[ok].nunique() < 2 or case[ok].sum() < 8: return np.nan, np.nan, int(ok.sum())
    Xd = sm.add_constant(pd.concat([case.astype(float).rename('case'), cov], axis=1).loc[ok]); fit = sm.OLS(y.values[ok], Xd).fit(cov_type='HC3')
    sd = np.nanstd(y.values)
    return fit.params['case'] / sd, fit.pvalues['case'], int(ok.sum())
rows = []
for c in top_ct:
    for k in range(K):
        e, p, n = donor_test(PB[c][k]); rows.append(dict(ct=c, topic=k, eff_sd=e, p=p, n=n))
T = pd.DataFrame(rows); T['q'] = np.nan; v = T.p.notna(); T.loc[v, 'q'] = multipletests(T.p[v], method='fdr_bh')[1]; T.to_csv(f'results/e3_tests_seed{seed}{SUF}.csv', index=False)
res['isg_by_ct'] = {c: dict(eff_sd=float(T[(T.ct == c) & (T.topic == k_isg)].eff_sd.iloc[0]), p=float(T[(T.ct == c) & (T.topic == k_isg)].p.iloc[0])) for c in top_ct}
res['n_sig_q05'] = int((T.q < .05).sum()); res['n_tests'] = int(v.sum())

# permutation null: shuffle disease labels among donors (covariates kept) -> p-values should be uniform
rng = np.random.default_rng(seed); nulls = []
for _ in range(10):
    pc = pd.Series(rng.permutation(dd.case.values), index=dd.index)
    for c in top_ct:
        for k in range(K):
            p = donor_test(PB[c][k], pc)[1]
            if not np.isnan(p): nulls.append(p)
nulls = np.array(nulls); res['null_perm'] = dict(n=int(len(nulls)), frac_p05=float((nulls < .05).mean()), frac_p01=float((nulls < .01).mean()))
np.save(f'results/e3_nullp_seed{seed}{SUF}.npy', nulls)

# ---- (ii) cross-validated donor-level disease prediction
def feat_prop():
    cnt = pd.crosstab(pd.Series(donor, name='d'), pd.Series(ctype, name='c')).reindex(donors).fillna(0)[top_ct]
    return (cnt.div(cnt.sum(1), axis=0)).values
def feat_pb(PBd):
    F = pd.concat([PBd[c].add_prefix(c + '|') for c in top_ct], axis=1)
    return F.fillna(F.mean()).values
Yf = lognorm(teh).toarray(); Z = PCA(30, random_state=seed).fit_transform(Yf - Yf.mean(0)); del Yf
PB_pca = pseudobulk(Z, top_ct)
y = dd.case.values
feats = {'covariates': cov.values, 'celltype_props': feat_prop(), 'topics_pb': feat_pb(PB), 'pca_pb': feat_pb(PB_pca)}
feats['props+cov'] = np.hstack([feats['celltype_props'], cov.values])
feats['topics+cov'] = np.hstack([feats['topics_pb'], cov.values])
feats['topics+props+cov'] = np.hstack([feats['topics_pb'], feats['celltype_props'], cov.values])
feats['pca+props+cov'] = np.hstack([feats['pca_pb'], feats['celltype_props'], cov.values])
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=seed)
auc = {}
for name, F in feats.items():
    a_ = []
    for tri, tei in cv.split(F, y):
        clf = make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=8, cv=3, penalty='l2', max_iter=2000, scoring='roc_auc'))
        clf.fit(F[tri], y[tri]); a_.append(roc_auc_score(y[tei], clf.predict_proba(F[tei])[:, 1]))
    auc[name] = a_
res['auc_mean'] = {k: float(np.mean(v)) for k, v in auc.items()}; res['auc_sd'] = {k: float(np.std(v)) for k, v in auc.items()}
json.dump({k: v for k, v in auc.items()}, open(f'results/e3_auc_folds_seed{seed}{SUF}.json', 'w'))
np.savez_compressed(f'results/e3_seed{seed}{SUF}.npz', beta=beta, genes=g)
json.dump(res, open(f'results/e3_seed{seed}{SUF}.json', 'w'), indent=1); print(json.dumps(res, indent=1))
from runlog import log_run
log_run(seed, [f'results/e3_seed{seed}{SUF}.json', f'results/e3_auc_folds_seed{seed}{SUF}.json', f'results/e3_tests_seed{seed}{SUF}.csv', f'results/e3_nullp_seed{seed}{SUF}.npy'],
        dict(K=K, mode=MODE, split=SPLIT, data=DATA, ntm_seed=seed, split_seed=seed, pca_seed=seed, cv_seed=seed, perm_seed=seed))

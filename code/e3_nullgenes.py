"""E3b: semi-synthetic null genes on real lupus counts -> gene-level calibration of naive vs count-split tests.
usage: python code/e3_nullgenes.py SEED [NNULL=300]
NNULL real genes (expressed in >=5% of cells) are made null by permuting their counts across cells within each
donor x cell-type stratum (keeps depth, sparsity and donor/cell-type means; breaks any link to the cell's topic usage).
They are appended to the matrix and always kept in the modeled gene set. NTM-ct (K=30) is fit either on all counts (naive)
or on the train half (split); gene-topic partial correlations (stratum means and log library size projected out of both
sides, so donor/cell-type structure cannot create associations) are computed on the same counts used for fitting (naive)
or on the held-out half (split). Reported: fraction of null gene x topic pairs with p<0.05 / 0.01.
"""
import sys, os, json, numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, scanpy as sc
from scipy import stats
sys.path.insert(0, 'code')
from nbtm import *
from nb_size import estimate_size

seed = int(sys.argv[1]); NN = int(sys.argv[2]) if len(sys.argv) > 2 else 300; K = 30
a = ad.read_h5ad('data/perez_sub.h5ad')
X = sp.csr_matrix(a.X).astype(np.int64); obs = a.obs.reset_index(drop=True); genes = np.array(a.var_names)
strata = pd.factorize(obs.donor_id.astype(str) + '|' + obs.cell_type.astype(str))[0]
ctcode = pd.Categorical(obs.cell_type.astype(str)).codes
rng = np.random.default_rng(1000 + seed)
frac = np.asarray((X > 0).mean(0)).ravel(); cand = np.where(frac >= 0.05)[0]
src = rng.choice(cand, NN, replace=False)
Xd = X[:, src].toarray(); order = np.argsort(strata, kind='stable'); bounds = np.flatnonzero(np.diff(strata[order])) + 1
for idx in np.split(order, bounds):                     # permute each null gene's counts within stratum (independently per gene)
    for j in range(NN): Xd[idx, j] = Xd[rng.permutation(idx), j]
Xa = sp.hstack([X, sp.csr_matrix(Xd)]).tocsr(); nullcols = np.arange(X.shape[1], X.shape[1] + NN)
gall = np.concatenate([genes, np.array([f'NULL{i}' for i in range(NN)])])

def stratum_demean(M):
    M = np.asarray(M, dtype=np.float64); out = M.copy()
    cnt = np.bincount(strata);
    for j in range(M.shape[1]):
        out[:, j] -= (np.bincount(strata, M[:, j]) / cnt)[strata]
    return out

def null_tests(theta, Y, lib):
    """Partial correlation of theta (n x K) with Y (n x NN), projecting out strata (group means) and log lib."""
    T = stratum_demean(theta); Z = stratum_demean(Y); c = stratum_demean(np.log(lib)[:, None])[:, 0]
    c = c / np.linalg.norm(c); T -= np.outer(c, c @ T); Z -= np.outer(c, c @ Z)
    n_eff = len(c) - strata.max() - 1 - 1
    T /= (np.linalg.norm(T, axis=0) + 1e-12); Z /= (np.linalg.norm(Z, axis=0) + 1e-12)
    r = T.T @ Z; t = r * np.sqrt((n_eff - 2) / np.clip(1 - r ** 2, 1e-12, None))
    return 2 * stats.t.sf(np.abs(t), n_eff - 2)

DIAG = [None]
def run(fit_X, enc_X, test_X, nulls_in_model=True):
    t = ad.AnnData(fit_X[:, :X.shape[1]].astype(np.float32)); t.var_names = genes
    sc.pp.filter_genes(t, min_cells=50); sc.pp.normalize_total(t, target_sum=1e4); sc.pp.log1p(t)
    sc.pp.highly_variable_genes(t, n_top_genes=2000, flavor='seurat')
    hv = np.where(np.isin(genes, t.var_names[t.var.highly_variable]))[0]
    cols = np.concatenate([hv, nullcols]) if nulls_in_model else hv                                   # modeled genes: HVGs + all null genes
    if os.environ.get('NULL_MODEL') == 'nmf':     # baseline check: sklearn NMF on log-normalized counts, usage = transform of the encode-part counts
        from sklearn.decomposition import NMF
        nm = NMF(K, init='nndsvda', random_state=seed, max_iter=300).fit(lognorm(fit_X[:, cols]).toarray())
        th = nm.transform(lognorm(enc_X[:, cols]).toarray()); th = th / (th.sum(1, keepdims=True) + 1e-12)
    elif os.environ.get('NULL_MODEL') in ('ldvae', 'scetm'):   # baseline checks with the settings of code/e3_baselines.py
        import torch, scvi
        nmodel = os.environ['NULL_MODEL']; gn = np.array([f'g{i}' for i in range(len(cols))])
        def mk(M):
            b = ad.AnnData(sp.csr_matrix(M).astype(np.float32)); b.var_names = gn; return b
        atr, ate = mk(fit_X[:, cols]), mk(enc_X[:, cols])
        if nmodel == 'ldvae':
            scvi.settings.seed = seed; torch.manual_seed(seed)
            scvi.model.LinearSCVI.setup_anndata(atr)
            m = scvi.model.LinearSCVI(atr, n_latent=K); m.train(max_epochs=60, accelerator='cpu', plan_kwargs=dict(lr=1e-3))
            th = m.get_latent_representation(ate)
        else:
            from scETM import scETM as SC, UnsupervisedTrainer
            torch.manual_seed(seed); np.random.seed(seed); atr.obs['batch_indices'] = 0; ate.obs['batch_indices'] = 0
            m = SC(atr.n_vars, 1, n_topics=K, trainable_gene_emb_dim=128, device=torch.device('cpu'))
            UnsupervisedTrainer(m, atr, test_ratio=0., seed=seed, batch_size=2000).train(n_epochs=100, eval=False, save_model_ckpt=False, n_samplers=1)
            m.get_all_embeddings_and_nll(ate, inplace=True); th = np.asarray(ate.obsm['theta']); th = th / th.sum(1, keepdims=True)
    else:
        m = fit_ntm(fit_X[:, cols], K, batch=ctcode, epochs=60, bs=256, seed=seed)
        th = encode_theta(m, enc_X[:, cols])
    lib = np.asarray(test_X.sum(1)).ravel().astype(float); lib[lib == 0] = 1
    Y = np.log1p(1e4 * test_X[:, nullcols].toarray() / lib[:, None])   # normalized by the cell's full library
    ths = th.copy()      # diagnostic: usage shuffled among cells within stratum -> exact null for the test itself
    for idx in np.split(order, bounds): ths[idx] = ths[rng.permutation(idx)]
    DIAG[0] = float((null_tests(ths, Y, lib) < .05).mean())
    return null_tests(th, Y, lib)

tr, te = split_counts(Xa, 0.5, seed)
tr3, rest = split_counts(Xa, 1 / 3, seed); enc3, tst3 = split_counts(rest, 0.5, seed + 1)   # three equal parts: fit / encode usage / test gene
rr = estimate_size(Xa, ctcode)    # per-gene NB size (method of moments by cell type, as in code/nb_size.py); nuisance, estimated on all counts
tr3b, restb = split_counts(Xa, 1 / 3, seed, size=rr); enc3b, tst3b = split_counts(restb, 0.5, seed + 1, size=rr)
ALL = dict(split_hvonly=(tr, te, te, False), split3bb_hvonly=(tr3b, enc3b, tst3b, False), naive=(Xa, Xa, Xa), split=(tr, te, te), split3=(tr3, enc3, tst3), split3bb=(tr3b, enc3b, tst3b))
MODES = sys.argv[3].split(',') if len(sys.argv) > 3 else list(ALL)    # 3rd arg: subset of modes; output file gets suffix when given
SUF = ('' if len(sys.argv) <= 3 else '_' + sys.argv[3].replace(',', '_')) + ('_' + os.environ['NULL_MODEL'] if os.environ.get('NULL_MODEL') else '')
out = dict(seed=seed, n_null=NN, K=K)
for name, args in {k: ALL[k] for k in MODES}.items():
    P = run(*args); out[name] = dict(shuffled_theta_frac_p05=DIAG[0], frac_p05=float((P < .05).mean()), frac_p01=float((P < .01).mean()), n=int(P.size))
    print(name, out[name], flush=True)
json.dump(out, open(f'results/e3_nullgenes_seed{seed}{SUF}.json', 'w'), indent=1)
from runlog import log_run
log_run(seed, [f'results/e3_nullgenes_seed{seed}{SUF}.json'], dict(n_null=NN, K=K, ntm_seed=seed, split_seed=seed, null_seed=1000 + seed))

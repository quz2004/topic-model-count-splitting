"""E2 baseline: LDVAE (linear-decoder scVI) on Kang, same count-split protocol, K=20.
Programs = latent dims; gene loadings from the linear decoder. Held-out usage = latent mean on the test half.
usage: python code/e2_scvi.py SEED"""
import sys, json, numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, scanpy as sc, torch
from scipy import stats
import scvi
sys.path.insert(0, 'code')
from nbtm import split_counts
seed = int(sys.argv[1]); K = 20
scvi.settings.seed = seed; torch.manual_seed(seed)
a = ad.read_h5ad('data/kang.h5ad'); X = sp.csr_matrix(a.X.astype(np.int64)); obs = a.obs.reset_index(drop=True)
genes = np.array(a.var_names); donor = obs.ind.astype('category').cat.codes.values
cond = (obs.stim == 'stim').values.astype(int); ctype = obs.cell.astype(str).values
hall = json.load(open('data/hallmark_ifn.json')); ISG = set(hall['Interferon Alpha Response']) | set(hall['Interferon Gamma Response'])
tr, te = split_counts(X, 0.5, seed)
t = ad.AnnData(tr.astype(np.float32)); t.var_names = genes
sc.pp.filter_genes(t, min_cells=30); sc.pp.normalize_total(t, target_sum=1e4); sc.pp.log1p(t)
sc.pp.highly_variable_genes(t, n_top_genes=2000, flavor='seurat')
hv = np.where(np.isin(genes, t.var_names[t.var.highly_variable]))[0]; g = genes[hv]; isg_mask = np.isin(g, list(ISG))
def mk(M):
    b = ad.AnnData(sp.csr_matrix(M[:, hv]).astype(np.float32)); b.var_names = g; return b
atr, ate = mk(tr), mk(te)
scvi.model.LinearSCVI.setup_anndata(atr)
m = scvi.model.LinearSCVI(atr, n_latent=K); m.train(max_epochs=100, accelerator='cpu', plan_kwargs=dict(lr=1e-3))
L = m.get_loadings().values.T  # K x genes
Z = m.get_latent_representation(ate)
def enrich(H, top=50):
    out = []
    for k in range(H.shape[0]):
        idx = np.argsort(-H[k])[:top]; out.append(stats.hypergeom.sf(isg_mask[idx].sum() - 1, len(g), isg_mask.sum(), top))
    return np.array(out)
mkm = (ctype == 'CD14+ Monocytes')
def mono(W, k):
    df = pd.DataFrame({'v': W[mkm, k], 'd': donor[mkm], 'c': cond[mkm]}); mm = df.groupby(['d', 'c']).v.mean().unstack()
    diff = (mm[1] - mm[0]).dropna(); return float(diff.mean() / (df.v.std() + 1e-12)), float(stats.ttest_1samp(diff, 0)[1])
best = None
for sgn in (1, -1):  # latent sign is arbitrary: test both orientations of each dim
    pe = enrich(sgn * L)
    for k in range(K):
        if best is None or pe[k] < best[0]: best = (pe[k], k, sgn)
pe, k, sgn = best; es, p = mono(sgn * Z, k)
res = dict(seed=seed, ldvae=dict(isg_enrich_p=float(pe), mono_effect_sd=es, mono_p=p))
json.dump(res, open(f'results/e2_scvi_seed{seed}.json', 'w'), indent=1); print(res)

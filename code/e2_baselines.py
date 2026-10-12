"""E2 baselines on Kang with the same count-split protocol: NMF (log-norm) and sklearn LDA (counts), K=20.
usage: python code/e2_baselines.py SEED"""
import sys, json, numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, scanpy as sc
from scipy import stats
from sklearn.decomposition import NMF, LatentDirichletAllocation
sys.path.insert(0, 'code')
from nbtm import split_counts, lognorm
seed = int(sys.argv[1]); K = 20
a = ad.read_h5ad('data/kang.h5ad'); X = sp.csr_matrix(a.X.astype(np.int64)); obs = a.obs.reset_index(drop=True)
genes = np.array(a.var_names); donor = obs.ind.astype('category').cat.codes.values
cond = (obs.stim == 'stim').values.astype(int); ctype = obs.cell.astype(str).values
hall = json.load(open('data/hallmark_ifn.json')); ISG = set(hall['Interferon Alpha Response']) | set(hall['Interferon Gamma Response'])
tr, te = split_counts(X, 0.5, seed)
t = ad.AnnData(tr.astype(np.float32)); t.var_names = genes
sc.pp.filter_genes(t, min_cells=30); sc.pp.normalize_total(t, target_sum=1e4); sc.pp.log1p(t)
sc.pp.highly_variable_genes(t, n_top_genes=2000, flavor='seurat')
hv = np.where(np.isin(genes, t.var_names[t.var.highly_variable]))[0]; g = genes[hv]; isg_mask = np.isin(g, list(ISG))
trh, teh = tr[:, hv], te[:, hv]

def enrich(H, top=50):
    out = []
    for k in range(H.shape[0]):
        w = H[k] / (H.mean(0) + 1e-12); idx = np.argsort(-w)[:top]
        out.append(stats.hypergeom.sf(isg_mask[idx].sum() - 1, len(g), isg_mask.sum(), top))
    return np.array(out)

def mono_effect(W, k):
    mk = (ctype == 'CD14+ Monocytes')
    df = pd.DataFrame({'v': W[mk, k], 'd': donor[mk], 'c': cond[mk]}); m = df.groupby(['d', 'c']).v.mean().unstack()
    diff = (m[1] - m[0]).dropna(); sd = np.std(df.v) + 1e-12
    return float(diff.mean() / sd), float(stats.ttest_1samp(diff, 0)[1])

res = dict(seed=seed)
# NMF on train log-norm; usage on test half by non-negative projection with fixed components
Ltr = lognorm(trh).toarray(); Lte = lognorm(teh).toarray()
nmf = NMF(K, init='nndsvda', random_state=seed, max_iter=400).fit(Ltr)
Wte = nmf.transform(Lte); Wte = Wte / (Wte.sum(1, keepdims=True) + 1e-12); Hn = nmf.components_
pe = enrich(Hn); k = int(np.argmin(pe)); es, p = mono_effect(Wte, k)
res['nmf'] = dict(isg_enrich_p=float(pe[k]), mono_effect_sd=es, mono_p=p); np.save(f'results/e2_nmfH_seed{seed}.npy', dict(H=Hn, genes=g), allow_pickle=True)
# sklearn LDA on train counts
lda = LatentDirichletAllocation(K, learning_method='online', random_state=seed, max_iter=20, batch_size=1024).fit(trh)
Tte = lda.transform(teh); Hl = lda.components_ / lda.components_.sum(1, keepdims=True)
pe = enrich(Hl); k = int(np.argmin(pe)); es, p = mono_effect(Tte, k)
res['lda'] = dict(isg_enrich_p=float(pe[k]), mono_effect_sd=es, mono_p=p); np.save(f'results/e2_ldaH_seed{seed}.npy', dict(H=Hl, genes=g), allow_pickle=True)
json.dump(res, open(f'results/e2_base_seed{seed}.json', 'w'), indent=1); print(res)

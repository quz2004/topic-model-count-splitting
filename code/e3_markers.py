"""E3 markers: naive (fit + test on all counts) vs count-split (fit on train half, test on held-out half) gene-level
tests of the label-free ISG topic, within one cell type, K=30, cell-type intercepts. Mirrors the E2 marker analysis.
usage: python code/e3_markers.py SEED [split3]  -> results/e3_markers_seed{SEED}.json; 'split3' = beta-binomial three-way split (fit / encode usage / test gene),
-> results/e3_markers_split3_seed{SEED}.json
Covariates: log library size and donor indicators. Cell-level tests (liberal by construction); the comparison is split vs naive."""
import sys, json, numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, scanpy as sc
from scipy import stats
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, 'code')
from nbtm import *
from runlog import log_run
seed = int(sys.argv[1]); K = 30; CTS = ['classical monocyte', 'CD4-positive, alpha-beta T cell']
a = ad.read_h5ad('data/perez_sub.h5ad'); X = sp.csr_matrix(a.X).astype(np.int64)
obs = a.obs.reset_index(drop=True); genes = np.array(a.var_names)
donor = obs.donor_id.astype(str).values; ctype = obs.cell_type.astype(str).values
hall = json.load(open('data/hallmark_ifn.json')); ISG = set(hall['Interferon Alpha Response']) | set(hall['Interferon Gamma Response'])
bt = pd.Categorical(ctype).codes

def hvg(Xc):
    t = ad.AnnData(Xc.astype(np.float32)); t.var_names = genes
    sc.pp.filter_genes(t, min_cells=50); sc.pp.normalize_total(t, target_sum=1e4); sc.pp.log1p(t)
    sc.pp.highly_variable_genes(t, n_top_genes=2000, flavor='seurat')
    return np.where(np.isin(genes, t.var_names[t.var.highly_variable]))[0]
def enrich(beta, isg_mask, G, top=50):
    out = []
    for k in range(beta.shape[0]):
        w = beta[k] / (beta.mean(0) + 1e-12); idx = np.argsort(-w)[:top]
        out.append(stats.hypergeom.sf(isg_mask[idx].sum() - 1, G, isg_mask.sum(), top))
    return np.array(out)
def covmat(Xs, dn):
    lib = np.asarray(Xs.sum(1)).ravel(); D = pd.get_dummies(dn, drop_first=True, dtype=float).values
    return np.column_stack([np.log(np.maximum(lib, 1)), D])

S3 = len(sys.argv) > 2 and sys.argv[2] == 'split3'
if S3:
    rr = np.load('data/perez_sub_nbsize.npy'); tr, rest = split_counts(X, 1 / 3, seed, size=rr); enc, te = split_counts(rest, 0.5, seed + 1, size=rr)
else: tr, te = split_counts(X, 0.5, seed); enc = te
res = dict(seed=seed, K=K, split_seed=seed, ntm_seed=seed)
for name in (['split3'] if S3 else ['split', 'naive']):
    fitX, testX = (tr, te) if name != 'naive' else (X, X)
    hv = hvg(fitX); g = genes[hv]; isg_mask = np.isin(g, list(ISG))
    m = fit_ntm(fitX[:, hv], K, batch=bt, epochs=60, bs=256, seed=seed)
    th = encode_theta(m, (enc if name != 'naive' else X)[:, hv]); pe = enrich(topic_gene(m), isg_mask, len(g)); k = int(np.argmin(pe))
    res[name] = dict(k_isg=k, enrich_p=float(pe[k]), n_genes=len(g), n_isg=int(isg_mask.sum()), by_ct={})
    for c in CTS:
        mk = ctype == c; Xs = testX[mk][:, hv]
        r, P = gene_topic_corr_test(th[mk][:, [k]], lognorm(Xs), covmat(Xs, donor[mk])); r, P = r[0], P[0]
        q = multipletests(P, method='fdr_bh')[1]; sig = (q < .05) & (r > 0); top = np.argsort(P)[:100]
        res[name]['by_ct'][c] = dict(n_cells=int(mk.sum()), n_markers=int(sig.sum()), isg_in_markers=int(isg_mask[sig].sum()),
            frac_nonISG_sig=float((q[~isg_mask] < .05).mean()), frac_ISG_sig=float((q[isg_mask] < .05).mean()),
            median_p_nonISG=float(np.median(P[~isg_mask])), precision_top100=float(isg_mask[top].mean()))
    print(name, json.dumps(res[name]), flush=True)
json.dump(res, open(f'results/e3_markers_{'split3_' if S3 else ''}seed{seed}.json', 'w'), indent=1)
log_run(seed, [f'results/e3_markers_{'split3_' if S3 else ''}seed{seed}.json'], dict(K=K, split_seed=seed, ntm_seed=seed, cell_types=CTS))

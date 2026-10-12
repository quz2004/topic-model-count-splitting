"""E2: Kang IFN-beta PBMC. Count-split neural topic model vs naive reuse.
usage: python code/e2_kang.py SEED [K]
"""
import sys, json, numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, scanpy as sc
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ARI, normalized_mutual_info_score as NMI
from statsmodels.stats.multitest import multipletests
sys.path.insert(0, 'code')
from nbtm import *

seed = int(sys.argv[1]); K = int(sys.argv[2]) if len(sys.argv) > 2 else 20
EPOCHS = 80
a = ad.read_h5ad('data/kang.h5ad')
X = sp.csr_matrix(a.X.astype(np.int64)); obs = a.obs.reset_index(drop=True)
genes = np.array(a.var_names)
donor = obs.ind.astype('category').cat.codes.values
cond = (obs.stim == 'stim').values.astype(int)
ctype = obs.cell.astype(str).values
hall = json.load(open('data/hallmark_ifn.json'))
ISG = set(hall['Interferon Alpha Response']) | set(hall['Interferon Gamma Response'])

# ---- split, then ALL data-dependent choices (HVGs, topics) use the train half only
tr, te = split_counts(X, 0.5, seed)
t = ad.AnnData(tr.astype(np.float32)); t.var_names = genes
sc.pp.filter_genes(t, min_cells=30)
sc.pp.normalize_total(t, target_sum=1e4); sc.pp.log1p(t)
sc.pp.highly_variable_genes(t, n_top_genes=2000, flavor='seurat')
hv = np.where(np.isin(genes, t.var_names[t.var.highly_variable]))[0]
g = genes[hv]; isg_mask = np.isin(g, list(ISG))
Xh, trh, teh = X[:, hv], tr[:, hv], te[:, hv]

def cov_mat(Xs):
    l = np.log1p(np.asarray(Xs.sum(1)).ravel()); l = (l - l.mean()) / l.std()
    D = np.eye(donor.max() + 1)[donor][:, 1:]
    C = np.eye(len(set(ctype)))[pd.Categorical(ctype).codes][:, 1:]
    return np.column_stack([l, l ** 2, D, C])

def donor_effect(theta):
    """paired (stim - ctrl) difference in mean topic usage per donor; one-sample t across donors."""
    df = pd.DataFrame(theta); df['d'] = donor; df['c'] = cond
    m = df.groupby(['d', 'c']).mean()
    diff = (m.xs(1, level='c') - m.xs(0, level='c')).values            # donors x K
    tt, p = stats.ttest_1samp(diff, 0, axis=0)
    return diff.mean(0), tt, p

def donor_effect_sub(theta, mask):
    df = pd.DataFrame(theta[mask]); df['d'] = donor[mask]; df['c'] = cond[mask]
    cnt = df.groupby(['d', 'c']).size().unstack(fill_value=0)
    ok = cnt.index[(cnt >= 10).all(1)]
    m = df.groupby(['d', 'c']).mean()
    diff = (m.xs(1, level='c') - m.xs(0, level='c')).loc[ok].values
    tt, p = stats.ttest_1samp(diff, 0, axis=0)
    return diff.mean(0), tt, p

def isg_enrichment(beta, top=50):
    out = []
    for k in range(beta.shape[0]):
        w = beta[k] / (beta.mean(0) + 1e-12)                           # specificity vs average topic
        idx = np.argsort(-w)[:top]
        hit = isg_mask[idx].sum()
        out.append(stats.hypergeom.sf(hit - 1, len(g), isg_mask.sum(), top))
    return np.array(out)

res = dict(seed=seed, K=K, n_hvg=len(hv))
# ================= split model (train half) =================
m_s = fit_ntm(trh, K, batch=donor, epochs=EPOCHS, bs=256, seed=seed)
th_tr = encode_theta(m_s, trh); th_te = encode_theta(m_s, teh)
beta_s = topic_gene(m_s)
# ================= naive model (all counts) =================
m_n = fit_ntm(Xh, K, batch=donor, epochs=EPOCHS, bs=256, seed=seed)
th_full = encode_theta(m_n, Xh); beta_n = topic_gene(m_n)

for name, th_sel, th_test, beta in [('split', th_tr, th_te, beta_s), ('naive', th_full, th_full, beta_n)]:
    d_sel, t_sel, _ = donor_effect(th_sel)
    k_star = int(np.argmax(np.abs(t_sel)))                              # topic selected on selection data
    d_te, t_te, p_te = donor_effect(th_test)
    pe = isg_enrichment(beta); k_isg = int(np.argmin(pe))
    res[name] = dict(k_star=k_star, effect=float(d_te[k_star]), t=float(t_te[k_star]), p=float(p_te[k_star]),
                     k_isg=k_isg, isg_enrich_p=float(pe[k_isg]), same_topic=bool(k_star == k_isg),
                     isg_topic_effect=float(d_te[k_isg]), isg_topic_p=float(p_te[k_isg]),
                     n_topics_sig_bonf=int((p_te < 0.05 / K).sum()))
    # cell-type-specific paired donor effect of the ISG topic (biologically meaningful: response within cell type)
    ct_eff = {}
    for c in ['CD14+ Monocytes', 'CD4 T cells', 'B cells', 'NK cells', 'CD8 T cells', 'FCGR3A+ Monocytes']:
        mk = ctype == c
        dd, tt, pp = donor_effect_sub(th_test[:, [k_isg]], mk)
        ct_eff[c] = dict(effect=float(dd[0]), p=float(pp[0]))
    res[name]['isg_topic_by_celltype'] = ct_eff
    k_star_sel = k_star; k_star = k_isg                                  # markers analysed for the label-free ISG topic
    # ---- gene-level markers of the selected topic (cell-level partial correlation, covariates: depth, donor, cell type)
    if name == 'split':
        Y = lognorm(teh); C = cov_mat(teh)
    else:
        Y = lognorm(Xh); C = cov_mat(Xh)
    r, P = gene_topic_corr_test(th_test[:, [k_star]], Y, C)
    r, P = r[0], P[0]
    q = multipletests(P, method='fdr_bh')[1]
    sig = (q < 0.05) & (r > 0)
    ranks = np.argsort(P)[:100]
    k_star = k_star_sel
    res[name].update(n_markers=int(sig.sum()), isg_in_markers=int(isg_mask[sig].sum()),
                     precision_top100=float(isg_mask[ranks].mean()),
                     median_p_nonISG=float(np.median(P[~isg_mask])), frac_nonISG_sig=float((q[~isg_mask] < .05).mean()))

# ---- clustering sanity (author cell types), embeddings from the full data with the train-half model
Z_ntm = encode_theta(m_s, Xh)
Yf = lognorm(Xh).toarray(); Z_pca = PCA(30, random_state=seed).fit_transform(Yf - Yf.mean(0))
lab = pd.Categorical(ctype).codes; kk = len(set(ctype))
for nm, Z in [('ntm_theta', Z_ntm), ('pca30', Z_pca)]:
    cl = KMeans(kk, n_init=5, random_state=seed).fit_predict(Z)
    res[f'cluster_{nm}'] = dict(ARI=float(ARI(lab, cl)), NMI=float(NMI(lab, cl)))

# ---- save per-seed artefacts for stability/figures
np.savez_compressed(f'results/e2_seed{seed}.npz', beta_s=beta_s, beta_n=beta_n, genes=g,
                    th_te=th_te.astype(np.float32), th_full=th_full.astype(np.float32))
json.dump(res, open(f'results/e2_seed{seed}.json', 'w'), indent=1)
print(json.dumps(res, indent=1))

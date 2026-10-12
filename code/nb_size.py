"""Per-gene NB size r_g by method of moments, conditional on library size and cell type:
mu_ig = n_i * p_{g,c(i)}; 1/r_g = sum_i[(x-mu)^2 - mu] / sum_i mu^2 (clipped to r in [R_MIN, R_MAX])."""
import numpy as np, scipy.sparse as sp
R_MIN, R_MAX = 0.5, 100.0
def estimate_size(X, groups):
    X = sp.csr_matrix(X).astype(np.float64); n = np.asarray(X.sum(1)).ravel(); G = X.shape[1]
    num = np.zeros(G); den = np.zeros(G)
    for c in np.unique(groups):
        m = groups == c; Xc = X[m]; nc = n[m]
        p = np.asarray(Xc.sum(0)).ravel() / nc.sum()            # gene proportions in group
        Xc2 = Xc.multiply(Xc).sum(0).A1; Xn = np.asarray(Xc.T @ nc).ravel()   # sum x^2 ; sum x*n
        mu_sum = p * nc.sum(); mu2_sum = p**2 * (nc**2).sum()
        # sum (x-mu)^2 = sum x^2 - 2 p sum x n + p^2 sum n^2
        sq = Xc2 - 2 * p * Xn + mu2_sum
        num += sq - mu_sum; den += mu2_sum
    inv_r = np.maximum(num / np.maximum(den, 1e-12), 1.0 / R_MAX)
    return np.clip(1.0 / inv_r, R_MIN, R_MAX)
if __name__ == '__main__':
    import anndata as ad, pandas as pd, sys
    a = ad.read_h5ad(sys.argv[1]); X = sp.csr_matrix(a.X)
    grp = np.array(a.obs[sys.argv[2]].astype(object).fillna('NA').astype(str)) if len(sys.argv) > 2 else np.zeros(X.shape[0], int)
    r = estimate_size(X, grp); keep = np.asarray(X.sum(0)).ravel() > 0.01 * X.shape[0]
    print('genes', X.shape[1], 'expressed in >1% cells', keep.sum(), 'size quantiles (expressed genes) 5/25/50/75/95%:', np.round(np.quantile(r[keep], [.05, .25, .5, .75, .95]), 2))
    np.save(sys.argv[1].replace('.h5ad', '_nbsize.npy'), r)

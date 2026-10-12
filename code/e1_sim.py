"""E1: calibration of gene-topic association tests, naive reuse vs count splitting (simulation)."""
import sys, json, numpy as np, scipy.sparse as sp, pandas as pd
sys.path.insert(0, 'code')
from nbtm import *

def simulate(n=3000, G_inf=100, G_null=200, K=4, depth=1.0, nb_size=None, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.dirichlet(np.ones(K) * 0.3, n)
    G = G_inf + G_null
    b = rng.normal(-1.0 + np.log(depth), 0.8, G)
    B = np.zeros((K, G))
    for k in range(K):                                  # each informative gene loads on one topic, +/- balanced
        idx = np.arange(k, G_inf, K)
        B[k, idx] = rng.choice([1.2, -1.2], len(idx))
    s = rng.lognormal(0, 0.3, n)
    e = np.exp(b + theta @ B)
    e[:, :G_inf] *= np.exp(b[:G_inf]).sum() / e[:, :G_inf].sum(1, keepdims=True)  # constant informative total => library size independent of theta
    mu = s[:, None] * e
    if nb_size is None: X = rng.poisson(mu)
    else: X = rng.poisson(mu * rng.gamma(nb_size, 1 / nb_size, mu.shape))  # NB(mean mu, size nb_size)
    return sp.csr_matrix(X), theta, np.arange(G) < G_inf, B

def cov(X):
    l = np.log1p(np.asarray(X.sum(1)).ravel()); l = (l - l.mean()) / l.std()
    return np.column_stack([l, l ** 2])

def run(scn, seed, K=4, eps=0.5):
    X, th, inf, B = simulate(K=K, seed=seed, **scn)
    r_size = scn.get('nb_size')
    res = {}
    # naive: fit and test on all counts
    m = fit_ntm(X, K, epochs=40, seed=seed)
    T = encode_theta(m, X); _, P = gene_topic_corr_test(T, lognorm(X), cov(X)); res['naive'] = P
    # count split (binomial) and NB-aware (beta-binomial)
    for name, size in [('split_binom', None), ('split_nb', r_size)]:
        if name == 'split_nb' and r_size is None: continue
        tr, te = split_counts(X, eps, seed, size)
        m = fit_ntm(tr, K, epochs=40, seed=seed)
        T = encode_theta(m, tr); _, P = gene_topic_corr_test(T, lognorm(te), cov(te)); res[name] = P
    out = []
    for k, P in res.items():
        Pn = P[:, ~inf].ravel(); Pi = P[:, inf]
        out.append(dict(method=k, fpr05=(Pn < .05).mean(), fpr01=(Pn < .01).mean(),
                        power05=(Pi.min(0) < .05 / K).mean(), **scn, eps=eps, seed=seed))
    return out

if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[2] == 'eps':
        rows = []
        for eps in [0.2, 0.35, 0.5, 0.65, 0.8]:
            for seed in range(int(sys.argv[1])):
                rows += run(dict(depth=1.0, nb_size=2.0), seed, eps=eps); print(eps, seed, flush=True)
        df = pd.DataFrame(rows); df.to_csv('results/e1_eps_raw.csv', index=False)
        print(df.groupby(['eps', 'method'])[['fpr05', 'fpr01', 'power05']].mean().round(3)); sys.exit()
    scns = [dict(depth=1.0), dict(depth=0.3), dict(depth=1.0, nb_size=2.0), dict(depth=0.3, nb_size=2.0), dict(depth=1.0, nb_size=0.5)]
    rows = []
    for scn in scns:
        for seed in range(int(sys.argv[1]) if len(sys.argv) > 1 else 5):
            rows += run(scn, seed); print(scn, seed, flush=True)
    df = pd.DataFrame(rows); df.to_csv('results/e1_sim_raw.csv', index=False)
    print(df.groupby(['depth', 'nb_size', 'method'], dropna=False)[['fpr05', 'fpr01', 'power05']].mean().round(3))

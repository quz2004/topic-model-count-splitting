"""Aggregate nested-CV AUROC over seeds: mean ± SD of per-seed means, and paired topics-vs-PCA / topics-vs-props differences."""
import json, glob, numpy as np
R = [json.load(open(f)) for f in sorted(glob.glob('results/e3_nested_seed*_ct.json'))]
names = list(R[0]['auc_mean']); print(f'{len(R)} seeds')
M = {n: np.array([r['auc_mean'][n] for r in R]) for n in names}
for n in names: print(f'{n:18s} {M[n].mean():.3f} ± {M[n].std(ddof=1):.3f}')
for a, b in [('topics_pb', 'pca_pb'), ('topics_pb', 'celltype_props')]:
    d = M[a] - M[b]; print(f'{a} - {b}: {d.mean():+.3f} ± {d.std(ddof=1):.3f}; {a} ahead in {(d>0).sum()}/{len(d)} seeds')

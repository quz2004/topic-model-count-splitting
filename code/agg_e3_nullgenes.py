"""Aggregate E3b null-gene runs. usage: python code/agg_e3_nullgenes.py  -> prints and writes results/e3_nullgenes_all_summary.txt"""
import json, glob, re, numpy as np
G = {}
for f in sorted(glob.glob('results/e3_nullgenes_seed*.json')):
    d = json.load(open(f)); model = 'nmf' if f.endswith('_nmf.json') else 'ldvae' if f.endswith('_ldvae.json') else 'scetm' if f.endswith('_scetm.json') else 'ntm'
    for k in ('naive', 'split', 'split3', 'split3bb', 'split_hvonly', 'split3bb_hvonly'):
        if k in d: G.setdefault((model, k), {})[d['seed']] = (d[k]['frac_p05'], d[k]['frac_p01'])
out = ['E3b null genes (300 permuted real genes per seed, K=30); rejection among null gene x topic pairs']
for (m, k), v in G.items():
    a = np.array(list(v.values())); out.append('%-6s %-16s seeds=%d p05 %.3f±%.3f  p01 %.3f±%.3f' % (m, k, len(v), a[:, 0].mean(), a[:, 0].std(ddof=1) if len(v) > 1 else 0, a[:, 1].mean(), a[:, 1].std(ddof=1) if len(v) > 1 else 0))
open('results/e3_nullgenes_all_summary.txt', 'w').write('\n'.join(out)); print('\n'.join(out))

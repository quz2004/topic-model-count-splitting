"""Fig: rejection at nominal 0.05 for permuted null genes, models x protocols. usage: python code/fig_nullgenes.py"""
import json, glob, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
suf = {'NTM': None, 'NMF': 'nmf', 'LDVAE': 'ldvae', 'scETM': 'scetm'}
prot = [('naive', 'naive', '#c0392b'), ('split', 'two-way split', '#e67e22'), ('split3bb', 'three-way beta-binomial', '#2980b9')]
V = {}
for f in glob.glob('results/e3_nullgenes_seed*.json'):
    m = next((n for n, s in suf.items() if s and f.endswith('_%s.json' % s)), 'NTM')
    d = json.load(open(f))
    for k, _, _ in prot:
        if k in d: V.setdefault((m, k), []).append(d[k]['frac_p05'])
fig, ax = plt.subplots(figsize=(7, 3.6)); w = 0.26
for j, (k, lab, c) in enumerate(prot):
    for i, m in enumerate(suf):
        v = np.array(V.get((m, k), []))
        if not len(v): continue
        x = i + (j - 1) * w
        ax.bar(x, v.mean(), w * 0.9, color=c, label=lab if i == 0 else None)
        ax.errorbar(x, v.mean(), v.std(ddof=1) if len(v) > 1 else 0, color='k', capsize=2, lw=.8)
        ax.plot(np.full(len(v), x), v, '.', color='k', ms=3, alpha=.5)
ax.axhline(0.05, ls='--', color='gray', lw=1); ax.text(3.45, 0.057, 'nominal 0.05', ha='right', fontsize=7, color='gray')
ax.set_xticks(range(len(suf))); ax.set_xticklabels(['%s\n(n=%d seeds)' % (m, len(V.get((m, 'naive'), []))) for m in suf], fontsize=8)
ax.set_ylabel('rejection rate at p<0.05\n(permuted null genes)'); ax.legend(fontsize=7, frameon=False); plt.tight_layout()
plt.savefig('paper/figs/fig_nullgenes.png', dpi=200); print({k: (len(v), round(float(np.mean(v)), 3)) for k, v in sorted(V.items())})

"""Aggregate permutation-null rejection rates for baselines (results/e3_basenull_{nmf,ldvae}_seed*.json)."""
import json, glob, numpy as np
for m in ['nmf', 'ldvae', 'scetm', 'cnmf']:
    r = [json.load(open(f)) for f in sorted(glob.glob(f'results/e3_basenull_{m}_seed*.json'))]
    if not r: continue
    a = np.array([x['frac_p05'] for x in r]); b = np.array([x['frac_p01'] for x in r])
    print(m, len(r), 'seeds; tests/seed', r[0]['n'], f'p<.05 {a.mean():.4f}±{a.std(ddof=1):.4f}  p<.01 {b.mean():.4f}±{b.std(ddof=1):.4f}')

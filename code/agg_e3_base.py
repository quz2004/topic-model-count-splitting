"""Aggregate E3 baselines (NMF, LDVAE) vs NTM (plain, ct): AUROC and label-free ISG program SLE shift."""
import json, glob, numpy as np
def load(pat, f): return [f(json.load(open(x))) for x in sorted(glob.glob(pat))]
def ms(v): v = np.array(v); return f'{v.mean():.3f} ± {v.std(ddof=1):.3f} (n={len(v)})'
for m in ['nmf', 'ldvae', 'scetm', 'cnmf']:
    R = [json.load(open(x)) for x in sorted(glob.glob(f'results/e3_base_{m}_seed*.json'))]
    if not R: continue
    print(m, 'AUROC topics_pb', ms([r['auc_mean']['topics_pb'] for r in R]), '| +cov', ms([r['auc_mean']['topics+cov'] for r in R]))
    print('  ISG enrich median p %.2g' % np.median([r['isg_topic']['enrich_p'] for r in R]))
    for c in ['classical monocyte', 'CD4-positive, alpha-beta T cell', 'B cell']:
        e = [r['isg_by_ct'][c]['eff_sd'] for r in R if c in r['isg_by_ct']]; p = [r['isg_by_ct'][c]['p'] for r in R if c in r['isg_by_ct']]
        print(f'  {c}: shift {np.mean(e):.2f} SD; p<.05 in {sum(np.array(p)<.05)}/{len(p)}')
for tag, suf in [('NTM plain', ''), ('NTM ct', '_ct')]:
    R = [json.load(open(x)) for x in sorted(glob.glob(f'results/e3_seed[0-9]{suf}.json'))]
    print(tag, 'AUROC topics_pb', ms([r['auc_mean']['topics_pb'] for r in R]), '| pca_pb', ms([r['auc_mean']['pca_pb'] for r in R]))

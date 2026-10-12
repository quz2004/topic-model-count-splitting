import json, glob, numpy as np
R = [json.load(open(f)) for f in sorted(glob.glob('results/e3_markers_seed[0-9].json'))]; print(len(R), 'seeds')
for c in R[0]['split']['by_ct']:
    print('==', c, f"(cells {R[0]['split']['by_ct'][c]['n_cells']})")
    for k in ['n_markers', 'isg_in_markers', 'frac_ISG_sig', 'frac_nonISG_sig', 'median_p_nonISG', 'precision_top100']:
        s = np.array([r['split']['by_ct'][c][k] for r in R]); n = np.array([r['naive']['by_ct'][c][k] for r in R])
        print(f'{k:18s} split {s.mean():.3f} ± {s.std(ddof=1):.3f}   naive {n.mean():.3f} ± {n.std(ddof=1):.3f}   naive>split in {(n>s).sum()}/{len(R)}')
for nm in ['split', 'naive']: print(nm, 'ISG-topic enrichment median p', np.median([r[nm]['enrich_p'] for r in R]))

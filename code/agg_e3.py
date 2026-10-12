import json, glob, sys, numpy as np, pandas as pd
SUF = '_' + sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != 'plain' else ''
from scipy import stats
R=[json.load(open(f)) for f in sorted(glob.glob(f'results/e3_seed[0-9]{SUF}.json'))]
print('seeds',len(R),'donors',R[0]['n_donors'],'cases',R[0]['n_cases'])
A=pd.DataFrame([r['auc_mean'] for r in R]); print('AUROC (mean over 25 CV folds per seed), mean +/- sd across seeds'); print(pd.DataFrame({'mean':A.mean(),'sd':A.std()}).round(3).to_string())
F=[json.load(open(f)) for f in sorted(glob.glob(f'results/e3_auc_folds_seed[0-9]{SUF}.json'))]
def paired(a,b):
    d=np.concatenate([np.array(f[a])-np.array(f[b]) for f in F]); sm=np.array([np.mean(np.array(f[a])-np.array(f[b])) for f in F])
    return round(d.mean(),4), round(sm.std(),4), int((sm>0).sum())
for a,b in [('topics_pb','celltype_props'),('topics_pb','covariates'),('topics_pb','pca_pb'),('topics+props+cov','props+cov'),('topics+props+cov','pca+props+cov'),('topics+cov','props+cov')]:
    print('delta AUROC',a,'-',b,paired(a,b),'(mean fold diff, sd of seed means, #seeds>0)')
print('ISG topic enrichment p (median)',np.median([r['isg_topic']['enrich_p'] for r in R]))
ct=pd.DataFrame({c:[r['isg_by_ct'][c]['p'] for r in R] for c in R[0]['isg_by_ct']}); eff=pd.DataFrame({c:[r['isg_by_ct'][c]['eff_sd'] for r in R] for c in R[0]['isg_by_ct']})
print('ISG topic: disease effect (SD units) mean / frac seeds p<.05'); print(pd.DataFrame({'eff':eff.mean(),'frac_p05':(ct<.05).mean(),'median_p':ct.median()}).round(4).to_string())
print('n sig q<.05 of tests',np.mean([r['n_sig_q05'] for r in R]),np.mean([r['n_tests'] for r in R]))
print('perm null frac p<.05',np.mean([r['null_perm']['frac_p05'] for r in R]).round(4),'p<.01',np.mean([r['null_perm']['frac_p01'] for r in R]).round(4))

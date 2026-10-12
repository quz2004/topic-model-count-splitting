import json, glob, sys, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SUF='_ct'
R=[json.load(open(f)) for f in sorted(glob.glob(f'results/e3_seed[0-9]{SUF}.json'))]
cts=[c for c in R[0]['isg_by_ct'] if c!='lymphocyte']
short={'CD4-positive, alpha-beta T cell':'CD4 T','CD8-positive, alpha-beta T cell':'CD8 T','classical monocyte':'cMono','non-classical monocyte':'ncMono','B cell':'B','natural killer cell':'NK','conventional dendritic cell':'cDC'}
fig,ax=plt.subplots(1,2,figsize=(11,4))
eff=pd.DataFrame({c:[r['isg_by_ct'][c]['eff_sd'] for r in R] for c in cts}); p=pd.DataFrame({c:[r['isg_by_ct'][c]['p'] for r in R] for c in cts})
ax[0].bar(range(len(cts)),eff.mean(),yerr=eff.std(),color='#8e44ad',capsize=3)
for i,c in enumerate(cts): ax[0].text(i,eff[c].mean()+eff[c].std()+0.03,f"{(p[c]<.05).mean()*10:.0f}/10",ha='center',fontsize=8)
ax[0].set_xticks(range(len(cts))); ax[0].set_xticklabels([short.get(c,c) for c in cts],rotation=30,ha='right'); ax[0].set_ylabel('SLE-control shift (SD units)')
ax[0].set_title('A. IFN program (labels: seeds with p<0.05)')
A=pd.DataFrame([r['auc_mean'] for r in R]); order=['covariates','celltype_props','topics_pb','pca_pb']; names=['covariates\n(age,sex,ethnicity)','cell-type\nproportions','topic usage\n(NTM)','PCA\npseudobulk']
ax[1].bar(range(4),A[order].mean(),yerr=A[order].std(),color=['#95a5a6','#e67e22','#27ae60','#2980b9'],capsize=3); ax[1].set_ylim(0.6,1.0)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(names,fontsize=8); ax[1].set_ylabel('donor-level AUROC (5x5 CV)'); ax[1].set_title('B. SLE vs control prediction, 261 donors')
plt.tight_layout(); plt.savefig('paper/figs/fig3_lupus.png',dpi=200)

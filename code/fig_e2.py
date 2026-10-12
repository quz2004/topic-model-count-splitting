import json, numpy as np, pandas as pd, anndata as ad, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
a=ad.read_h5ad('data/kang.h5ad'); obs=a.obs.reset_index(drop=True)
r=json.load(open('results/e2_seed0.json')); d=np.load('results/e2_seed0.npz',allow_pickle=True)
k=r['split']['k_isg']; th=d['th_te'][:,k]
df=pd.DataFrame(dict(v=th,d=obs.ind.values,c=obs.stim.values,t=obs.cell.values))
cts=['CD14+ Monocytes','FCGR3A+ Monocytes','B cells','CD4 T cells','CD8 T cells','NK cells']
fig,ax=plt.subplots(1,3,figsize=(13,3.8))
for i,c in enumerate(cts):
    m=df[df.t==c].groupby(['d','c']).v.mean().unstack()
    for _,row in m.iterrows(): ax[0].plot([i-0.15,i+0.15],[row['ctrl'],row['stim']],'-',color='#bbb',lw=.8)
    ax[0].plot(np.full(len(m),i-0.15),m['ctrl'],'o',color='#2980b9',ms=3); ax[0].plot(np.full(len(m),i+0.15),m['stim'],'o',color='#c0392b',ms=3)
ax[0].set_xticks(range(6)); ax[0].set_xticklabels([c.replace(' cells','').replace('Monocytes','Mono') for c in cts],rotation=25,ha='right',fontsize=8)
ax[0].set_ylabel('mean IFN-program usage (held-out counts)'); ax[0].set_title('A. Paired donors: ctrl (blue) -> IFN-b (red), seed 0')
b=np.load('results/e2_seed0.npz',allow_pickle=True)['beta_s']; g=d['genes']; w=b[k]/b.mean(0); top=np.argsort(-w)[:15]
ax[1].barh(range(15),w[top][::-1],color='#8e44ad'); ax[1].set_yticks(range(15)); ax[1].set_yticklabels(g[top][::-1],fontsize=7); ax[1].set_title('B. Top specific genes of IFN program'); ax[1].set_xlabel('specificity (beta_k / mean beta)')
rows=[json.load(open(f'results/e2_seed{s}.json')) for s in range(10)]
v=[[x['split']['frac_nonISG_sig'] for x in rows],[x['naive']['frac_nonISG_sig'] for x in rows]]
ax[2].boxplot(v,tick_labels=['count-split\n(held-out)','naive reuse']); ax[2].set_ylabel('fraction of non-ISG genes called\n(FDR 5%, program markers)'); ax[2].set_title('C. Marker calls, 10 seeds')
plt.tight_layout(); plt.savefig('paper/figs/fig2_kang.png',dpi=200)

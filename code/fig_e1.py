import pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
d=pd.read_csv('results/e1_sim_raw.csv'); d['nb_size']=d.nb_size.fillna(np.inf)
d['scenario']=d.apply(lambda r:('Poisson' if np.isinf(r.nb_size) else f'NB r={r.nb_size:g}')+f', depth {r.depth:g}',axis=1)
g=d.groupby(['scenario','method']).agg(fpr=('fpr05','mean'),sd=('fpr05','std'),pw=('power05','mean'),n=('seed','count')).reset_index()
g.to_csv('results/e1_table.csv',index=False); print(g.round(3).to_string())
order=['Poisson, depth 1','Poisson, depth 0.3','NB r=2, depth 1','NB r=2, depth 0.3','NB r=0.5, depth 1']
fig,ax=plt.subplots(1,2,figsize=(11,3.8))
cols={'naive':'#c0392b','split_binom':'#2980b9','split_nb':'#27ae60'}; lab={'naive':'naive reuse','split_binom':'count split (binomial)','split_nb':'count split (NB-aware)'}
for i,m in enumerate(['naive','split_binom','split_nb']):
    s=g[g.method==m].set_index('scenario').reindex(order)
    ax[0].bar(np.arange(5)+(i-1)*0.27,s.fpr,0.27,yerr=s.sd,color=cols[m],label=lab[m],capsize=2)
ax[0].axhline(0.05,color='k',ls='--',lw=1); ax[0].set_xticks(range(5)); ax[0].set_xticklabels(order,rotation=30,ha='right',fontsize=8)
ax[0].set_ylabel('false-positive rate at nominal 0.05'); ax[0].set_title('A. Null genes vs topic usage'); ax[0].legend(fontsize=8)
e=pd.read_csv('results/e1_eps_raw.csv').groupby(['eps','method'])[['fpr05','power05']].mean().reset_index()
for m in ['split_binom','split_nb']:
    s=e[e.method==m]; ax[1].plot(s.eps,s.power05,'o-',color=cols[m],label=lab[m]+' power')
ax[1].set_xlabel('training fraction eps'); ax[1].set_ylabel('power (informative genes)'); ax[1].set_title('B. Power vs split fraction (NB r=2)'); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig('paper/figs/fig1_calibration.png',dpi=200)

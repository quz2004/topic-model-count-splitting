import json, glob, numpy as np, pandas as pd
rows=[]
for f in sorted(glob.glob('results/e2_seed*.json')):
    r=json.load(open(f))
    for m in ['split','naive']:
        d=dict(seed=r['seed'],method=m,**{k:v for k,v in r[m].items() if k!='isg_topic_by_celltype'})
        for c,v in r[m]['isg_topic_by_celltype'].items(): d['ct_p_'+c]=v['p']; d['ct_eff_'+c]=v['effect']
        rows.append(d)
    for c in ['ntm_theta','pca30']:
        rows.append(dict(seed=r['seed'],method='cluster_'+c,**r['cluster_'+c]))
df=pd.DataFrame(rows); df.to_csv('results/e2_summary_rows.csv',index=False)
pd.set_option('display.width',200)
num=df.drop(columns=['same_topic'],errors='ignore')
print(num.groupby('method').agg(['mean','std']).round(4).T.dropna(how='all').to_string())
# ISG topic recovery across seeds
for m in ['split','naive']:
    d=df[df.method==m]; print(m,'n seeds',len(d),'k_star==k_isg',d.same_topic.mean(),'ISG topic p<.05',(d.isg_topic_p<.05).mean())
# topic stability across seeds on the common gene set (HVGs differ per split)
fs=sorted(glob.glob('results/e2_seed*.npz'))
D=[np.load(f,allow_pickle=True) for f in fs]
common=sorted(set.intersection(*[set(d['genes']) for d in D])); print('common genes',len(common))
def lg(d,key):
    ix=[list(d['genes']).index(g) for g in common]; b=d[key][:,ix]; b=b/b.sum(1,keepdims=True)
    l=np.log(b+1e-9); return l-l.mean(1,keepdims=True)
import itertools
K=D[0]['beta_s'].shape[0]; st=[]; ig=[]
isg_k=[json.load(open(f.replace('.npz','.json')))['split']['k_isg'] for f in fs]
for i,j in itertools.combinations(range(len(D)),2):
    c=np.corrcoef(lg(D[i],'beta_s'),lg(D[j],'beta_s'))[:K,K:]
    st.append(c.max(1).mean()); ig.append(c[isg_k[i]].max())
print('mean best-match topic corr (all topics) over seed pairs',np.round(np.mean(st),3),'| ISG topic best-match',np.round(np.mean(ig),3),'min',np.round(np.min(ig),3))

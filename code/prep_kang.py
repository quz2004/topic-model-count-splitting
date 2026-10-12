import pandas as pd, scipy.io, scipy.sparse as sp, anndata as ad, gzip, numpy as np
d='data/'
genes=pd.read_csv(d+'GSE96583_batch2.genes.tsv.gz',sep='\t',header=None)
meta=pd.read_csv(d+'GSE96583_batch2.total.tsne.df.tsv.gz',sep='\t')
parts=[]
for gsm,lab in [('GSM2560248_2.1','ctrl'),('GSM2560249_2.2','stim')]:
    X=scipy.io.mmread(f'{d}kang_raw/{gsm}.mtx.gz').T.tocsr()
    bc=pd.read_csv(f'{d}kang_raw/{gsm.split("_")[0]}_barcodes.tsv.gz',header=None)[0].values
    a=ad.AnnData(X.astype(np.float32)); a.obs_names=bc; a.obs['batchlab']=lab; parts.append(a)
a=ad.concat(parts)
dup=a.obs_names.duplicated(keep=False)  # 313 barcodes shared by ctrl/stim are ambiguous in metadata: drop
a=a[~dup].copy()
a.var_names=genes[1].values if genes.shape[1]>1 else genes[0].values
a.var_names_make_unique()
print(a, meta.columns.tolist(), meta.shape)
a=a[a.obs_names.isin(meta.index)].copy()
a.obs=a.obs.join(meta)
print(a.obs.head()); 
for c in ['cell','stim','ind','multiplets']:
    if c in a.obs: print(a.obs[c].value_counts().head(12))
a=a[a.obs['multiplets']=='singlet'].copy()
assert a.obs_names.is_unique and (a.obs.batchlab==a.obs.stim).all()
from scipy import sparse; a.X=sparse.csr_matrix(a.X)
a.write_h5ad(d+'kang.h5ad'); print(a)

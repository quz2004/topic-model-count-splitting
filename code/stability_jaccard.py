import glob, itertools, numpy as np
def load(pat, key):
    out=[]
    for f in sorted(glob.glob(pat)):
        d=np.load(f,allow_pickle=True)
        if f.endswith('.npy'): d=d.item(); out.append((d['H'],d['genes']))
        else: out.append((d[key],d['genes']))
    return out
def tops(H,genes,common,top=50):
    ix=[list(genes).index(g) for g in common]; H=H[:,ix]; w=H/(H.mean(0)+1e-12)   # specificity, as used for ISG labeling
    return [set(np.argsort(-w[k])[:top]) for k in range(H.shape[0])]
res={}
for name,pat,key in [('ntm','results/e2_seed*.npz','beta_s'),('nmf','results/e2_nmfH_seed*.npy',None),('lda','results/e2_ldaH_seed*.npy',None)]:
    L=load(pat,key); common=sorted(set.intersection(*[set(g) for _,g in L]))
    T=[tops(H,g,common) for H,g in L]; vals=[]
    for a,b in itertools.combinations(range(len(T)),2):
        J=np.array([[len(x&y)/len(x|y) for y in T[b]] for x in T[a]]); vals.append(J.max(1).mean())
    res[name]=(round(float(np.mean(vals)),3),len(common)); 
print('mean best-match Jaccard of top-50 specific genes (common genes)',res)

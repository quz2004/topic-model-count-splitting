"""E3 nested CV: donor-level disease prediction with topics refit INSIDE each outer training fold (no transduction).
usage: python code/e3_nested.py SEED [K] [MODE]    (MODE 'ct' default)
Per outer fold (5 stratified donor folds): count split (seed) -> gene filter/HVG + NTM on train-donor train-half cells only
(labels unused; test donors' cells never seen) -> frozen encoder on held-out half of all cells -> donor x cell-type pseudobulk
-> L2 logistic regression fit on train donors, AUROC on test donors. PCA pseudobulk baseline: PCA fit on train donors only.
"""
import sys, json, numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad, scanpy as sc
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
sys.path.insert(0, 'code')
from nbtm import *

seed = int(sys.argv[1]); K = int(sys.argv[2]) if len(sys.argv) > 2 else 30
MODE = sys.argv[3] if len(sys.argv) > 3 else 'ct'
a = ad.read_h5ad('data/perez_sub.h5ad')
X = sp.csr_matrix(a.X); X = X.astype(np.int64)
obs = a.obs.reset_index(drop=True); genes = np.array(a.var_names)
donor = obs.donor_id.astype(str).values; ctype = obs.cell_type.astype(str).values
top_ct = pd.Series(ctype).value_counts().index[:8].tolist()
dd = obs.groupby('donor_id', observed=True).agg(case=('disease', lambda s: int((s.astype(str) != 'normal').iloc[0])),
     sex=('sex', 'first'), eth=('self_reported_ethnicity', 'first'), stage=('development_stage', 'first'))
dd['age'] = dd.stage.astype(str).str.extract(r'(\d+)')[0].astype(float); dd['age'] = dd.age.fillna(dd.age.median())
dd['male'] = (dd.sex.astype(str) == 'male').astype(int)
E = pd.get_dummies(dd.eth.astype(str), drop_first=True, dtype=float); E = E.loc[:, E.sum() >= 10]
cov = pd.concat([dd[['age', 'male']].astype(float), E], axis=1); cov = (cov - cov.mean()) / (cov.std() + 1e-9); cov = cov.loc[:, cov.std() > 0]
dd.index = dd.index.astype(str); donors = dd.index.tolist(); y = dd.case.values
dpos = pd.Series(np.arange(len(donors)), index=donors)

tr, te = split_counts(X, 0.5, seed)
bt_all = pd.Categorical(ctype).codes if MODE == 'ct' else None

def pseudobulk(M, min_cells=10):
    cols = []
    for c in top_ct:
        mk = ctype == c
        df = pd.DataFrame(M[mk]); df['d'] = donor[mk]
        cnt = df.groupby('d').size(); mean = df.groupby('d').mean(); mean[cnt.reindex(mean.index) < min_cells] = np.nan
        cols.append(mean.reindex(donors).add_prefix(c + '|'))
    return pd.concat(cols, axis=1).values
cnt = pd.crosstab(pd.Series(donor, name='d'), pd.Series(ctype, name='c')).reindex(donors).fillna(0)[top_ct]
props = cnt.div(cnt.sum(1), axis=0).values

def clf():
    return make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(), LogisticRegressionCV(Cs=8, cv=3, penalty='l2', max_iter=2000, scoring='roc_auc'))

skf = StratifiedKFold(5, shuffle=True, random_state=seed)
res = dict(seed=seed, K=K, mode=MODE, folds=[]); out = {k: [] for k in ['covariates', 'celltype_props', 'topics_pb', 'pca_pb', 'topics+props+cov']}
for f, (tri_d, tei_d) in enumerate(skf.split(np.zeros(len(donors)), y)):
    trd = set(np.array(donors)[tri_d]); cells_tr = np.where(np.isin(donor, list(trd)))[0]
    t = ad.AnnData(tr[cells_tr].astype(np.float32)); t.var_names = genes
    sc.pp.filter_genes(t, min_cells=50); sc.pp.normalize_total(t, target_sum=1e4); sc.pp.log1p(t)
    sc.pp.highly_variable_genes(t, n_top_genes=2000, flavor='seurat')
    hv = np.where(np.isin(genes, t.var_names[t.var.highly_variable]))[0]
    m = fit_ntm(tr[cells_tr][:, hv], K, batch=None if bt_all is None else bt_all[cells_tr], epochs=60, bs=256, seed=seed * 10 + f)
    th = encode_theta(m, te[:, hv]); PBt = pseudobulk(th)
    Yf = lognorm(te[:, hv]).toarray(); mu = Yf[cells_tr].mean(0)
    pca = PCA(30, random_state=seed).fit(Yf[cells_tr] - mu); Z = pca.transform(Yf - mu); del Yf
    PBp = pseudobulk(Z)
    F = {'covariates': cov.values, 'celltype_props': props, 'topics_pb': PBt, 'pca_pb': PBp,
         'topics+props+cov': np.hstack([PBt, props, cov.values])}
    r = {}
    for name, M in F.items():
        c_ = clf().fit(M[tri_d], y[tri_d]); r[name] = float(roc_auc_score(y[tei_d], c_.predict_proba(M[tei_d])[:, 1])); out[name].append(r[name])
    res['folds'].append(r); print(f, r, flush=True)
res['auc_mean'] = {k: float(np.mean(v)) for k, v in out.items()}
json.dump(res, open(f'results/e3_nested_seed{seed}_{MODE}.json', 'w'), indent=1); print(json.dumps(res['auc_mean']))
from runlog import log_run
log_run(seed, [f'results/e3_nested_seed{seed}_{MODE}.json'], dict(K=K, mode=MODE, split_seed=seed, fold_seed=seed, ntm_seeds=[seed * 10 + f for f in range(5)], pca_seed=seed))

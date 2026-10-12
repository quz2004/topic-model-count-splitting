"""Fetch a donor-stratified subsample of the Stephenson et al. COVID-19 PBMC dataset via CELLxGENE Census (run with .venv_census).
usage: .venv_census/bin/python code/fetch_covid.py [N_PER_DONOR=400] -> data/covid_sub.h5ad (COVID-19 vs normal donors only)"""
import cellxgene_census, numpy as np, pandas as pd, anndata as ad, sys
DS = 'c7775e88-49bf-4ba2-a03b-93f00447c958'; N = int(sys.argv[1]) if len(sys.argv) > 1 else 400
sys.path.insert(0, 'code'); from covid_cell_types import coarse
with cellxgene_census.open_soma(census_version='2025-11-08') as c:
    obs = cellxgene_census.get_obs(c, 'homo_sapiens', value_filter=f"dataset_id == '{DS}' and is_primary_data == True and disease in ['COVID-19','normal']",
        column_names=['soma_joinid', 'donor_id', 'disease', 'cell_type', 'sex', 'development_stage', 'self_reported_ethnicity'])
    print(len(obs), obs.donor_id.nunique(), obs.groupby('disease', observed=True).donor_id.nunique().to_dict())
    keep = obs.groupby('donor_id', group_keys=False, observed=True).apply(lambda d: d.sample(min(len(d), N), random_state=0))
    a = cellxgene_census.get_anndata(c, 'homo_sapiens', obs_coords=keep.soma_joinid.tolist(), measurement_name='RNA', X_name='raw',
        obs_column_names=['donor_id', 'disease', 'cell_type', 'sex', 'development_stage', 'self_reported_ethnicity'], var_column_names=['feature_name'])
a.obs['cell_type_fine'] = a.obs.cell_type.astype(str); a.obs['cell_type'] = a.obs.cell_type_fine.map(coarse)
a.var_names = a.var.feature_name.values; a.var_names_make_unique()
a.write_h5ad('data/covid_sub.h5ad'); print(a); print(a.obs.cell_type.value_counts())
print(pd.crosstab(a.obs.cell_type_fine, a.obs.cell_type).stack().loc[lambda s: s > 0].groupby(level=1).head(3))

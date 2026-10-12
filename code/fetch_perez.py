"""Fetch a donor-stratified subsample of the Perez et al. 2022 lupus PBMC dataset via CELLxGENE Census."""
import cellxgene_census, numpy as np, pandas as pd, anndata as ad, sys
DS = '218acb0f-9f2f-4f76-b90b-15a4b7c7f629'
N_PER_DONOR = int(sys.argv[1]) if len(sys.argv) > 1 else 400
with cellxgene_census.open_soma(census_version='stable') as c:
    obs = cellxgene_census.get_obs(c, 'homo_sapiens', value_filter=f"dataset_id == '{DS}'",
                                   column_names=['soma_joinid', 'donor_id', 'disease', 'cell_type', 'sex', 'development_stage', 'self_reported_ethnicity', 'tissue'])
    print(len(obs), obs.donor_id.nunique(), obs.disease.value_counts().to_dict())
    rng = np.random.default_rng(0)
    keep = obs.groupby('donor_id', group_keys=False).apply(lambda d: d.sample(min(len(d), N_PER_DONOR), random_state=0))
    ids = keep.soma_joinid.tolist()
    a = cellxgene_census.get_anndata(c, 'homo_sapiens', obs_coords=ids, measurement_name='RNA', X_name='raw',
                                     obs_column_names=['donor_id', 'disease', 'cell_type', 'sex', 'development_stage', 'self_reported_ethnicity'],
                                     var_column_names=['feature_name'])
a.var_names = a.var.feature_name.values; a.var_names_make_unique()
a.write_h5ad('data/perez_sub.h5ad'); print(a)

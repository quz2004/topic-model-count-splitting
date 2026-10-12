# Reducing double-dipping in single-cell topic models with three-way count splitting

Code, figures and result tables for the manuscript "Reducing double-dipping in single-cell topic models with three-way count splitting".

Author and contact: Zheng Qu, Fei Tian College Northern Campus, Middletown, NY 10940; zqu001@mt.feitian.edu. License: MIT.

## Layout
- `code/` analysis code: `nbtm.py` (topic model, count splitting, tests), `e1_sim.py` (simulation), `e2_kang.py` and `e2_baselines.py` (IFN-β PBMCs), `e3_*.py` (lupus cohort, baselines, nested CV, null genes, markers), `agg_*.py` (aggregation), `fig_*.py` (figures), `fetch_*.py` and `prep_kang.py` (data download and preparation).
- `scripts/` shell helpers for the cNMF and scETM baselines.
- `figures/` manuscript figures (PNG).
- `results/` per-seed result tables (CSV/JSON) and aggregate summaries (`*summary*.txt`).

## Data (not included)
- Kang et al. 2018: GEO GSE96583 (batch 2).
- Perez et al. 2022 (lupus): CELLxGENE Census dataset 218acb0f-9f2f-4f76-b90b-15a4b7c7f629 (release 2025-11-08).
- Stephenson et al. 2021 (COVID-19): CELLxGENE Census dataset c7775e88-49bf-4ba2-a03b-93f00447c958.

Scripts expect to be run from the repository root with data under `data/` and write to `results/`. Seeds 0-9.

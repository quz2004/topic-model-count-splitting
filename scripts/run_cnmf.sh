#!/bin/bash
cd "$(dirname "$0")/.." && mkdir -p logs
for s in "$@"; do
  [ -f results/e3_basenull_cnmf_seed$s.json ] || .venv/bin/python code/e3_baselines.py $s cnmf null > logs/cnmf_null_$s.log 2>&1
  [ -f results/e3_base_cnmf_seed$s.json ] || .venv/bin/python code/e3_baselines.py $s cnmf > logs/cnmf_$s.log 2>&1
done

#!/bin/bash
cd "$(dirname "$0")/.." && mkdir -p logs
for s in "$@"; do
  [ -f results/e3_base_scetm_seed$s.json ] || .venv/bin/python code/e3_baselines.py $s scetm > logs/scetm_$s.log 2>&1
  [ -f results/e3_basenull_scetm_seed$s.json ] || .venv/bin/python code/e3_baselines.py $s scetm null > logs/scetm_null_$s.log 2>&1
done

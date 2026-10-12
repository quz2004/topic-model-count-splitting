#!/bin/bash
# Per-seed cNMF status: done / RUNNING / NOT-RUNNING (missing and nobody working on it)
cd "$(dirname "$0")/.."
for s in ${@:-0 1 2 3 4 5 6 7 8 9}; do
  n=results/e3_basenull_cnmf_seed$s.json; c=results/e3_base_cnmf_seed$s.json
  if [ -f $n ] && [ -f $c ]; then st=done
  elif pgrep -f "e3_baselines.py $s cnmf" >/dev/null; then st=RUNNING
  else st=NOT-RUNNING; fi
  echo "seed $s: null=$([ -f $n ] && echo y || echo n) cv=$([ -f $c ] && echo y || echo n) $st"
done

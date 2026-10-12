#!/bin/bash
# Usage: scripts/wait_jobs.sh PROC_REGEX EXPECTED_FILE...
# Waits until no worker process matching PROC_REGEX is alive (liveness, not file count), then
# verifies every expected file exists. Exit 0 = all present; exit 1 = some missing (listed),
# so unlaunched or crashed work is reported instead of waited on forever.
# Wrapper shells ("bash -c ...") and this script itself are ignored, so the regex cannot self-match.
re="$1"; shift
alive() { pgrep -fa "$re" | grep -v -e 'wait_jobs\.sh' -e 'bash -c' -e '^[0-9]* pgrep' | grep -q .; }
while alive; do sleep 60; done
miss=0; for f in "$@"; do [ -e "$f" ] || { echo "MISSING $f"; miss=1; }; done
[ $miss = 0 ] && echo "ALL PRESENT ($# files)"; exit $miss

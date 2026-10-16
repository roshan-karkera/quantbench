#!/usr/bin/env bash
set -euo pipefail
for p in fp16 int8 nf4; do
  echo "=== $p ==="
  python -m src.run_eval --precision "$p"
done
python -m src.report

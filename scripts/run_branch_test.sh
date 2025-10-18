#!/usr/bin/env bash
set -euo pipefail

python -m src.main --mode branch-test --dataset synthetic --steps 200 --batch_size 8 --seq_len 128


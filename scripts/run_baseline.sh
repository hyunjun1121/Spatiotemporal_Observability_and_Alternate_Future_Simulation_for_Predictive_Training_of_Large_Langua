#!/usr/bin/env bash
set -euo pipefail

python -m src.main --mode baseline --dataset synthetic --steps 400 --batch_size 8 --seq_len 128


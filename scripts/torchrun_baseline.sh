#!/usr/bin/env bash
set -euo pipefail

torchrun --nproc_per_node=2 -m src.main --mode baseline --dataset synthetic --steps 800 --batch_size 8 --seq_len 128


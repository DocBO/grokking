#!/usr/bin/env bash
set -euo pipefail

uv run python grokking/stats.py \
  --optimizers adamw ensemble-simple \
  --training_fractions 0.11 0.12 0.13 0.14 0.15 0.16 0.18 0.20 \
  --num_seeds 1 \
  --repeats 1 \
  --num_steps 100000 \
  --eval_interval 100 \
  --threshold 0.95 \
  --strict_threshold \
  --stop_at_threshold \
  --ensemble_size 4 \
  --temperature_heating 1.50 \
  --stall_window 5 \
  --max_temperature 0.02 \
  --min_temperature 1e-7 \
  --entropy_weight 0 \
  --time_cooling_rate 0 \
  --temperature_start 0.01 \
  --temperature_cooling 0.9 \
  --output_dir boundary_sweep_runs \
  "$@"

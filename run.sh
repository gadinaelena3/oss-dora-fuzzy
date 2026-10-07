#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python src/compute_metrics.py
python src/run_analysis.py
python src/robustness.py
python src/make_figures.py

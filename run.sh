#!/usr/bin/env bash
# Run the entire OSS longitudinal study end-to-end on the bundled sample.
#
# Usage:
#   bash run.sh
#
# To use live GitHub data instead of the bundled sample:
#   1. export GITHUB_TOKEN=ghp_xxx
#   2. python src/fetch_github.py
#   3. python src/compute_metrics.py
#   4. (then run.sh skips generate_sample and uses derived data)

set -euo pipefail
cd "$(dirname "$0")"

echo "============================================================"
echo " OSS Longitudinal Study — Fuzzy DSS validation"
echo "============================================================"

if [[ ! -f "data/derived/quarterly_metrics.csv" ]]; then
  echo
  echo "[1/3] Generating bundled sample dataset..."
  python src/generate_sample.py
else
  echo
  echo "[1/3] Skipping sample generation (live data found at"
  echo "      data/derived/quarterly_metrics.csv)"
fi

echo
echo "[2/3] Running fuzzy DSS over the dataset and computing statistics..."
python src/run_analysis.py

echo
echo "[3/3] Generating figures..."
python src/make_plots.py

echo
echo "============================================================"
echo " Done. Open results/REPORT.md for the executive summary."
echo "============================================================"

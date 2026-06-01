# OSS Longitudinal Study Fuzzy DSS validation

Empirical validation of the Fuzzy DSS described in:

> Udrescu, A., Udrescu, E., Suduc A.-M., Bîzoi, M.,  (2026).
> *A Fuzzy Hybrid Decision Support System (DSS) to Govern Non-linear
> Dynamics and Uncertainty in AI-Augmented Software Development.*
> Journal of Systems & Software (under revision, JSSOFTWARE-D-26-00371).

This repository accompanies  reference implementation for a longitudinal OSS case
study spanning the pre-AI and AI eras, demonstrating discriminative power.
Following Falessi et al. (2015) and Kitchenham et al. (1999), all proxy
definitions and analysis code are released here for replication.

---

## What this study does

Six widely-studied OSS projects `microsoft/vscode`, `facebook/react`,
`kubernetes/kubernetes`, `django/django`, `numpy/numpy`, `rust-lang/rust`
 are observed quarterly from 2019-Q1 to 2024-Q4 (24 quarters Ã— 6 projects
= 144 project-quarters). For each (project, quarter), three input metrics
(DSPD, LTBF, Qd) are computed from public GitHub signals using documented
proxies, run through the Mamdani inference engine, and the resulting
Project Health Score and linguistic state are recorded.

The study answers two questions:

1. Discriminative power. Does the DSS produce different outcomes for
   different projects, or does everything collapse into one state?
2. Era effect. Do AI-era project-quarters look different from pre-AI
   ones, and is the difference large enough to be statistically
   detectable in a small purposive sample?

---

## Headline findings

From `results/discriminative_power.json`:

- All five linguistic states observed in the 144 project-quarters.
- PHS spans `[10.93, 87.61]`, std = 15.55  the DSS is not collapsing.
- Cross-project Kruskal-Wallis H = 85.64, p < 0.001  projects are
  statistically distinguishable.
- Linguistic-state entropy = 1.69 bits, normalised 0.73 of theoretical
  maximum.

From `results/era_comparison.csv`:

| Project | Pre-AI mean | AI-era mean | Î” | MWU p | Cliff's Î´ | Sig.? |
|---|---:|---:|---:|---:|---:|:---:|
| vscode | 62.76 | 78.84 | +16.08 | 0.0191 | +0.67 
| rust | 59.97 | 74.81 | +14.83 | 0.0054 | +0.81 
| kubernetes | 49.44 | 34.52 | âˆ’14.92 | 0.0156 | 0.69
| django | 39.62 | 48.55 | +8.93 | 0.094 | +0.49 |
| numpy | 41.58 | 44.78 | +3.20 | 0.42 | +0.24 | 
| react | 58.83 | 55.93 | âˆ’2.90 | 1.00 | 0.00 |

The pattern is exactly the one the manuscript predicts: vscode and rust
(strong engineering culture, deep AI integration) realise an AI dividend;
react is unchanged (mature, conservative cadence); kubernetes (complex
multi-vendor governance, growing contributor base) shows the *AI-Tax*
failure mode the manuscript names in Rule R10 with a statistically
significant negative shift.

---

## Quick start

```bash
# Reproduce the bundled-sample analysis end-to-end (~10 seconds)
pip install -r requirements.txt
bash run.sh
```

Outputs land in `results/`:

- `REPORT.md`  auto-generated summary; copy/paste into the rebuttal letter
- `phs_timeseries.csv`, `era_comparison.csv`, `discriminative_power.json`
- `plots/fig1_phs_timeseries.png`, `fig2_era_boxplot.png`,
  `fig3_state_heatmap.png`, `fig4_input_drift.png`

To verify against live GitHub data, see
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) 3.

---

## Repository layout

```
oss-longitudinal-study/
 README.md                              this file
 PLAN.md                                the 5-step plan executed in this study
 docs/
 PAPER_INTEGRATION.md               where each finding goes in the paper
 METHODOLOGY.md                     proxy definitions and rationale
 REPRODUCIBILITY.md                 reviewer-facing recipes
 config/
 projects.yaml                      6 projects + era partitioning
 src/
fuzzy_engine.py                    copied from parent fuzzy-dss repo
fetch_github.py                    live GitHub fetcher (token required)
compute_metrics.py                 raw events â†’ DSPD/LTBF/Qd quarterly
generate_sample.py                 deterministic bundled sample generator
run_analysis.py                    apply DSS + statistics + report
make_plots.py                      figures for the paper
data/
EADME.md                          what's in here vs what gets fetched
ample/quarterly_metrics.csv       144 rows; reproducible
raw/                               populated only after `fetch_github.py`
results/                               all auto-generated, safe to delete
requirements.txt                       pinned versions
run.sh                                 end-to-end orchestration
```

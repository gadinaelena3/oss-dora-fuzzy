# OSS longitudinal feasibility study of the Fuzzy DSS

Replication package for Section 5, Appendices B and C and the Addendum of

> Udrescu, A., Udrescu, E., Suduc, A.-M., Bîzoi, M. (2026). *A Fuzzy Hybrid Decision Support System (DSS) to Govern Non-linear Dynamics and Uncertainty in AI-Augmented Software Development.* Journal of Systems and Software, under review (JSSOFTWARE-D-26-00371).

Six repositories (microsoft/vscode, facebook/react, kubernetes/kubernetes, django/django, numpy/numpy, rust-lang/rust), 24 quarters (2019-Q1 to 2024-Q4), 144 project-quarters.

## Reproduce

```bash
pip install -r requirements.txt
bash run.sh
```

| Step | Output |
|---|---|
| `python src/compute_metrics.py` | `data/derived/quarterly_metrics.csv`, `data/derived/calibration.json` (inputs; Addendum Table S2.7) |
| `python src/run_analysis.py` | `results/phs_timeseries.csv`, `results/era_comparison.csv`, `results/discriminative_power.json` (Tables X, XI, C.1, C.2) |
| `python src/robustness.py` | `results/robustness/` (Addendum, Tables S2.1 to S2.6 and S2.8, with `restored.csv` and `entropy_calibration.csv`) |
| `python src/diagnostics.py` | `results/diagnostics.json` (exact p-values, serial dependence, pairwise comparisons, output granularity, proxy correlations; Sections 5.2 to 5.4, 7.1 and 7.4) |
| `python src/make_figures.py` | `results/figures/` (Figs. 8 to 13, B.1 to B.12) |

Tested on Python 3.13 with the versions in `requirements.txt`.

## Data

`data/raw/{project}.jsonl` holds every merged pull request returned by the GitHub search for the study window: 110,481 records collected in October 2026 with `src/fetch_github.py` (GraphQL search, one window per project and month, windows above 1,000 results split automatically). `data/raw/completeness.csv` gives, per window, the count reported by the API and the count retrieved; they are equal in all 432 windows.

The search does not return pull requests whose author account has been deleted. `data/raw/search_coverage.csv` (from `src/check_search_coverage.py`) puts this at 0.09% to 0.82% per repository.

facebook/react has been transferred to react/react and is collected under that name.

To collect again, put a GitHub token in `GITHUB_TOKEN` or in a file named `.github_token` and run `python src/fetch_github.py`. The run can be interrupted and resumed.

## Method

- Bot-authored pull requests are excluded (615, 0.56%); `python src/compute_metrics.py --include-bots` keeps them.
- Scaling factors are anchored on the Pre-AI era: each maps the pooled Pre-AI median of its raw signal onto the human-baseline value of the paper (DSPD 12, LTBF 7, Qd 3). Result: k_DSPD = 12.18, k_LTBF = 7.22, k_Qd = 4.35.
- The DSPD proxy is the quarterly count of merged pull requests relative to the project's Pre-AI mean.
- Pull requests to non-default branches (7.0%) are included; `results/robustness/default_branch_only.csv` gives the analysis without them.
- `results/robustness/temporal_holdout.csv` calibrates on one Pre-AI year and tests against the other.

## Reported results

| | |
|---|---|
| State distribution | Sustainable 97, High Performance 20, At Risk 19, Critical Risk 5, Elite AI Maturity 3 |
| Kruskal-Wallis across projects | H = 78.23, p < 0.001 |
| No-rule fallback | 29.17% (42 of 144) |
| Significant era shift (Bonferroni, α′ = 0.0083) | microsoft/vscode only (δ = +1.000, p = 0.0002) |

The vscode shift coincides with a change in how that project uses pull requests from 2022-Q2 and is not interpreted as an effect of AI adoption.

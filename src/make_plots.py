"""
Generate the figures for the paper.

Outputs (all in `results/plots/`):
  fig1_phs_timeseries_<project>.{png,tiff}   Per-project PHS over time
  fig2_era_boxplot_<project>.{png,tiff}       Pre-AI / Transition / AI-era boxplot
  fig3_state_heatmap_<project>.{png,tiff}     Linguistic state over time
  fig4_input_drift_<project>.{png,tiff}       DSPD, LTBF, Qd drift over time

Aesthetic: scientific, ink-on-cream, no decorative chart-junk.
Resolution: 300 DPI. Formats: PNG + TIFF.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import yaml
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[1]
TS_PATH  = ROOT / "results" / "phs_timeseries.csv"
CONFIG   = ROOT / "config" / "projects.yaml"
PLOT_DIR = ROOT / "results" / "plots"

DPI = 300

# Editorial-scientific palette
INK    = "#1c1916"
INK2   = "#3a352e"
INK3   = "#6e6557"
PAPER  = "#f4efe6"
ACCENT = "#b1542a"
RULE   = "#d4cab7"

STATE_COLORS = {
    "Critical Risk":     "#8c2a1f",
    "At Risk":           "#b1542a",
    "Sustainable":       "#6e7d4a",
    "High Performance":  "#3f7050",
    "Elite AI Maturity": "#214d39",
}

PROJECT_ORDER = ["vscode", "react", "kubernetes", "django", "numpy", "rust"]

# Distinct colour per project for multi-line plots
PROJECT_COLOURS = dict(zip(
    PROJECT_ORDER,
    plt.cm.tab10(np.linspace(0, 0.6, len(PROJECT_ORDER))),
))

MILESTONES = [
    ("2021Q3", "Copilot beta"),
    ("2022Q4", "ChatGPT GA"),
    ("2023Q2", "GPT-4"),
]

YEAR_TICKS = ["2019Q1", "2020Q1", "2021Q1", "2022Q1", "2023Q1", "2024Q1"]
YEAR_LABELS = ["2019", "2020", "2021", "2022", "2023", "2024"]


def _setup_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK2)
    ax.spines["bottom"].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=RULE, linewidth=0.5, linestyle="-", alpha=0.6)
    ax.set_axisbelow(True)


def _quarter_to_x(q: str, all_qs: list[str]) -> int:
    return all_qs.index(q)


def _save(fig, stem: Path, facecolor=PAPER) -> None:
    for ext in ("png", "tiff"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=DPI,
                    facecolor=facecolor, bbox_inches="tight")


def _draw_milestones(ax, all_qs: list[str], top_y: float | None = None) -> None:
    for q, label in MILESTONES:
        mx = _quarter_to_x(q, all_qs)
        ax.axvline(mx, color=ACCENT, linewidth=0.8, linestyle="--", alpha=0.5, zorder=2)
        if top_y is not None:
            ax.text(mx + 0.2, top_y, label, fontsize=7, color=ACCENT,
                    rotation=90, va="top", ha="left")


def _set_year_xticks(ax, all_qs: list[str]) -> None:
    tick_idx = [_quarter_to_x(q, all_qs) for q in YEAR_TICKS]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(YEAR_LABELS, fontsize=8, color=INK2)
    ax.set_xlabel("Year", color=INK2, fontsize=10, labelpad=6)


# ---------------------------------------------------------------------------
# Figure 1 — PHS time series, one file per project
# ---------------------------------------------------------------------------
def fig_phs_timeseries(df: pd.DataFrame, all_qs: list[str], eras: dict) -> None:
    state_handles = [mpatches.Patch(color=c, label=s) for s, c in STATE_COLORS.items()]
    milestone_handle = plt.Line2D([0], [0], color=ACCENT, linestyle="--",
                                   linewidth=1, label="AI-tooling milestone")

    for project in PROJECT_ORDER:
        fig, ax = plt.subplots(figsize=(9, 5), facecolor=PAPER)
        ax.set_facecolor(PAPER)

        sub = df[df["project_id"] == project].sort_values("quarter")
        x = [_quarter_to_x(q, all_qs) for q in sub["quarter"]]
        y = sub["phs"].values

        for ymin, ymax, colour, alpha in [
            (0,  25,  "#8c2a1f", 0.06),
            (25, 50,  "#b1542a", 0.05),
            (50, 70,  "#6e7d4a", 0.05),
            (70, 100, "#214d39", 0.06),
        ]:
            ax.axhspan(ymin, ymax, color=colour, alpha=alpha, zorder=0)

        ax.plot(x, y, color=INK, linewidth=1.5, zorder=3)
        for xi, yi, state in zip(x, y, sub["linguistic_state"]):
            ax.scatter([xi], [yi], color=STATE_COLORS[state],
                       edgecolor=INK, linewidth=0.5, s=30, zorder=4)

        _draw_milestones(ax, all_qs, top_y=97)

        ax.set_ylim(0, 100)
        _set_year_xticks(ax, all_qs)
        ax.set_ylabel("Project Health Score (PHS)", color=INK2, fontsize=10, labelpad=6)
        ax.set_title(f"PHS over time — {project}",
                     fontsize=12, fontweight="bold", color=INK, loc="left", pad=8)
        _setup_axes(ax)

        ax.legend(handles=[milestone_handle] + state_handles,
                  loc="lower left", frameon=False, fontsize=8, ncol=2)

        plt.tight_layout()
        stem = PLOT_DIR / f"fig1_phs_timeseries_{project}"
        _save(fig, stem)
        plt.close()
        print(f"  -> fig1_phs_timeseries_{project}.png / .tiff")


# ---------------------------------------------------------------------------
# Figure 2 — Pre-AI / Transition / AI-era boxplots, one file per project
# ---------------------------------------------------------------------------
def fig_era_boxplot(df: pd.DataFrame, eras: dict) -> None:
    era_order  = ["pre_ai", "transition", "ai_era"]
    era_labels = {"pre_ai": "Pre-AI", "transition": "Transition", "ai_era": "AI-era"}
    box_colours = ["#d4cab7", "#d97f4e", "#214d39"]

    for project in PROJECT_ORDER:
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=PAPER)
        ax.set_facecolor(PAPER)

        data = []
        for era in era_order:
            sub = df[(df["project_id"] == project) &
                     (df["quarter"].isin(eras[era]))]
            data.append(sub["phs"].values if len(sub) else np.array([0.0]))

        bp = ax.boxplot(data, positions=[0, 1, 2], widths=0.55,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, linewidth=1.5),
                        whiskerprops=dict(color=INK2, linewidth=0.8),
                        capprops=dict(color=INK2, linewidth=0.8))

        for patch, colour in zip(bp["boxes"], box_colours):
            patch.set_facecolor(colour)
            patch.set_edgecolor(INK)
            patch.set_alpha(0.78)

        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels([era_labels[e] for e in era_order],
                            fontsize=10, color=INK2)
        ax.set_xlabel("Era", color=INK2, fontsize=10, labelpad=6)
        ax.set_ylabel("Project Health Score (PHS)", color=INK2, fontsize=10, labelpad=6)
        ax.set_ylim(0, 100)
        ax.set_title(f"PHS distribution by era — {project}",
                     fontsize=12, fontweight="bold", color=INK, loc="left", pad=8)
        _setup_axes(ax)

        legend_handles = [mpatches.Patch(color=c, label=era_labels[e])
                           for c, e in zip(box_colours, era_order)]
        ax.legend(handles=legend_handles, loc="upper right",
                  frameon=False, fontsize=9)

        plt.tight_layout()
        stem = PLOT_DIR / f"fig2_era_boxplot_{project}"
        _save(fig, stem)
        plt.close()
        print(f"  -> fig2_era_boxplot_{project}.png / .tiff")


# ---------------------------------------------------------------------------
# Figure 3 — State heatmap, one file per project
# ---------------------------------------------------------------------------
def fig_state_heatmap(df: pd.DataFrame, all_qs: list[str]) -> None:
    state_to_int = {
        "Critical Risk":      0,
        "At Risk":            1,
        "Sustainable":        2,
        "High Performance":   3,
        "Elite AI Maturity":  4,
    }
    cmap = ListedColormap([STATE_COLORS["Critical Risk"],
                            STATE_COLORS["At Risk"],
                            STATE_COLORS["Sustainable"],
                            STATE_COLORS["High Performance"],
                            STATE_COLORS["Elite AI Maturity"]])

    for project in PROJECT_ORDER:
        fig, ax = plt.subplots(figsize=(13, 1.8), facecolor=PAPER)
        ax.set_facecolor(PAPER)

        matrix = np.full((1, len(all_qs)), np.nan)
        sub = df[df["project_id"] == project]
        for _, row in sub.iterrows():
            j = all_qs.index(row["quarter"])
            matrix[0, j] = state_to_int.get(row["linguistic_state"], np.nan)

        im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=4,
                        interpolation="nearest")

        ax.set_yticks([0])
        ax.set_yticklabels([project], fontsize=10, color=INK2)
        ax.set_ylabel("Project", color=INK2, fontsize=10, labelpad=6)

        tick_idx = [_quarter_to_x(q, all_qs) for q in YEAR_TICKS]
        ax.set_xticks(tick_idx)
        ax.set_xticklabels(YEAR_LABELS, fontsize=9, color=INK2)
        ax.set_xlabel("Year", color=INK2, fontsize=10, labelpad=6)

        for q, _ in MILESTONES:
            mx = _quarter_to_x(q, all_qs)
            ax.axvline(mx - 0.5, color=PAPER, linewidth=2)
            ax.axvline(mx - 0.5, color=ACCENT, linewidth=1, linestyle="--")

        ax.set_title(f"Linguistic state over time — {project}",
                     fontsize=12, fontweight="bold", color=INK, loc="left", pad=8)

        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4],
                             shrink=0.85, pad=0.02, aspect=8)
        cbar.ax.set_yticklabels(list(state_to_int.keys()), fontsize=8)
        cbar.outline.set_visible(False)

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)

        plt.tight_layout()
        stem = PLOT_DIR / f"fig3_state_heatmap_{project}"
        _save(fig, stem)
        plt.close()
        print(f"  -> fig3_state_heatmap_{project}.png / .tiff")


# ---------------------------------------------------------------------------
# Figure 4 — Input drift, one file per project
# ---------------------------------------------------------------------------
def fig_input_drift(df: pd.DataFrame, all_qs: list[str]) -> None:
    vars_meta = [
        ("dspd", "DSPD (Volume)",       (0, None)),
        ("ltbf", "LTBF (Lead time, d)", (0, None)),
        ("qd",   "Qd (Viability)",       (0, None)),
    ]

    for project in PROJECT_ORDER:
        colour = PROJECT_COLOURS[project]
        fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True, facecolor=PAPER)

        sub = df[df["project_id"] == project].sort_values("quarter")
        x = [_quarter_to_x(q, all_qs) for q in sub["quarter"]]

        for ax, (var, ylabel, ylim) in zip(axes, vars_meta):
            ax.set_facecolor(PAPER)
            ax.plot(x, sub[var], color=colour, linewidth=1.5, alpha=0.9, label=project)
            for q, _ in MILESTONES:
                ax.axvline(_quarter_to_x(q, all_qs), color=ACCENT,
                            linewidth=0.7, linestyle="--", alpha=0.45)
            if ylim[0] is not None:
                ax.set_ylim(bottom=ylim[0])
            if ylim[1] is not None:
                ax.set_ylim(top=ylim[1])
            ax.set_ylabel(ylabel, color=INK2, fontsize=9, labelpad=6)
            _setup_axes(ax)

        # Milestone labels on top subplot
        for q, label in MILESTONES:
            mx = _quarter_to_x(q, all_qs)
            ylim_top = axes[0].get_ylim()[1]
            axes[0].text(mx + 0.2, ylim_top * 0.97, label, fontsize=7,
                          color=ACCENT, rotation=90, va="top", ha="left")

        _set_year_xticks(axes[-1], all_qs)
        axes[0].set_title(f"Input metric drift — {project}",
                           fontsize=12, fontweight="bold", color=INK,
                           loc="left", pad=8)

        plt.tight_layout()
        stem = PLOT_DIR / f"fig4_input_drift_{project}"
        _save(fig, stem)
        plt.close()
        print(f"  -> fig4_input_drift_{project}.png / .tiff")


# ---------------------------------------------------------------------------
def main() -> int:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(TS_PATH)

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    eras = {era: meta["quarters"] for era, meta in cfg["eras"].items()}

    all_qs = sorted(set(eras["pre_ai"] + eras["transition"] + eras["ai_era"]))

    print("Generating plots (300 DPI, PNG + TIFF, one file per project):")
    fig_phs_timeseries(df, all_qs, eras)
    fig_era_boxplot(df, eras)
    fig_state_heatmap(df, all_qs)
    fig_input_drift(df, all_qs)
    print(f"\nAll plots written to {PLOT_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
DPI = 300
STATES = ["Critical Risk", "At Risk", "Sustainable", "High Performance", "Elite AI Maturity"]
COLOR = dict(zip(STATES, ["#922B21", "#E67E22", "#F1C40F", "#2ECC71", "#16A085"]))
MARKER = dict(zip(STATES, ["v", "s", "o", "^", "D"]))
LETTER = dict(zip(STATES, ["C", "R", "S", "H", "E"]))
MILESTONES = [("2021Q2", "Copilot beta"), ("2022Q4", "ChatGPT GA"), ("2023Q1", "GPT-4")]
ERA_COLOR = {"Pre-AI": "#B2BABB", "Transition": "#D5D8DC", "AI-era": "#5DBBAA"}
INK = "#2C3E50"


def repos():
    cfg = yaml.safe_load(open(ROOT / "config" / "projects.yaml"))
    eras = {q: name for name, key in [("Pre-AI", "pre_ai"), ("Transition", "transition"), ("AI-era", "ai_era")]
            for q in cfg["eras"][key]["quarters"]}
    return {p["id"]: f"{p['owner']}/{p['repo']}" for p in cfg["projects"]}, eras


def milestones(ax, qs, labels=True):
    for q, name in MILESTONES:
        x = qs.index(q) - 0.5
        ax.axvline(x, color="#555555", lw=0.9, ls="--", zorder=1)
        if labels:
            ax.text(x + 0.12, 4, name, rotation=90, fontsize=6.5, color="#555555", va="bottom", ha="left")


def timeseries(d, name, path):
    qs = list(d.quarter)
    fig, ax = plt.subplots(figsize=(2637 / DPI, 1442 / DPI), dpi=DPI)
    ax.axhline(50, color="#BBBBBB", lw=0.7, ls=":", zorder=0)
    ax.plot(range(len(qs)), d.phs, color=INK, lw=1.3, zorder=2)
    for s in STATES:
        m = (d.linguistic_state == s).values
        if m.any():
            ax.scatter(np.arange(len(qs))[m], d.phs[m], s=42, color=COLOR[s], marker=MARKER[s],
                       edgecolor=INK, linewidth=0.6, zorder=3)
    hollow = (d.rules_fired == "NONE").values
    if hollow.any():
        ax.scatter(np.arange(len(qs))[hollow], d.phs[hollow], s=120, facecolor="none", edgecolor="#555555",
                   linewidth=0.8, zorder=4)
    milestones(ax, qs)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, len(qs) - 0.5)
    ax.set_xticks(range(0, len(qs), 2))
    ax.set_xticklabels(qs[::2], rotation=45, ha="right", fontsize=7.5)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_ylabel("PHS", fontsize=8.5)
    ax.set_title(f"PHS time series — {name}", fontsize=9.5)
    present = [s for s in STATES if (d.linguistic_state == s).any()]
    handles = [Line2D([], [], marker=MARKER[s], color="none", markerfacecolor=COLOR[s], markeredgecolor=INK,
                      markersize=6, label=s) for s in present]
    if hollow.any():
        handles.append(Line2D([], [], marker="o", color="none", markerfacecolor="none", markeredgecolor="#555555",
                              markersize=9, label="No rule fired (fallback)"))
    ax.legend(handles=handles, fontsize=6.8, loc="upper left", frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def timeline(d, name, path):
    qs = list(d.quarter)
    fig, ax = plt.subplots(figsize=(3300 / DPI, 630 / DPI), dpi=DPI)
    for i, s in enumerate(d.linguistic_state):
        ax.bar(i, 1, width=0.97, color=COLOR[s], align="center")
        ax.text(i, 0.5, LETTER[s], ha="center", va="center", fontsize=7.5,
                color="white" if s in ("Critical Risk", "Elite AI Maturity") else "#1B2631")
    milestones(ax, qs, labels=False)
    ax.set_xlim(-0.5, len(qs) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(range(0, len(qs), 2))
    ax.set_xticklabels(qs[::2], rotation=45, ha="right", fontsize=7)
    ax.set_title(f"Linguistic state over time — {name}", fontsize=9)
    present = [s for s in STATES if (d.linguistic_state == s).any()]
    ax.legend(handles=[Patch(color=COLOR[s], label=f"{LETTER[s]} = {s}") for s in present], fontsize=6.8,
              ncol=len(present), loc="upper center", bbox_to_anchor=(0.5, -0.62), frameon=False)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.42)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def boxplot(d, name, path):
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(1737 / DPI, 1439 / DPI), dpi=DPI)
    for i, e in enumerate(ERA_COLOR):
        v = d[d.era == e].phs.values
        ax.boxplot(v, positions=[i], widths=0.6, patch_artist=True, showfliers=False,
                   boxprops=dict(facecolor=ERA_COLOR[e], edgecolor="#333333", linewidth=0.9),
                   medianprops=dict(color="black", linewidth=1.1),
                   whiskerprops=dict(color="#333333", linewidth=0.9), capprops=dict(color="#333333", linewidth=0.9))
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=11, color="#566573", alpha=0.85, zorder=3, linewidth=0)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{e}\n(n = {(d.era == e).sum()})" for e in ERA_COLOR], fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_ylabel("PHS", fontsize=8.5)
    ax.set_title(f"PHS distribution by era — {name}", fontsize=8.8)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    names, eras = repos()
    t = pd.read_csv(ROOT / "results" / "phs_timeseries.csv")
    t["era"] = t.quarter.map(eras)
    for p, d in t.groupby("project_id"):
        d = d.sort_values("quarter").reset_index(drop=True)
        timeseries(d, names[p], OUT / f"phs_timeseries_{p}.png")
        timeline(d, names[p], OUT / f"state_timeline_{p}.png")
        boxplot(d, names[p], OUT / f"era_boxplot_{p}.png")
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()

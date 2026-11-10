"""Result plots (PNG) for the README and the PDF report.

Colours: validated categorical palette, fixed order; each SYSTEM always keeps its colour in every chart
(colour follows the entity, never its rank). Thin marks, 2px surface gaps, recessive grid, one axis per chart.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SYSTEM_COLORS = {            # slot order of the reference palette
    "dynamic": "#2a78d6",    # 1 blue   - ours
    "static": "#eb6834",     # 2 orange
    "vector_rerank": "#1baf7a",  # 3 aqua
    "vector": "#eda100",     # 4 yellow
    "none": "#e87ba4",       # 5 magenta
}
SYSTEM_LABELS = {"dynamic": "Dynamic graph (ours)", "static": "Static graph (rebuild)",
                 "vector_rerank": "Vector + reranker", "vector": "Vector RAG", "none": "No retrieval"}
TYPE_LABELS = {"inference_query": "Inference", "comparison_query": "Comparison", "temporal_query": "Temporal",
               "null_query": "Null (unanswerable)", "overall": "Overall"}


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=13, color=INK, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0)


def plot_stream(results_dir: Path, suffix: str = "", title: str | None = None) -> Path:
    data = json.loads((results_dir / f"stream_experiment{suffix}.json").read_text())
    b = data["batches"]
    x = [r["total_docs"] for r in b]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    for key, label in (("static", "Static: rebuild whole graph"), ("dynamic", "Dynamic: add new batch only")):
        y = [r[key]["seconds"] for r in b]
        ax.plot(x, y, color=SYSTEM_COLORS[key], linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.annotate(f"{y[-1]:.0f}s", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    _style(ax, title or "Time to update the graph after each new batch of articles", "seconds per update")
    ax.set_xlabel("articles in the graph after the update", color=INK2, fontsize=10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    out = results_dir / "plots" / f"stream_update_time{suffix}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def _grouped(ax, groups: list[str], systems: list[str], values: dict[str, list[float]], fmt: str = "{:.0%}",
             label_top: bool = True) -> None:
    n = len(systems)
    width = 0.8 / n
    for i, s in enumerate(systems):
        xs = [g + (i - (n - 1) / 2) * width for g in range(len(groups))]
        bars = ax.bar(xs, values[s], width=width, color=SYSTEM_COLORS[s], edgecolor=SURFACE, linewidth=2,
                      label=SYSTEM_LABELS[s])
        if label_top and s == "dynamic":  # selective direct labels: ours only
            for rect, v in zip(bars, values[s]):
                ax.annotate(fmt.format(v), (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                            xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups)


def plot_accuracy(results_dir: Path, systems: list[str]) -> Path:
    ev = results_dir / "eval"
    ms = {s: json.loads((ev / f"{s}_metrics.json").read_text()) for s in systems if (ev / f"{s}_metrics.json").exists()}
    systems = [s for s in systems if s in ms]
    types = ["inference_query", "comparison_query", "temporal_query", "null_query", "overall"]
    vals = {s: [ms[s]["accuracy"].get(t, 0.0) for t in types] for s in systems}
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=160)
    _grouped(ax, [TYPE_LABELS[t] for t in types], systems, vals)
    _style(ax, "Answer accuracy on MultiHop-RAG (2,556 questions)", "accuracy")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8.5, ncol=len(systems), loc="upper center", bbox_to_anchor=(0.5, -0.08),
              labelcolor=INK)
    fig.tight_layout()
    out = results_dir / "plots" / "accuracy_by_type.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_retrieval(results_dir: Path, systems: list[str]) -> Path:
    ev = results_dir / "eval"
    ms = {s: json.loads((ev / f"{s}_metrics.json").read_text()) for s in systems
          if (ev / f"{s}_metrics.json").exists() and s != "none"}
    systems = [s for s in systems if s in ms]
    metrics = ["hits@4", "hits@10", "recall@10", "mrr@10", "map@10"]
    vals = {s: [ms[s]["retrieval"].get(m, 0.0) for m in metrics] for s in systems}
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=160)
    _grouped(ax, ["Hits@4", "Hits@10", "Recall@10", "MRR@10", "MAP@10"], systems, vals, fmt="{:.2f}")
    _style(ax, "Did we retrieve the right articles? (answerable questions)", "score")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False, fontsize=8.5, ncol=len(systems), loc="upper center", bbox_to_anchor=(0.5, -0.08),
              labelcolor=INK)
    fig.tight_layout()
    out = results_dir / "plots" / "retrieval_metrics.png"
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out

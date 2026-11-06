"""plot_emphasis_results.py | Graphs of output/emphasis_test/results.json (from test_emphasis_llms.py).

Writes three PNGs next to the results:

    quality.png    focus accuracy, valid format, rank correlation and distance
                   from the Opus 5.5 benchmark, per model, with the expected value
    speed.png      load time, seconds per line and tokens per second, with the expected value
    per_line.png   every test line: the word each model stressed most, right or wrong

    python -m backend.emphasis.plot_emphasis_results

Needs matplotlib (in the dramatizer env; the llm env does not have it).
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_DIR / "output" / "emphasis_test"
TESTSET = PROJECT_DIR / "backend" / "data" / "emphasis_testset.json"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
MEASURED = "#2a78d6"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": TEXT, "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT,
    "axes.edgecolor": GRID, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
})


def bars(ax, names, values, expected, title, benchmark=None, limit=None, better="higher", fmt="{:.2f}"):
    """One horizontal bar per model, the expected value as a diamond, the benchmark as a dashed line."""
    y = np.arange(len(names))[::-1]
    ax.barh(y, values, height=0.55, color=MEASURED, zorder=2)
    low = limit[0] if limit else 0
    span = (limit[1] - low) if limit else max(max(values), 1e-9)
    for yi, v, e in zip(y, values, expected):
        near = e is not None and abs(e - v) < 0.12 * span          # the diamond sits on the bar end
        if v - low > 0.18 * span and v > 0:
            # Inside the bar: at its end, or at its start when the diamond covers the end
            x, ha = (0, "left") if near else (v, "right")
            ax.text(x, yi, ("  " + fmt.format(v)) if near else (fmt.format(v) + "  "), va="center", ha=ha,
                    color=SURFACE, fontsize=9, fontweight="bold", zorder=4)
        else:
            # Outside: past the bar and past the diamond
            ax.text(max(v, 0, e if near else v), yi, "   " + fmt.format(v), va="center", ha="left",
                    color=TEXT, fontsize=9, zorder=4)
    guesses = [(yi, e) for yi, e in zip(y, expected) if e is not None]
    if guesses:
        ax.scatter([e for _, e in guesses], [yi for yi, _ in guesses], marker="D", s=46,
                   facecolor=SURFACE, edgecolor=TEXT, linewidth=1.5, zorder=3, label="Expected (Opus 5.5's guess)")
    if benchmark is not None:
        ax.axvline(benchmark, color=TEXT_2, linestyle="--", linewidth=1.2, zorder=1, label="Opus 5.5 benchmark")
    ax.set_yticks(y, names)
    ax.set_title(f"{title}  ({better} is better)", loc="left")
    if limit:
        ax.set_xlim(*limit)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.tick_params(length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def legend(fig, axes):
    handles, labels = {}, {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=MEASURED)] + list(handles.values()),
               ["Measured"] + list(handles), loc="upper left", bbox_to_anchor=(0.005, 0.95), ncol=3, frameon=False)


def quality(results, expected, names):
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    get = lambda key: [results[n][key] for n in names]
    guess = lambda key: [expected.get(n, {}).get(key) for n in names]
    bars(axes[0, 0], names, get("focus_accuracy"), guess("focus_accuracy"),
         "Focus accuracy: right word stressed most", benchmark=1.0, limit=(0, 1.15))
    bars(axes[0, 1], names, get("valid_format"), guess("valid_format"),
         "Valid format: one value per word", benchmark=1.0, limit=(0, 1.15))
    bars(axes[1, 0], names, get("rank_corr"), [None] * len(names),
         "Rank correlation with the benchmark values", benchmark=1.0, limit=(-0.2, 1.15))
    bars(axes[1, 1], names, get("mae"), [None] * len(names),
         "Distance from benchmark, per word", benchmark=0.0, limit=(0, 0.45), better="lower")
    legend(fig, axes.flat)
    fig.suptitle("Emphasis LLM test: how close each model comes to Opus 5.5 (24 lines)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


def speed(results, expected, names):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    get = lambda key: [results[n][key] for n in names]
    guess = lambda key: [expected.get(n, {}).get(key) for n in names]
    load, line = get("load_s"), get("per_line_s")
    bars(axes[0], names, load, guess("load_s"), "Load time (s)", better="lower",
         limit=(0, max(load + [g or 0 for g in guess("load_s")]) * 1.3), fmt="{:.1f}")
    bars(axes[1], names, line, guess("per_line_s"), "Seconds per line", better="lower",
         limit=(0, max(line + [g or 0 for g in guess("per_line_s")]) * 1.3), fmt="{:.1f}")
    tps = get("tokens_per_s")
    bars(axes[2], names, tps, [None] * len(names), "Tokens per second", limit=(0, max(tps) * 1.3), fmt="{:.1f}")
    for ax in axes[1:]:
        ax.set_yticks(ax.get_yticks(), [""] * len(names))
    legend(fig, axes)
    fig.suptitle("Emphasis LLM test: speed (4-bit, RTX 3080 10 GB)", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.87), w_pad=2.5)
    return fig


def per_line(results, names):
    items = json.loads(TESTSET.read_text(encoding="utf-8"))["items"]
    columns = ["Opus 5.5 (benchmark)"] + names
    fig, ax = plt.subplots(figsize=(2.1 * len(columns) + 2.6, 0.36 * len(items) + 1.6))
    ax.set_xlim(0, len(columns))
    ax.set_ylim(len(items), 0)
    for r, item in enumerate(items):
        words = item["line"].split()
        cells = [("•", " / ".join(item["focus"]), None)]
        for n in names:
            row = results[n]["lines"][r]
            # Every word sharing the top value (a tie of two still counts)
            tied = [w.strip(".,!?") for w, v in zip(words, row["values"]) if v == max(row["values"])]
            top = " / ".join(tied) if len(tied) <= 2 else f"{len(tied)}-way tie"
            if not row["valid"]:
                # No clean answer (e.g. thinking ran out): the values were scraped from the text
                top += " *"
            cells.append(("✓" if row["correct"] else "✗", top, row["correct"]))
        for c, (mark, word, correct) in enumerate(cells):
            colour = SURFACE if correct is None else (GOOD if correct else CRITICAL)
            ax.add_patch(plt.Rectangle((c + 0.03, r + 0.06), 0.94, 0.88, color=colour,
                                       alpha=1 if correct is None else 0.18, linewidth=0))
            ax.text(c + 0.1, r + 0.5, f"{mark} {word.strip('.,!?')}", va="center", ha="left", fontsize=8.5,
                    color=TEXT if correct is not False else "#8a1f1f")
    ax.set_xticks(np.arange(len(columns)) + 0.5, columns, fontsize=9, fontweight="bold")
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(items)) + 0.5, [f"{i['id']}  “{i['line']}”" for i in items], fontsize=8.5)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    totals = "   ".join(f"{n}: {sum(r['correct'] for r in results[n]['lines'])}/{len(items)}" for n in names)
    fig.suptitle("Word stressed most on each line (✓ = one of the benchmark's focus words)", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.text(0.01, 0.005, "Right per model:   " + totals + "      * no clean answer: values read out of the text", fontsize=9, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return fig


def main():
    results = json.loads((OUT_DIR / "results.json").read_text(encoding="utf-8"))
    expected = json.loads(TESTSET.read_text(encoding="utf-8"))["expected"]
    names = list(results)
    for name, fig in [("quality", quality(results, expected, names)), ("speed", speed(results, expected, names)),
                      ("per_line", per_line(results, names))]:
        fig.savefig(OUT_DIR / f"{name}.png", dpi=130)
        print(OUT_DIR / f"{name}.png")


if __name__ == "__main__":
    main()

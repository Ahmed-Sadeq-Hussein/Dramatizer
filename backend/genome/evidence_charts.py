"""evidence_charts.py | Charts that answer the six genome questions, from the scores given between two dates.

    python main_backend.py evidence-charts                                    # every score so far
    python main_backend.py evidence-charts --from 2026-11-11 --to 2026-11-12
    python main_backend.py evidence-charts --from "2026-11-11 16:00" --to "2026-11-11 18:00"
    python main_backend.py evidence-charts --out "D:/somewhere"

The six questions, one chart each (question 6 has a chart and a table):

    1  Which genomes work best for each voice?
    2  Which genomes produce artifacts?
    3  Which genome configurations give bad MOS?
    4  Which genome configurations consistently give good genomes?
    5  How often does a genome beat the plain line of its voice, for each number of active features?
    6  Which genomes can we exclude?

The data is the evidence corpus (backend/rag/corpus.py), built fresh from its sources:
the listeners' scores (history/ratings/who_rated_what.csv) and the sanity checks. The sanity
check of every gene type on its own (sanity-all-genes) is used when it lies in the range, since
it pins a failure on one gene type; otherwise Milestone 2's random-v2 search, where each genome
had three genes. A score is used when the time it was given (rated_on) lies in the date range;
a sanity check when the time its search was made lies in it. --from and --to
take a date (YYYY-MM-DD) or a date and time (YYYY-MM-DD HH:MM); a --to date alone takes the
whole day. Each genome is compared with the same listener's score for the plain line of its
voice ("above plain"), as stored with the score.

The charts are JPEG files in OUTPUT_DIR/evidence charts/<the range>/, with comments.txt (each
chart's question, what it shows and what to keep in mind) and summary.json (the same, as data).
A chart without data in the range is skipped, and the reason is written down instead.
"""
import argparse
import json
import statistics as st
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from backend import model, safe_files

CHARTS_DIR = model.OUTPUT_DIR / "evidence charts"
FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d")
FEW = 20                   # fewer genomes than this in a bar: drawn grey, too few to trust
MIN_GENOMES = 10           # a gene type needs this many rated genomes to be called consistent (question 4)
LOWER_BELOW = 25           # question 6: better than plain in under this % (and below every listener's average)
KEEP_AT_LEAST = 35         # question 6: better than plain in at least this % (and below no listener's average)

INK, MUTED, GRID, REF, OTHER = "#1d1f22", "#6b7078", "#e6e7ea", "#8a8f98", "#c3c7cd"
BLUE, GOOD, BAD = "#2f6fdb", "#2e8b57", "#c0392b"
LISTENER_COLORS = ["#2f6fdb", "#d9822b", "#7a4fd0", "#1a9e8f"]      # by listener name, in fixed order


# ---- the date range and the data -------------------------------------------------------------

def parse_when(text, end=False):
    """'2026-11-11' or '2026-11-11 16:00' -> '2026-11-11 16:00:00'; a date alone as --to is the end of that day."""
    if text in (None, ""):
        return None
    text = text.strip()
    for fmt in FORMATS:
        try:
            when = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if end and fmt == "%Y-%m-%d":
            when += timedelta(days=1, seconds=-1)
        return when.strftime(FORMATS[0])
    raise ValueError(f"Not a date: {text!r}. Use YYYY-MM-DD or YYYY-MM-DD HH:MM.")


def in_range(when, start, end):
    return bool(when) and (start is None or when >= start) and (end is None or when <= end)


def select(records, start=None, end=None):
    """The genome scores and sanity checks whose time lies in [start, end], and every listener (for fixed colours)."""
    clips = [r for r in records if r["kind"] == "clip"]
    listeners = sorted({c["fields"]["listener"] for c in clips})
    genomes = [r["fields"] | {"id": r["id"], "when": r["revision"]} for r in clips
               if r["fields"]["genome"] is not None and r["fields"]["above_plain"] is not None
               and in_range(r["revision"], start, end)]
    sanity = [r["fields"] for r in records if r["kind"] == "sanity" and in_range(r["revision"], start, end)]
    alone = [s for s in sanity if s.get("genes") == 1]
    sanity, label = (alone, "each gene type alone") if alone else (sanity, "random-v2, 3 genes per genome")
    return {"genomes": genomes, "sanity": sanity, "sanity_label": label, "listeners": listeners,
            "start": start, "end": end}


def rnd(x):
    """Round half up (round() would give 22 for 22.5)."""
    return int(x + 0.5)


def pct(part, whole):
    return rnd(100 * part / whole) if whole else 0


def per_gene_type(genomes):
    out = defaultdict(list)
    for g in genomes:
        for t in g["gene_types"]:
            out[t].append(g)
    return out


def gaps(genomes):
    """{(gene type, listener): mean score of that listener's genomes with the type minus their own average}."""
    by_listener = defaultdict(list)
    for g in genomes:
        by_listener[g["listener"]].append(g["score"])
    average = {name: st.mean(s) for name, s in by_listener.items()}
    out = {}
    for t, items in per_gene_type(genomes).items():
        for name in average:
            s = [g["score"] for g in items if g["listener"] == name]
            if s:
                out[(t, name)] = st.mean(s) - average[name]
    return out, average


def gene_type_table(data):
    """One row per gene type: listeners who heard it, how many were below their average, better than plain, sanity."""
    genomes, sanity = data["genomes"], data["sanity"]
    gap, _ = gaps(genomes)
    overall = 100 * sum(g["above_plain"] > 0 for g in genomes) / len(genomes) if genomes else 0
    checked = per_gene_type(sanity)
    rows = []
    for t, items in sorted(per_gene_type(genomes).items()):
        heard = [name for name in data["listeners"] if (t, name) in gap]
        below = sum(gap[(t, name)] < 0 for name in heard)
        better = sum(g["above_plain"] > 0 for g in items)
        s = checked.get(t, [])
        rows.append({"gene_type": t, "genomes": len(items), "better": better, "better_pct": pct(better, len(items)),
                     "listeners": len(heard), "below_average": below,
                     "consistent": (len(items) >= MIN_GENOMES and below == 0 and len(heard) > 0
                                        and 100 * better / len(items) > overall),
                     "sanity_checked": len(s), "sanity_failed_pct": pct(sum(x["verdict"] == "failed" for x in s), len(s))
                     if s else None})
    for r in rows:
        r["suggestion"] = ("Lower chance" if r["listeners"] and r["below_average"] == r["listeners"] and r["better_pct"] < LOWER_BELOW
                           else "Keep" if r["below_average"] == 0 and r["better_pct"] >= KEEP_AT_LEAST else "No clear signal")
    return rows


# ---- drawing ---------------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.spines.left": False, "savefig.facecolor": "white"})
    return plt


def _grid(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y" if axis == "x" else "x", length=0)


def _legend(ax, handles, **kw):
    ax.legend(handles=handles, loc="lower left", frameon=False, ncol=len(handles), bbox_to_anchor=(0, 1.0), **kw)


def _dot(plt, color, name):
    return plt.Line2D([], [], marker="o", linestyle="", markersize=7, markerfacecolor=color, markeredgecolor="white", label=name)


def _span(data):
    first = min((g["when"] for g in data["genomes"]), default=None)
    last = max((g["when"] for g in data["genomes"]), default=None)
    return f"{len(data['genomes'])} genome scores given {first} to {last}" if first else "no genome scores"


def chart_best_per_voice(plt, data):
    genomes = data["genomes"]
    if not genomes:
        return None, "no genome scores in this range"
    above = defaultdict(list)
    for g in genomes:
        above[(g["voice"], g["genome"])].append(g["above_plain"])
    mean = {k: st.mean(v) for k, v in above.items()}
    voices = sorted({v for v, _ in mean})
    best = {v: max(x for (w, _), x in mean.items() if w == v) for v in voices}
    tied = sum(1 for v in voices if sum(x == best[v] for (w, _), x in mean.items() if w == v) > 1)
    better = sum(best[v] > 0 for v in voices)
    fig, ax = plt.subplots(figsize=(10, max(3.2, 0.33 * len(voices) + 1.6)))
    for i, v in enumerate(voices):
        for (w, number), x in mean.items():
            if w == v:
                top = x == best[v]
                ax.scatter(x, i + (number % 5 - 2) * 0.07, s=42 if top else 30, color=BLUE if top else OTHER,
                           edgecolor="white", linewidth=0.8, zorder=3 if top else 2)
    ax.axvline(0, color=REF, linewidth=1.4, zorder=1)
    ax.set_yticks(range(len(voices)), voices)
    ax.set_ylim(len(voices) - 0.5, -0.5)
    ax.set_xlim(-5.2, 5.2)
    ax.set_xticks(range(-5, 6), [f"+{x}" if x > 0 else str(x) for x in range(-5, 6)])
    ax.set_xlabel("Genome score minus the plain line of its voice (same listener; mean if several listened)")
    _grid(ax)
    _legend(ax, [_dot(plt, BLUE, "Best for its voice"), _dot(plt, OTHER, "Other genomes")])
    claim = (f"Every voice has a best genome, but {tied} of {len(voices)} voices share first place between two or more "
             f"genomes, and {better} of {len(voices)} voices have at least one genome rated above their plain line.")
    note = ("Right of the line: better than the plain line. Where one listener scored a voice, \"best\" is that "
            f"listener's choice. Data: {_span(data)}.")
    return (fig, claim, note), None


def _bars(ax, rows, xlabel, overall):
    """Horizontal % bars, one per (name, part, whole), with the overall % as a dashed line."""
    rows = sorted(rows, key=lambda r: r[1] / r[2])
    ax.barh([r[0] for r in rows], [100 * r[1] / r[2] for r in rows], height=0.65, zorder=2,
            color=[OTHER if r[2] < FEW else BLUE for r in rows])
    for i, (_, part, whole) in enumerate(rows):
        ax.text(100 * part / whole + 0.8, i, f"{pct(part, whole)}%  ({part} of {whole})", va="center", fontsize=8.5, color=INK)
    ax.axvline(overall, color=REF, linestyle="--", linewidth=1.3, zorder=3)
    top = max([100 * r[1] / r[2] for r in rows] + [overall])
    ax.set_xlim(0, min(100, top * 1.35 + 5))
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    ax.set_xlabel(xlabel)
    _grid(ax)


def chart_artifacts(plt, data):
    genomes, sanity = data["genomes"], data["sanity"]
    answered = [g for g in genomes if g["artifact"] is not None]
    if not answered and not sanity:
        return None, "no artifact answers and no sanity checks in this range"
    panels = (["listeners"] if answered else []) + (["sanity"] if sanity else [])     # only the panels with data
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels) + 1, 6.4), squeeze=False)
    fig.subplots_adjust(wspace=0.75)
    ax = dict(zip(panels, axes[0]))
    heard = sum(g["artifact"] for g in answered)
    if answered:
        ax["listeners"].set_title("Listeners: artifact heard", loc="left", color=INK, fontsize=10.5)
        rows = [(t, sum(g["artifact"] for g in items), len(items)) for t, items in per_gene_type(answered).items()]
        _bars(ax["listeners"], rows, "Share of answered genomes with this gene type", pct(heard, len(answered)))
    failed = sum(x["verdict"] == "failed" for x in sanity)
    if sanity:
        ax["sanity"].set_title(f"Sanity check: failed ({data['sanity_label']})", loc="left", color=INK, fontsize=10.5)
        rows = [(t, sum(x["verdict"] == "failed" for x in items), len(items)) for t, items in per_gene_type(sanity).items()]
        _bars(ax["sanity"], rows, "Share of checked genomes with this gene type", pct(failed, len(sanity)))
        worst = max(rows, key=lambda r: r[1] / r[2])
    parts = []
    if answered:
        parts.append(f"Listeners heard an artifact in {heard} of {len(answered)} genomes they answered for ({pct(heard, len(answered))}%).")
    else:
        parts.append(f"No listener answered the artifact question in this range ({len(genomes)} genome scores), so the "
                     "chart shows only the sanity check.")
    if not sanity:
        parts.append("No sanity check was made in this range, so the chart shows only what listeners heard.")
    else:
        parts.append(f"The sanity check failed {failed} of {len(sanity)} genomes ({pct(failed, len(sanity))}%: words dropped or "
                     f"changed, voice drift or broken pace); most often with {worst[0]} ({pct(worst[1], worst[2])}%).")
    note = ("Dashed lines: all genomes. A genome counts once for each of its gene types. Grey bars: fewer than "
            f"{FEW} genomes. A failed sanity check is a measured fault, not what a listener heard. "
            f"Data: {_span(data)}; {len(sanity)} sanity checks.")
    return (fig, " ".join(parts), note), None


def chart_bad_mos(plt, data):
    genomes = data["genomes"]
    if not genomes:
        return None, "no genome scores in this range"
    gap, average = gaps(genomes)
    names = [n for n in data["listeners"] if n in average]
    color = {n: LISTENER_COLORS[i] if i < len(LISTENER_COLORS) else OTHER for i, n in enumerate(data["listeners"])}
    types = sorted({t for t, _ in gap}, key=lambda t: st.mean(gap[(t, n)] for n in names if (t, n) in gap), reverse=True)
    all_below = [t for t in reversed(types) if all(gap[(t, n)] < 0 for n in names if (t, n) in gap)]
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.36 * len(types) + 1.8)))
    for i, t in enumerate(types):
        ax.axhline(i, color=GRID, linewidth=0.8, zorder=1)
        for n in names:
            if (t, n) in gap:
                ax.scatter(gap[(t, n)], i, color=color[n], s=46, edgecolor="white", linewidth=0.8, zorder=3)
    ax.axvline(0, color=REF, linewidth=1.4, zorder=2)
    ax.set_yticks(range(len(types)), [f"{t}  ✕" if t in all_below else t for t in types])
    ax.set_ylim(len(types) - 0.4, -0.6)
    lim = max(2, max(abs(x) for x in gap.values()) + 0.2)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Mean score of genomes with this gene type, minus the listener's own average")
    ax.tick_params(axis="y", length=0)
    _legend(ax, [_dot(plt, color[n], n) for n in names])
    claim = (f"Below the average of every listener who scored them: {', '.join(all_below)}." if all_below else
             "No gene type scored below the average of every listener who scored it.")
    scales = ", ".join(f"{n} {average[n]:.1f}" for n in names)
    note = ("Marked ✕: below average for every listener who heard it. Left of the line: below that listener's average. "
            f"Comparing each listener with their own average removes their scale (average genome score: {scales}). "
            f"Whole configurations can't be judged: each was scored once. Data: {_span(data)}.")
    return (fig, claim, note), None


def chart_consistently_good(plt, data):
    genomes = data["genomes"]
    if not genomes:
        return None, "no genome scores in this range"
    rows = sorted(gene_type_table(data), key=lambda r: r["better"] / r["genomes"])
    overall = pct(sum(g["above_plain"] > 0 for g in genomes), len(genomes))
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.36 * len(rows) + 1.8)))
    ax.barh([r["gene_type"] for r in rows], [100 * r["better"] / r["genomes"] for r in rows], height=0.65, zorder=2,
            color=[GOOD if r["consistent"] else OTHER for r in rows])
    for i, r in enumerate(rows):
        ax.text(100 * r["better"] / r["genomes"] + 0.8, i, f"{r['better_pct']}%  ({r['better']} of {r['genomes']})  "
                f"{r['listeners'] - r['below_average']} of {r['listeners']} listeners above avg.",
                va="center", fontsize=8.5, color=INK)
    ax.axvline(overall, color=REF, linestyle="--", linewidth=1.3, zorder=3)
    ax.set_xlim(0, min(100, max(100 * r["better"] / r["genomes"] for r in rows) * 1.8 + 5))
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    ax.set_xlabel("Genomes with this gene type rated better than their plain line")
    _grid(ax)
    from matplotlib.patches import Patch
    _legend(ax, [Patch(color=GOOD, label="Consistent: above every listener's average and the dashed line"), Patch(color=OTHER, label="Not consistent")])
    good = [r["gene_type"] for r in reversed(rows) if r["consistent"]]
    claim = (f"Consistently good: {', '.join(good)} scored above the average of every listener who heard them." if good else
             "No gene type scored above the average of every listener who heard it.")
    note = (f"Consistent = above the average of every listener who scored it, better than its plain line more often than "
            f"all genomes together, and at least {MIN_GENOMES} genomes. "
            f"Dashed line: all genomes ({overall}% better than their plain line). Whole configurations can't be judged: "
            f"each was scored once, so this is by gene type. Data: {_span(data)}.")
    return (fig, claim, note), None


def chart_feature_count(plt, data):
    genomes = data["genomes"]
    if not genomes:
        return None, "no genome scores in this range"
    by_n = defaultdict(list)
    for g in genomes:
        by_n[g["features"]].append(g)
    ns = sorted(by_n)
    share = {k: (sum(g["above_plain"] > 0 for g in by_n[k]), len(by_n[k])) for k in ns}
    fig, ax = plt.subplots(figsize=(max(5, 1.6 * len(ns) + 2), 5))
    ax.bar([f"{k} feature{'s' if k != 1 else ''}" for k in ns], [100 * b / n for b, n in share.values()], width=0.55, zorder=2,
           color=[OTHER if share[k][1] < FEW else BLUE for k in ns])
    for i, k in enumerate(ns):
        b, n = share[k]
        ax.text(i, 100 * b / n + 1.2, f"{pct(b, n)}%\n{b} of {n}", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_ylim(0, min(100, max(100 * b / n for b, n in share.values()) + 18))
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
    _grid(ax, "y")
    ax.set_ylabel("Better than their plain line")
    trusted = [k for k in ns if share[k][1] >= FEW]
    claim = ("Share of genomes that beat their plain line: "
             + ", ".join(f"{k} feature{'s' if k != 1 else ''} {pct(*share[k])}%" for k in ns) + ".")
    note = (f"Grey bars: fewer than {FEW} genomes, too few to trust. "
            + (f"Bars with enough genomes: {', '.join(map(str, trusted))} features. " if trusted else "")
            + f"Nothing above {max(ns)} features was scored in this range. Data: {_span(data)}.")
    return (fig, claim, note), None


def chart_exclude(plt, data):
    genomes = data["genomes"]
    if not genomes:
        return None, "no genome scores in this range"
    rows = gene_type_table(data)
    tone = {"Lower chance": BAD, "Keep": GOOD, "No clear signal": OTHER}
    overall_better = pct(sum(g["above_plain"] > 0 for g in genomes), len(genomes))
    fig = plt.figure(figsize=(13.5, max(5.2, 0.3 * len(rows) + 1.6)))
    ax = fig.add_axes([0.06, 0.1, 0.38, 0.78])
    tab = fig.add_axes([0.5, 0.02, 0.49, 0.9])
    plotted = [r for r in rows if r["sanity_failed_pct"] is not None]
    if plotted:
        spots = []                     # [x, y, names]: labelled points close together share one label
        for r in plotted:
            grey = r["suggestion"] == "No clear signal"
            x, y = r["sanity_failed_pct"], r["better_pct"]
            ax.scatter(x, y, color=tone[r["suggestion"]], s=70, edgecolor="white", zorder=2 if grey else 4)
            if not grey:
                near = next((p for p in spots if abs(p[0] - x) <= 2 and abs(p[1] - y) <= 1.5), None)
                if near:
                    near[2].append(r["gene_type"])
                    near[0] = max(near[0], x)
                else:
                    spots.append([x, y, [r["gene_type"]]])
        for x, y, names in spots:
            ax.annotate(", ".join(names), (x, y), xytext=(9, 0), textcoords="offset points", va="center", fontsize=8.5,
                        color=INK, zorder=5, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))
        failed_all = pct(sum(x["verdict"] == "failed" for x in data["sanity"]), len(data["sanity"]))
        ax.axvline(failed_all, color=REF, linestyle="--", linewidth=1.2)
        xs = [r["sanity_failed_pct"] for r in plotted]
        ax.set_xlim(max(-3, min(xs) - 8), min(100, max(xs) + 12))           # room for whole dots at 0%
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
        ax.set_xlabel(f"Failed the sanity check ({data['sanity_label']})")
    else:
        failed_all = None
        ax.set_xticks([])
        ax.set_xlabel("No sanity check in this range")
    ax.axhline(overall_better, color=REF, linestyle="--", linewidth=1.2)
    ys = [r["better_pct"] for r in rows]
    ax.set_ylim(max(0, min(ys) - 8), min(100, max(ys) + 8))
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0f}%")
    ax.set_ylabel("Better than the plain line (listening)")
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    _legend(ax, [_dot(plt, c, s) for s, c in tone.items()], fontsize=9)
    tab.axis("off")
    order = sorted(rows, key=lambda r: ({"Lower chance": 0, "No clear signal": 1, "Keep": 2}[r["suggestion"]], r["better_pct"]))
    cells = [[r["gene_type"], f"{r['below_average']} of {r['listeners']}", f"{r['better_pct']}% of {r['genomes']}",
              "not checked" if r["sanity_failed_pct"] is None else f"{r['sanity_failed_pct']}% of {r['sanity_checked']}",
              r["suggestion"]] for r in order]
    t = tab.table(cellText=cells, colLabels=["Gene type", "Listeners\nbelow avg.", "Better than\nplain", "Sanity\nfailed",
                                             "Suggestion"], loc="upper left", cellLoc="left", colLoc="left",
                  colWidths=[0.29, 0.16, 0.19, 0.15, 0.21])
    t.auto_set_font_size(False)
    t.set_fontsize(8.5)
    t.scale(1, 1.5)
    for (row, col), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.visible_edges = "B"
        if row == 0:
            cell.set_text_props(color=MUTED, fontweight="bold")
        elif col == 4:
            s = cell.get_text().get_text()
            cell.set_text_props(color=MUTED if s == "No clear signal" else tone[s], fontweight="bold")
    down = [r["gene_type"] for r in order if r["suggestion"] == "Lower chance"]
    keep = [r["gene_type"] for r in order if r["suggestion"] == "Keep"]
    claim = (f"Give a lower chance in the next search: {', '.join(down) or 'none'}. Keep: {', '.join(keep) or 'none'}. "
             f"Rule: lower chance = below the average of every listener who heard it and better than the plain line in "
             f"under {LOWER_BELOW}% of its genomes; keep = below no listener's average and better in at least {KEEP_AT_LEAST}%.")
    note = (f"Dashed lines: all genomes ({overall_better}% better than plain"
            + (f", {failed_all}% failed the sanity check" if failed_all is not None else "") + "). Sanity failure is shown, "
            "not used in the rule. A lower chance rather than removal, because each clip had one listener. "
            f"Data: {_span(data)}; {len(data['sanity'])} sanity checks.")
    return (fig, claim, note), None


QUESTIONS = [
    ("1", "Which genomes work best for each voice?", "q1_best_genome_per_voice.jpg", chart_best_per_voice),
    ("2", "Which genomes produce artifacts?", "q2_artifacts.jpg", chart_artifacts),
    ("3", "Which genome configurations give bad MOS?", "q3_bad_mos.jpg", chart_bad_mos),
    ("4", "Which genome configurations consistently give good genomes?", "q4_consistently_good.jpg", chart_consistently_good),
    ("5", "How often does a genome beat the plain line of its voice, for each number of active features?",
     "q5_active_features.jpg", chart_feature_count),
    ("6", "Which genomes can we exclude?", "q6_exclude.jpg", chart_exclude),
]


def folder_name(start, end):
    if not start and not end:
        return "all dates"
    short = lambda w: w[:16].replace(":", "-").replace(" ", "_")    # noqa: E731
    return f"{short(start) if start else 'start'} to {short(end) if end else 'now'}"


def make_charts(start=None, end=None, out_dir=None, records=None):
    """Draw the six charts for the scores given between start and end (text dates, either may be None).

    Returns {"range", "folder", "genome_scores", "sanity_checks", "charts": [{"number", "question", "file",
    "claim", "note"} or {"number", "question", "skipped"}]}. records: the corpus (default: built from its sources).
    """
    start, end = parse_when(start), parse_when(end, end=True)
    if start and end and start > end:
        raise ValueError(f"The range starts ({start}) after it ends ({end}).")
    if records is None:
        from backend.rag import corpus
        records = corpus.build()
    data = select(records, start, end)
    folder = Path(out_dir) if out_dir else CHARTS_DIR / folder_name(start, end)
    folder.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    charts = []
    for number, question, name, draw in QUESTIONS:
        made, why = draw(plt, data)
        if made is None:
            charts.append({"number": number, "question": question, "skipped": why})
            continue
        fig, claim, note = made
        path = folder / name
        fig.savefig(path, dpi=200, format="jpeg", pil_kwargs={"quality": 92}, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)
        charts.append({"number": number, "question": question, "file": str(path), "claim": claim, "note": note})
    summary = {"made": datetime.now().strftime(FORMATS[0]), "range": {"from": start, "to": end}, "folder": str(folder),
               "genome_scores": len(data["genomes"]), "sanity_checks": len(data["sanity"]),
               "sanity_source": data["sanity_label"], "charts": charts}
    with open(folder / "comments.txt", "w", encoding="utf-8") as f:
        f.write(f"Scores given from {start or 'the first score'} to {end or 'now'}: {len(data['genomes'])} genome scores, "
                f"{len(data['sanity'])} sanity checks.\n\n")
        for c in charts:
            f.write(f"{c['number']}. {c['question']}\n")
            f.write(f"Skipped: {c['skipped']}.\n\n" if "skipped" in c else f"{c['claim']}\n{c['note']}\n\n")
    safe_files.write_json(folder / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Charts for the six genome questions, from the scores given between two dates.")
    parser.add_argument("--from", dest="start", metavar="WHEN", help="first date (YYYY-MM-DD or 'YYYY-MM-DD HH:MM')")
    parser.add_argument("--to", dest="end", metavar="WHEN", help="last date; a date alone takes the whole day")
    parser.add_argument("--out", help=f"folder for the charts (default: {CHARTS_DIR}/<the range>)")
    args = parser.parse_args(argv)
    try:
        summary = make_charts(args.start, args.end, args.out)
    except ValueError as e:
        raise SystemExit(str(e))
    r = summary["range"]
    print(f"From {r['from'] or 'the first score'} to {r['to'] or 'now'}: {summary['genome_scores']} genome scores, "
          f"{summary['sanity_checks']} sanity checks")
    for c in summary["charts"]:
        print(f"  {c['number']}. {c['question']}\n     " + (f"skipped: {c['skipped']}" if "skipped" in c else c["file"]))
    print(f"Comments: {Path(summary['folder']) / 'comments.txt'}")


if __name__ == "__main__":
    main()

"""mos_study.py | The listening studies, counted only from the piles each listener was given to rate.

    python main_backend.py mos-study            # the summary, the charts and the CSV, in OUTPUT_DIR/mos_study
    python main_backend.py mos-study --print    # also print the summary

Only the scores of a pile ASSIGNED to the listener who gave them count here: a score of an account outside its
piles (a pile looked at for testing, a pile given later to someone else, a test account) is left out. Two
studies are summed up:

  Report evaluation  (report_eval.py) the paper's blind test on the 12 locked test voices: every item (voice x
                     sentence) rendered three ways, the CONDITIONS plain (no genome), random (a random genome, no
                     checks) and guarded (the Dramatizer's guarded draw), each item heard by two listeners. The
                     condition of each version is read from the key beside the plan, which is opened here only
                     because every assigned score is in (status() says so). Only totals leave this module: means,
                     intervals, paired differences and counts, never which version of which run was which.
  Pilot              mos-diverse-v1 (explore.py): far-apart genomes of the voice-map voices, one listener a pile,
                     each genome scored against the plain line of its voice.

Every interval is a 95 % bootstrap over items (the item, not the single score, is the unit: an item's scores
are not independent), seeded so it is the same on every run. Paired differences compare two versions of the
same item scored by the same listener, which removes each listener's own scale and each voice's own quality.
"""
import argparse
import csv
import json
import random
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from backend import model, safe_files

OUT_DIR = model.OUTPUT_DIR / "mos_study"
REPORT, PILOT = "report-evaluation", "mos-diverse-v1"
CONDITIONS = ("plain", "random", "guarded")
PAIRS = (("guarded", "plain"), ("random", "plain"), ("guarded", "random"))
BOOT, SEED = 4000, 20261123
STUDIES = {"report-evaluation": "Report evaluation", "mos-diverse-v1": "Pilot (mos-diverse-v1)"}
LABELS = {"plain": "Plain line (no genome)", "random": "Random genome (no checks)",
          "guarded": "Guarded genome (the Dramatizer)"}


# ---- the scores that count ---------------------------------------------------------------------------------

def assignments():
    """{username: {"batch|pile", ...}} of every account: the piles each was given to rate."""
    from backend.api import accounts
    return {a.username: set(a.assignments) for a in accounts.load().values()}


def assigned_scores(rows=None, given=None):
    """The who_rated_what.csv rows of an assigned pile, scored by an account it was assigned to. Returns
    (kept rows, {reason left out: count})."""
    from backend.genome import ratings

    if rows is None:
        with open(ratings.who_path(), encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    given = assignments() if given is None else given
    kept, left = [], Counter()
    for row in rows:
        pile = f"{row['batch']}|{row['pile']}" if row.get("batch") and row.get("pile") else None
        if pile is None:
            left["not in a pile"] += 1
        elif pile not in given.get(row.get("account") or "", set()):
            left[f"{row['batch']}: not assigned to {row.get('rater') or row.get('account')}"] += 1
        else:
            kept.append(row)
    return kept, dict(left)


def coverage(kept, given=None):
    """Per listener and assigned pile: clips scored of clips in it (from the scores and the runs)."""
    given = assignments() if given is None else given
    scored = Counter((r["account"], f"{r['batch']}|{r['pile']}") for r in kept)
    out = []
    for user, piles in sorted(given.items()):
        for pile in sorted(piles):
            batch = pile.split("|")[0]
            if batch in (REPORT, PILOT):
                out.append({"account": user, "batch": batch, "study": STUDIES[batch], "pile": int(pile.split("|")[1]),
                            "scored": scored.get((user, pile), 0)})
    return out


# ---- small statistics ----------------------------------------------------------------------------------------

def _mean(xs):
    return round(st.fmean(xs), 3) if xs else None


def _boot(groups, stat, rng):
    """A 95 % interval of stat(list of groups) by resampling the groups (items) with replacement."""
    if len(groups) < 2:
        return None
    values = sorted(stat([groups[rng.randrange(len(groups))] for _ in groups]) for _ in range(BOOT))
    return [round(values[int(0.025 * BOOT)], 3), round(values[int(0.975 * BOOT) - 1], 3)]


def _wilcoxon(diffs):
    from scipy.stats import wilcoxon
    nonzero = [d for d in diffs if d]
    if len(nonzero) < 6:
        return None
    return float(wilcoxon(nonzero).pvalue)


def _holm(ps):
    """Holm-adjusted p values, in the same order."""
    order = sorted(range(len(ps)), key=lambda i: (ps[i] is None, ps[i] or 0))
    out, worst = [None] * len(ps), 0.0
    m = sum(p is not None for p in ps)
    for rank, i in enumerate(order):
        if ps[i] is None:
            continue
        worst = max(worst, min(1.0, ps[i] * (m - rank)))
        out[i] = worst
    return out


def _spearman(a, b):
    from scipy.stats import spearmanr
    return round(float(spearmanr(a, b).statistic), 3) if len(a) > 2 else None


# ---- the Report evaluation -----------------------------------------------------------------------------------

def report_scores(kept, key):
    """One record per assigned score of the Report evaluation, with its condition and what the item was."""
    from backend.genome import report_eval

    plan = report_eval.prepare()
    sentences = {s["id"]: s for s in plan["sentences"]}
    out = []
    for row in kept:
        if row["batch"] != REPORT:
            continue
        k = key["runs"][row["run"]]
        condition = k["versions"][row["clip"]]
        render = k["renders"][condition]
        types = sorted({report_eval.gene_type(g) for g in render.get("operations") or []})
        out.append({"item": k["item"], "voice": k["voice"], "dataset": k["voice"].split("/")[0],
                    "sentence": k["sentence"], "length": sentences[k["sentence"]]["length"],
                    "situation": sentences[k["sentence"]]["situation"], "genes": k["genes"], "pile": k["pile"],
                    "listener": row["rater"], "condition": condition, "score": float(row["score"]),
                    "artifact": row["artifact"] == "yes", "different": row["different_person"] == "yes",
                    "passed_checks": render.get("passed"), "guard_failed": render.get("guard_failed"),
                    "guard_tries": k["guarded_tries"], "gene_types": types, "markup": render.get("markup") or "",
                    "measures": {m: (render.get("sanity") or {}).get(m) for m in ("wer", "similarity", "pace")}})
    return out


def _by_item(records):
    items = defaultdict(list)
    for r in records:
        items[r["item"]].append(r)
    return list(items.values())


def _cond_mean(groups, condition):
    xs = [r["score"] for g in groups for r in g if r["condition"] == condition]
    return st.fmean(xs) if xs else 0.0


def condition_table(records, rng, z=None):
    """Per condition: scores, mean, sd, 95 % interval over items; flags; and the listener-normalised mean."""
    groups = _by_item(records)
    out = {}
    for c in CONDITIONS:
        mine = [r for r in records if r["condition"] == c]
        scores = [r["score"] for r in mine]
        out[c] = {"label": LABELS[c], "scores": len(scores), "mean": _mean(scores),
                  "sd": round(st.stdev(scores), 3) if len(scores) > 1 else None,
                  "ci95": _boot(groups, lambda gs, c=c: _cond_mean(gs, c), rng),
                  "median": st.median(scores) if scores else None,
                  "score_counts": {str(k): v for k, v in sorted(Counter(scores).items())},
                  "share_4_or_more": round(sum(s >= 4 for s in scores) / len(scores), 3) if scores else None,
                  "share_2_or_less": round(sum(s <= 2 for s in scores) / len(scores), 3) if scores else None,
                  "artifact_flags": sum(r["artifact"] for r in mine),
                  "different_person_flags": sum(r["different"] for r in mine)}
        if z is not None:
            out[c]["mean_z"] = _mean([z[id(r)] for r in mine])
    return out


def listener_z(records):
    """{id(record): the score as a z-score of its listener's own scores in this study}."""
    by = defaultdict(list)
    for r in records:
        by[r["listener"]].append(r["score"])
    norm = {k: (st.fmean(v), st.stdev(v) or 1.0) for k, v in by.items()}
    return {id(r): (r["score"] - norm[r["listener"]][0]) / norm[r["listener"]][1] for r in records}


def paired(records, rng):
    """For each pair of conditions: the differences of the same item scored by the same listener."""
    cell = {(r["item"], r["listener"], r["condition"]): r["score"] for r in records}
    pairs = sorted({(r["item"], r["listener"]) for r in records})
    out = []
    for a, b in PAIRS:
        rows = [(item, cell[(item, who, a)] - cell[(item, who, b)]) for item, who in pairs
                if (item, who, a) in cell and (item, who, b) in cell]
        diffs = [d for _, d in rows]
        per_item = defaultdict(list)
        for item, d in rows:
            per_item[item].append(d)
        groups = list(per_item.values())
        out.append({"a": a, "b": b, "pairs": len(diffs), "mean_difference": _mean(diffs),
                    "ci95": _boot(groups, lambda gs: st.fmean(d for g in gs for d in g), rng),
                    "a_higher": sum(d > 0 for d in diffs), "tie": sum(d == 0 for d in diffs),
                    "b_higher": sum(d < 0 for d in diffs), "wilcoxon_p": _wilcoxon(diffs)})
    for row, p in zip(out, _holm([r["wilcoxon_p"] for r in out])):
        row["wilcoxon_p"] = None if row["wilcoxon_p"] is None else round(row["wilcoxon_p"], 4)
        row["holm_p"] = None if p is None else round(p, 4)
    return out


def split_by(records, field):
    """{value: {condition: mean, ... , "scores": n}} for one field (listener, dataset, sentence, gene count)."""
    out = {}
    for value in sorted({r[field] for r in records}, key=str):
        mine = [r for r in records if r[field] == value]
        out[str(value)] = {c: _mean([r["score"] for r in mine if r["condition"] == c]) for c in CONDITIONS}
        out[str(value)]["scores"] = len(mine)
    return out


def winners(records):
    """Per item, the condition with the highest mean of its two listeners (a tie shares the point)."""
    points = Counter()
    for group in _by_item(records):
        means = {c: st.fmean(r["score"] for r in group if r["condition"] == c) for c in CONDITIONS
                 if any(r["condition"] == c for r in group)}
        top = max(means.values())
        best = [c for c, m in means.items() if m == top]
        for c in best:
            points[c] += 1 / len(best)
    return {c: round(points[c], 2) for c in CONDITIONS}


def agreement(records):
    """How much the two listeners of an item agree: on the scores, and on which version is best."""
    cell = defaultdict(dict)
    for r in records:
        cell[(r["item"], r["condition"])][r["listener"]] = r["score"]
    a, b = [], []
    for scores in cell.values():
        if len(scores) == 2:
            x, y = (scores[k] for k in sorted(scores))
            a.append(x)
            b.append(y)
    same_best = total = 0
    by_item = defaultdict(lambda: defaultdict(dict))
    for r in records:
        by_item[r["item"]][r["listener"]][r["condition"]] = r["score"]
    for listeners in by_item.values():
        if len(listeners) != 2:
            continue
        bests = [{c for c, s in v.items() if s == max(v.values())} for v in listeners.values()]
        total += 1
        same_best += bool(bests[0] & bests[1])
    return {"clips_with_two_scores": len(a), "spearman": _spearman(a, b),
            "mean_absolute_difference": _mean([abs(x - y) for x, y in zip(a, b)]),
            "items_same_best_version": same_best, "items": total}


def checks_against_listeners(records):
    """Did the automatic checks (sanity + audio) agree with the listeners? Random genomes were not filtered,
    so their pass/fail is a fair test; guarded ones that never passed (guard_failed) are shown apart."""
    out = {}
    rnd = [r for r in records if r["condition"] == "random"]
    for passed in (True, False):
        mine = [r["score"] for r in rnd if r["passed_checks"] is passed]
        out[f"random_{'passed' if passed else 'failed'}_checks"] = {"scores": len(mine), "mean": _mean(mine),
                                                                     "items": len({r["item"] for r in rnd if r["passed_checks"] is passed}) }
    g = [r for r in records if r["condition"] == "guarded"]
    for failed in (False, True):
        mine = [r["score"] for r in g if bool(r["guard_failed"]) is failed]
        out[f"guarded_{'guard_failed' if failed else 'passed'}"] = {"scores": len(mine), "mean": _mean(mine),
                                                                   "items": len({r["item"] for r in g if bool(r["guard_failed"]) is failed})}
    tries = Counter()
    for item in {r["item"]: r for r in g}.values():
        tries[str(item["guard_tries"])] += 1
    out["guarded_tries_per_item"] = dict(sorted(tries.items()))
    return out


def genome_diffs(records):
    """One row per genome version (random or guarded of an item): its two listeners' scores minus their own
    score of the plain line of the same item, averaged. Internal: never written out row by row."""
    plain = {(r["item"], r["listener"]): r["score"] for r in records if r["condition"] == "plain"}
    by = defaultdict(list)
    for r in records:
        if r["condition"] != "plain" and (r["item"], r["listener"]) in plain:
            by[(r["item"], r["condition"])].append(r)
    out = []
    for (item, condition), rs in by.items():
        out.append({"item": item, "condition": condition, "genes": rs[0]["genes"], "gene_types": rs[0]["gene_types"],
                    "markup": rs[0]["markup"], "measures": rs[0]["measures"], "flagged": any(x["artifact"] for x in rs),
                    "diff": st.fmean(x["score"] - plain[(item, x["listener"])] for x in rs)})
    return out


def _harm_row(diffs):
    return {"genomes": len(diffs), "minus_plain": _mean(diffs), "worse": sum(d < 0 for d in diffs),
            "same": sum(d == 0 for d in diffs), "better": sum(d > 0 for d in diffs)}


def gene_types(genomes, least=4):
    """Exploratory, per genome: genomes that carry each gene type, against the plain line of the same item and
    listener (only types in at least `least` genomes), worst first."""
    by = defaultdict(list)
    for g in genomes:
        for t in g["gene_types"]:
            by[t].append(g)
    out = [{"gene_type": t, "in_random": sum(g["condition"] == "random" for g in gs),
            "in_guarded": sum(g["condition"] == "guarded" for g in gs), **_harm_row([g["diff"] for g in gs])}
           for t, gs in by.items() if len(gs) >= least]
    return sorted(out, key=lambda x: x["minus_plain"])


TAG = re.compile(r"\[[a-z_]+\]")
POSITIONS = ("at a pause", "inside a phrase", "before the last word")


def tag_position(markup):
    """Where a genome's inline events ([breath], [sigh] ...) sit: "at a pause" when every one is next to
    punctuation, the start or the end; "before the last word" when one sits just before the line's last word;
    else "inside a phrase". None when it has no inline event (emphasis, laughing speech, speed only)."""
    kinds = set()
    for tag in TAG.finditer(markup):
        before, after = markup[:tag.start()].rstrip(), markup[tag.end():].lstrip()
        if not before or before.endswith((",", ".", "!", "?", "\u2014", "...")) or after[:1] in ("", ",", ".", "!", "?"):
            kinds.add("at a pause")
        elif re.match(r"^[A-Za-z']+[.!?]$", after.split(" ")[0]):
            kinds.add("before the last word")
        else:
            kinds.add("inside a phrase")
    if not kinds:
        return None
    return "before the last word" if "before the last word" in kinds else (
        "inside a phrase" if "inside a phrase" in kinds else "at a pause")


def harm(genomes):
    """Which genome versions hurt: how the loss against the plain line is spread, where their events sit, how
    many genes they had, and whether the automatic measures or the listeners' flags saw it. Totals only."""
    from scipy.stats import spearmanr

    diffs = sorted(g["diff"] for g in genomes)
    total = sum(diffs)
    spread = {c: {"genomes": len(ds), "much_worse": sum(d <= -1.5 for d in ds), "worse": sum(-1.5 < d <= -0.5 for d in ds),
                  "about_the_same": sum(-0.5 < d < 0.5 for d in ds), "better": sum(d >= 0.5 for d in ds)}
              for c in ("random", "guarded") for ds in [[g["diff"] for g in genomes if g["condition"] == c]]}
    positions = defaultdict(list)
    for g in genomes:
        where = tag_position(g["markup"])
        if where:
            positions[where].append(g["diff"])
    measures = {}
    for m in ("wer", "similarity", "pace"):
        pairs = [(g["measures"][m], g["diff"]) for g in genomes if isinstance(g["measures"][m], (int, float))]
        measures[m] = round(float(spearmanr(*zip(*pairs)).statistic), 2) if len(pairs) > 5 else None
    return {"genomes": len(genomes), "net_loss": round(total, 2),
            "share_of_loss_in_worst_10": round(sum(diffs[:10]) / total, 2) if total else None,
            "rest_without_worst": {str(n): _mean(diffs[n:]) for n in (5, 10, 15)},
            "spread": spread,
            "positions": {p: _harm_row(positions[p]) for p in POSITIONS if positions[p]},
            "gene_count": {str(n): _harm_row([g["diff"] for g in genomes if g["genes"] == n]) for n in (1, 2, 3)},
            "measures_vs_listeners": measures,
            "flagged": _harm_row([g["diff"] for g in genomes if g["flagged"]]),
            "not_flagged": _harm_row([g["diff"] for g in genomes if not g["flagged"]]),
            "note": "Exploratory, decided after the scores were in: a hypothesis for the next listening round, not a "
                    "claim of this one. Each genome is the mean of its two listeners, minus their plain-line scores."}


def report_breakdown(kept, key):
    records = report_scores(kept, key)
    if not records:
        return None
    rng = random.Random(SEED)
    z = listener_z(records)
    items = {r["item"] for r in records}
    return {"label": "Report evaluation", "scores": len(records), "items": len(items),
            "listeners": sorted({r["listener"] for r in records}),
            "voices": len({r["voice"] for r in records}),
            "conditions": condition_table(records, rng, z), "paired": paired(records, rng),
            "best_version_per_item": winners(records),
            "by_listener": split_by(records, "listener"), "by_dataset": split_by(records, "dataset"),
            "by_sentence": {k: {**v, "about": next(f"{r['length']}, {r['situation']}" for r in records
                                                   if r["sentence"] == k)}
                            for k, v in split_by(records, "sentence").items()},
            "by_gene_count": split_by([r for r in records], "genes"),
            "agreement": agreement(records), "checks": checks_against_listeners(records),
            "gene_types": gene_types(genome_diffs(records)), "harm": harm(genome_diffs(records))}


# ---- the pilot -------------------------------------------------------------------------------------------------

def pilot_summary(kept):
    rows = [r for r in kept if r["batch"] == PILOT]
    if not rows:
        return None
    genomes = [r for r in rows if r["clip"].startswith("genome")]
    plain = [float(r["score"]) for r in rows if not r["clip"].startswith("genome")]
    diffs = [float(r["above_plain_line"]) for r in genomes if r["above_plain_line"] not in ("", None)]
    per = {}
    for who in sorted({r["rater"] for r in rows}):
        g = [float(r["score"]) for r in genomes if r["rater"] == who]
        p = [float(r["score"]) for r in rows if r["rater"] == who and not r["clip"].startswith("genome")]
        d = [float(r["above_plain_line"]) for r in genomes if r["rater"] == who and r["above_plain_line"] not in ("", None)]
        per[who] = {"genome_scores": len(g), "genome_mean": _mean(g), "plain_mean": _mean(p),
                    "above_plain": sum(x > 0 for x in d), "same_as_plain": sum(x == 0 for x in d),
                    "below_plain": sum(x < 0 for x in d)}
    return {"label": "Pilot (mos-diverse-v1)", "scores": len(rows), "genome_scores": len(genomes),
            "plain_scores": len(plain), "genome_mean": _mean([float(r["score"]) for r in genomes]),
            "plain_mean": _mean(plain), "mean_above_plain": _mean(diffs),
            "above_plain": sum(x > 0 for x in diffs), "same_as_plain": sum(x == 0 for x in diffs),
            "below_plain": sum(x < 0 for x in diffs),
            "artifact_flags": sum(r["artifact"] == "yes" for r in rows),
            "different_person_flags": sum(r["different_person"] == "yes" for r in rows), "by_listener": per}


# ---- the files -----------------------------------------------------------------------------------------------

def report_complete(cover):
    rows = [c for c in cover if c["batch"] == REPORT]
    return bool(rows) and all(c["scored"] >= 24 for c in rows)


def build(out_dir=OUT_DIR, charts=True):
    """summary.json, scores_assigned.csv (without login names; the Report evaluation's runs left out, since a
    run with its scores would say which version is which) and the charts. Returns the summary."""
    from backend.genome import report_eval

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    given = assignments()
    kept, left = assigned_scores(given=given)
    cover = coverage(kept, given)
    complete = report_complete(cover)
    key = None
    if complete:
        key = json.loads((report_eval.batch_dir() / "key.json").read_text(encoding="utf-8"))
    names = {}
    for r in kept:
        names.setdefault(r["account"], r["rater"])
    summary = {"made": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "about": "Only scores of a pile assigned to the listener who gave them. Intervals: 95 % bootstrap "
                        "over items. Paired differences: the same item scored by the same listener.",
               "scores_kept": len(kept), "scores_left_out": left,
               "coverage": [{**{k: v for k, v in c.items() if k not in ("account", "batch")}, "listener": names.get(c["account"], c["account"])}
                            for c in cover if not c["account"].startswith("test_user")],
               "report": report_breakdown(kept, key) if key else None,
               "report_note": None if key else "Not every assigned score of the Report evaluation is in yet: its "
                                               "conditions stay sealed.",
               "pilot": pilot_summary(kept)}
    safe_files.write_json(out_dir / "summary.json", summary)
    fields = ["rater", "batch", "pile", "voice", "run", "clip", "genome", "score", "artifact", "different_person",
              "above_plain_line", "rated_on", "verdict", "markup", "text"]
    with open(out_dir / "scores_assigned.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(r for r in kept if r["batch"] != REPORT)
    if charts:
        summary["charts"] = make_charts(summary, out_dir)
        safe_files.write_json(out_dir / "summary.json", summary)
    return summary


# ---- the charts ----------------------------------------------------------------------------------------------

COLORS = {"plain": "#8a8a85", "random": "#d2752f", "guarded": "#2f6fd2"}     # gray = no genome; two hues


def make_charts(summary, out_dir):
    """JPEGs of the Report evaluation: [{file, title, note}]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rep = summary.get("report")
    if not rep:
        return []
    out_dir = Path(out_dir)
    made = []

    def style(ax):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y" if ax.get_xlim() else "x", color="#e6e6e1", linewidth=0.8)
        ax.set_axisbelow(True)

    def save(fig, name, title, note):
        path = out_dir / name
        fig.savefig(path, dpi=130, bbox_inches="tight", pil_kwargs={"quality": 90})
        plt.close(fig)
        made.append({"file": name, "title": title, "note": note})

    # 1. MOS per condition with its interval
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    conds = rep["conditions"]
    for i, c in enumerate(CONDITIONS):
        d = conds[c]
        ax.bar(i, d["mean"], width=0.55, color=COLORS[c], zorder=2)
        lo, hi = d["ci95"]
        ax.errorbar(i, d["mean"], yerr=[[d["mean"] - lo], [hi - d["mean"]]], color="#1d1d1b", capsize=5, lw=1.5, zorder=3)
        ax.text(i, hi + 0.08, f"{d['mean']:.2f}", ha="center", fontsize=10)
    ax.set_xticks(range(3), [LABELS[c].replace(" (", "\n(") for c in CONDITIONS], fontsize=9)
    ax.set_ylim(0, 5)
    ax.set_ylabel("Mean score (0–5)")
    style(ax)
    ax.set_title("Mean score of each method", fontsize=11, loc="left")
    save(fig, "1_mos_by_method.jpg", "Mean score of each method",
         f"{rep['scores']} scores: {rep['items']} items, each heard by 2 of the 4 listeners. Bars: mean; lines: 95 % "
         "interval (bootstrap over items).")

    # 2. Paired comparisons: who won, same item and listener
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    labels = []
    for i, p in enumerate(rep["paired"]):
        n = p["pairs"]
        left = 0
        for part, color in ((p["a_higher"], COLORS[p["a"]]), (p["tie"], "#d9d9d4"), (p["b_higher"], COLORS[p["b"]])):
            ax.barh(i, part / n * 100, left=left, color=color, edgecolor="white", linewidth=2, height=0.6)
            if part / n > 0.07:
                ax.text(left + part / n * 50, i, str(part), ha="center", va="center", fontsize=9,
                        color="white" if color != "#d9d9d4" else "#1d1d1b")
            left += part / n * 100
        labels.append(f"{p['a']} vs {p['b']}\nΔ {p['mean_difference']:+.2f}")
    ax.set_yticks(range(len(labels)), labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of pairs (same item, same listener): first higher · tie · second higher")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("Head to head", fontsize=11, loc="left")
    save(fig, "2_head_to_head.jpg", "Head to head",
         "Each bar: the pairs where the first method scored higher (its color), the same (gray) or lower. Δ: mean "
         "difference in points.")

    # 3. Per listener (small multiples in one axis: dots per condition)
    def dots(table, name, title, note, label=lambda k: k):
        keys = list(table)
        fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(keys) + 1.2))
        for j, k in enumerate(keys):
            vals = [table[k][c] for c in CONDITIONS if table[k][c] is not None]
            ax.plot([min(vals), max(vals)], [j, j], color="#cfcfca", lw=2, zorder=1)
            for c in CONDITIONS:
                if table[k][c] is not None:
                    ax.scatter(table[k][c], j, s=70, color=COLORS[c], edgecolor="white", linewidth=1.5, zorder=2,
                               label=LABELS[c] if j == 0 else None)
        ax.set_yticks(range(len(keys)), [label(k) for k in keys], fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(1, 5)
        ax.set_xlabel("Mean score (the scale is 0–5)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", color="#e6e6e1", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28 if len(keys) < 4 else -0.18), ncol=3, frameon=False, fontsize=8)
        ax.set_title(title, fontsize=11, loc="left")
        save(fig, name, title, note)

    dots(rep["by_listener"], "3_by_listener.jpg", "Each listener",
         "Each listener's mean for each method (24 items each). Listeners use the scale differently; the order of the "
         "methods is what to compare.")
    dots(rep["by_dataset"], "4_by_dataset.jpg", "Each dataset",
         "Mean per method for the 4 test voices of each dataset (16 items each).")
    dots(rep["by_sentence"], "5_by_sentence.jpg", "Each sentence",
         "Mean per method for each of the four lines (12 items each).",
         label=lambda k: f"{k} ({rep['by_sentence'][k]['about']})")
    dots(rep["by_gene_count"], "6_by_gene_count.jpg", "Number of genes",
         "Mean per method by how many genes both genomes of the item had (16 items each). Plain has no genes; it is "
         "shown for the same items.", label=lambda k: f"{k} gene{'s' if k != '1' else ''}")

    def diverging(rows, name, title, note, label):
        fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(rows) + 1.0))
        for j, r in enumerate(rows):
            v = r["minus_plain"]
            ax.barh(j, v, color="#c0503a" if v < 0 else "#2f6fd2", height=0.62, zorder=2)
            ax.text(v + (0.04 if v >= 0 else -0.04), j, f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=8)
        ax.axvline(0, color="#8a8a85", lw=1)
        ax.set_yticks(range(len(rows)), [label(r) for r in rows], fontsize=9)
        ax.invert_yaxis()
        lo = min(r["minus_plain"] for r in rows)
        hi = max(r["minus_plain"] for r in rows)
        ax.set_xlim(min(lo, 0) - 0.45, max(hi, 0) + 0.45)
        ax.set_xlabel("Points against the plain line of the same item and listener")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="x", color="#e6e6e1", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=11, loc="left")
        save(fig, name, title, note)

    diverging(rep["gene_types"], "7_gene_types.jpg", "Gene types against the plain line",
              "Each bar: the genomes carrying that gene type (random and guarded), their score minus the plain line's. "
              "Exploratory: a genome usually carries more than one type.",
              lambda r: f"{r['gene_type']} ({r['genomes']}{', guarded' if r['in_guarded'] else ''})")
    h = rep["harm"]
    diverging([{"label": k, **v} for k, v in h["positions"].items()] +
              [{"label": f"{k} gene{'s' if k != '1' else ''}", **v} for k, v in h["gene_count"].items()],
              "8_where_and_how_many.jpg", "Where the events sit, and how many genes",
              "Top: genomes by where their inline events ([breath], [sigh] ...) sit in the line. Bottom: by gene count. "
              "Exploratory, decided after the scores were in.",
              lambda r: f"{r['label']} ({r['genomes']})")

    return made


def main(argv=None):
    parser = argparse.ArgumentParser(description="The listening studies, from assigned piles only.")
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--print", action="store_true", help="print the summary")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    summary = build(args.out)
    if args.print:
        print(json.dumps({k: v for k, v in summary.items() if k != "charts"}, indent=1, ensure_ascii=False))
    rep = summary["report"]
    print(f"Kept {summary['scores_kept']} assigned scores; left out {sum(summary['scores_left_out'].values())}.")
    if rep:
        print("Report evaluation: " + ", ".join(f"{c} {rep['conditions'][c]['mean']:.2f}" for c in CONDITIONS))
    print(f"Written to {args.out}")


if __name__ == "__main__":
    main()

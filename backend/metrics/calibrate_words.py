"""calibrate_words.py | A stricter words rule for the sanity check, calibrated before it is used (Compute.txt item 7).

Milestone 2 showed the words limit is too loose for a guard: it lets a genome's clip differ from the plain
render by 10 % of the words (at least 1), and 120 of the 483 genomes it passed had a word wrong against the
script. OBJECTIVE.md says the limit changes only after a new calibration. This is that calibration.

The rules compared (n = words in the line, fillers such as "uh" left out everywhere):

    now        what is heard in the clip differs from what is heard in the plain render by at most
               max(1, floor(0.10 n)) words                                  (the current rule)
    one_word   ... by at most 1 word, however long the line
    same       ... by no word at all
    no_new     the clip gets no more words wrong against the SCRIPT than its plain render does: a genome
               may not break a word the plain voice says right       (the proposed rule)

A rule must pass what chance alone does: two plain renders of the same voice and line, with different
seeds, are each other's baseline. The proposed rule is kept only if it lets at least 95 % of these plain
pairs through (CHANCE); otherwise the strictest rule that does is suggested.

The plain renders: Milestone 2's six lines (the lines the guard is judged on), new validation voices
(pick seed 1, not the 6 Nov ones), several seeds. The 6 Nov calibration's 400 pairs (output/calibration/
sanity) are counted too. Then every rule is applied to the 720 genomes of Milestone 2 (random-v2), with
the transcripts saved in their runs, to show what it would have let through.

    python -m backend.metrics.calibrate_words                 # render, transcribe, compare (about 1 hour on the GPU)
    python -m backend.metrics.calibrate_words --measure-only  # compare what is there

Renders go to OUTPUT_DIR/calibration/words/ (never committed); a stopped run continues where it was.
The results are in summary.json there; the limit is put in RewardConfig by hand, after a decision.
"""
import argparse
import glob
import itertools
import json
import math
from pathlib import Path

import numpy as np

from backend import model
from backend.metrics import calibrate_sanity as cs

OUT_DIR = model.OUTPUT_DIR / "calibration" / "words"
OLD_DIR = model.PROJECT_DIR / "output" / "calibration" / "sanity"
MILESTONE2 = model.PROJECT_DIR / "output" / "milestone2" / "random-v2" / "runs"
LINES = [
    "I can't believe we actually pulled that off.",
    "Please close the door on your way out.",
    "The meeting has been moved to Thursday afternoon, so we still have time to prepare.",
    "If you had told me this morning how the day would end, I would never have believed you.",
    "We should have left the house an hour ago, but nobody could find the keys, and now the last train is gone.",
    "Well, you really did it this time, and I honestly didn't think you had it in you, so I'll be waiting.",
]
VOICES = 9                 # 3 per dataset
SEEDS = 4                  # 12 ordered pairs per voice and line
PICK_SEED = 1              # other voices than the 6 Nov calibration (pick seed 0)
CHANCE = cs.CHANCE         # a rule must let 95 % of plain pairs through
RULES = ("now", "one_word", "same", "no_new")


# --------------------------------------------------------------------------- the rules (pure, tested)
def verdicts(script, base_heard, test_heard, config=None):
    """{rule: passes} for one clip against its plain render."""
    from backend.metrics.reward import CONFIG, sanity_words_allowed
    from backend.metrics.WER import word_differences

    config = config or CONFIG
    between = word_differences(base_heard, test_heard)
    test_wrong = word_differences(script, test_heard)["errors"]
    base_wrong = word_differences(script, base_heard)["errors"]
    return {
        "now": between["errors"] <= sanity_words_allowed(between["words"], config),
        "one_word": between["errors"] <= 1,
        "same": between["errors"] == 0,
        "no_new": test_wrong <= base_wrong,
        "_test_wrong": test_wrong, "_base_wrong": base_wrong, "_words": len(script.split()),
    }


def share(rows, rule):
    return sum(r[rule] for r in rows) / len(rows) if rows else None


def interval(rows, rule, z=1.96):
    """Wilson 95 % interval of a pass share."""
    n = len(rows)
    if not n:
        return None
    p = share(rows, rule)
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


# --------------------------------------------------------------------------- the data
def plain_pairs(out_dir, plan, heard):
    """Every ordered pair of plain renders of one voice and line, judged by every rule."""
    rows = []
    for voice in plan["voices"]:
        for line, script in enumerate(plan["lines"]):
            keys = [cs.clip_path(out_dir, voice, line, s).relative_to(out_dir).as_posix() for s in plan["seeds"]]
            keys = [k for k in keys if k in heard]
            for base, test in itertools.permutations(keys, 2):
                rows.append({"voice": voice["name"], "line": line,
                             **verdicts(script, heard[base]["heard"], heard[test]["heard"])})
    return rows


def old_pairs():
    """The 6 Nov calibration's pairs (2 lines, 10 other voices), judged by every rule."""
    plan_path, heard_path = OLD_DIR / "plan.json", OLD_DIR / "heard.json"
    if not plan_path.is_file() or not heard_path.is_file():
        return []
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    return plain_pairs(OLD_DIR, plan, json.loads(heard_path.read_text(encoding="utf-8")))


def milestone2_genomes(runs_dir=MILESTONE2):
    """Milestone 2's genomes with their transcripts: each one judged by every rule, with its other checks."""
    rows = []
    for path in sorted(glob.glob(str(Path(runs_dir) / "*" / "run.json"))):
        run = json.loads(Path(path).read_text(encoding="utf-8"))
        for g in run["genomes"]:
            s = g.get("sanity")
            if not s:
                continue
            rows.append({"run": run["id"], "genome": g["number"], "line": run["text"],
                         "other_checks_ok": bool(s["similarity_ok"] and s["pace_ok"]), "fine_now": g["verdict"] == "fine",
                         **verdicts(run["text"], s["baseline_heard"], s["test_heard"])})
    return rows


# --------------------------------------------------------------------------- the summary
def summarise(new, old, genomes):
    plain = new + old
    out = {"chance": CHANCE, "plain_pairs": {"new": len(new), "6 Nov": len(old), "all": len(plain)}, "rules": {}}
    passed_other = [g for g in genomes if g["other_checks_ok"]]
    for rule in RULES:
        by_line = {}
        for line in sorted({r["line"] for r in new}):
            rows = [r for r in new if r["line"] == line]
            by_line[LINES[line][:40]] = round(share(rows, rule), 3)
        fine = [g for g in passed_other if g[rule]]
        out["rules"][rule] = {
            "plain_pass": share(plain, rule), "plain_pass_95": interval(plain, rule),
            "plain_pass_new_by_line": by_line,
            "milestone2_fine": len(fine), "milestone2_of": len(genomes),
            "milestone2_fine_with_a_word_wrong": sum(g["_test_wrong"] > 0 for g in fine),
            "milestone2_fine_breaking_a_word": sum(g["_test_wrong"] > g["_base_wrong"] for g in fine),
            "milestone2_fine_wer": float(np.mean([g["_test_wrong"] / g["_words"] for g in fine])) if fine else None,
        }
    passing = [r for r in RULES if out["rules"][r]["plain_pass"] >= CHANCE]
    strict = sorted(passing, key=lambda r: out["rules"][r]["milestone2_fine"])
    out["suggested"] = "no_new" if "no_new" in passing else (strict[0] if strict else "now")
    out["passing_chance"] = passing
    return out


def show(summary):
    print(f"\nPlain pairs: {summary['plain_pairs']}  (a rule must pass {summary['chance']:.0%} of them)\n")
    print(f"{'rule':10s} {'plain pass':>11s} {'95% interval':>15s} {'M2 fine':>8s} {'word wrong':>11s} "
          f"{'breaks a word':>14s} {'fine WER':>9s}")
    for rule, r in summary["rules"].items():
        lo, hi = r["plain_pass_95"]
        print(f"{rule:10s} {r['plain_pass']:11.1%} {lo:7.1%}-{hi:6.1%} {r['milestone2_fine']:5d}/{r['milestone2_of']:<3d}"
              f"{r['milestone2_fine_with_a_word_wrong']:11d} {r['milestone2_fine_breaking_a_word']:14d} "
              f"{r['milestone2_fine_wer']:9.3f}")
    print(f"\nRules that pass chance: {', '.join(summary['passing_chance']) or 'none'}; suggested: {summary['suggested']}")


def main(voices=VOICES, seeds=SEEDS, pick_seed=PICK_SEED, measure_only=False, out_dir=OUT_DIR):
    from backend.metrics.reward import CONFIG

    out_dir = Path(out_dir)
    plan_path = out_dir / "plan.json"
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    else:
        plan = cs.plan_of(out_dir, voices, seeds, pick_seed)     # validation voices only
        plan["lines"] = LINES
        cs._save(plan_path, plan)
    if not measure_only:
        cs.render_all(out_dir, plan)
    heard = cs.measure_all(out_dir, plan, CONFIG)
    new, old, genomes = plain_pairs(out_dir, plan, heard), old_pairs(), milestone2_genomes()
    if not new and not old:
        raise SystemExit("Nothing to compare yet: render first.")
    summary = summarise(new, old, genomes)
    cs._save(out_dir / "summary.json", summary)
    show(summary)
    print(f"\nSaved in {out_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate a stricter words rule for the sanity check.")
    parser.add_argument("--voices", type=int, default=VOICES)
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--pick-seed", type=int, default=PICK_SEED)
    parser.add_argument("--measure-only", action="store_true")
    args = parser.parse_args()
    main(args.voices, args.seeds, args.pick_seed, args.measure_only)

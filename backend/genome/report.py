"""report.py | Answers the genome questions from the listener ratings.

    python -m backend.genome.report                 # the six answers, printed
    python -m backend.genome.report --csv FILE      # also one row per rated genome as CSV,
                                                    # and a manifest.json next to it

The manifest records the hash of every ratings file and of the review log, the
code commit, the date and the settings, so the same numbers can be reproduced.

Scores are MOS on 0-5 (see ratings.py). A genome beats the plain line when its
MOS is higher than the plain line's MOS for the same run. Only genomes with at
least MIN_LISTENERS scores are used in the answers; the rest are counted and left out.

GOOD, BAD and MIN_LISTENERS are working choices. They must be justified in the
report, for example from the spread of the scores, before they are used to decide.
"""
import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean

from backend import model
from backend.genome import ratings as R
from backend.genome import search as S

GOOD = 4.0
BAD = 2.5
MIN_LISTENERS = 2
FIELDS = ["voice", "run", "genome", "features", "verdict", "mos", "n", "low", "high",
          "above_baseline", "baseline_mos", "markup"]


def rated_genomes():
    rows = []
    for folder, run in S.list_runs():
        if not (folder / R.RATINGS_FILE).is_file():
            continue
        rank = R.ranked(folder, run)
        base = rank["baseline"]
        for g in rank["genomes"]:
            if not g["mos"]:
                continue
            rows.append({
                "voice": run["voice"], "run": run["id"], "genome": g["number"],
                "features": len(g["genes"]), "verdict": g["verdict"],
                "mos": g["mos"]["mos"], "n": g["mos"]["n"], "low": g["mos"]["low"], "high": g["mos"]["high"],
                "above_baseline": g["above_baseline"], "baseline_mos": base["mos"] if base else None,
                "markup": g["markup"], "genes": json.dumps(g["genes"], sort_keys=True),
            })
    return rows


def _hash_files(paths):
    h = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p)):
        h.update(str(path.name).encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()


def _code_commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=model.PROJECT_DIR, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def write_manifest(path, rows):
    rating_files = [folder / R.RATINGS_FILE for folder, _ in S.list_runs()
                    if (folder / R.RATINGS_FILE).is_file()]
    log = R.LOG_PATH if R.LOG_PATH.is_file() else None
    manifest = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "runs_with_ratings": len(rating_files),
        "genomes_in_report": len(rows),
        "ratings_sha256": _hash_files(rating_files) if rating_files else None,
        "review_log_sha256": _hash_files([log]) if log else None,
        "code_commit": _code_commit(),
        "settings": {"GOOD": GOOD, "BAD": BAD, "MIN_LISTENERS": MIN_LISTENERS},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def show(title, lines):
    print(f"\n{title}")
    for line in lines or ["  (no data yet)"]:
        print(f"  {line}")


def describe(r):
    return f"{r['voice']}, genome {r['genome']} in {r['run']}: MOS {r['mos']:.2f} ({r['n']} listeners)  {r['markup']}"


def answers(rows):
    ok = [r for r in rows if r["n"] >= MIN_LISTENERS]
    print(f"{len(rows)} rated genomes, {len(ok)} with at least {MIN_LISTENERS} listeners "
          f"({len(rows) - len(ok)} left out, too few listeners).")

    # 1. Which genomes work best for each voice?
    best = {}
    for r in ok:
        if r["voice"] not in best or r["mos"] > best[r["voice"]]["mos"]:
            best[r["voice"]] = r
    show("1. Best genome per voice", [describe(r) for _, r in sorted(best.items())])

    # 2. Which genomes produce artifacts?
    # Listener comments are not stored, so the automatic sanity check is the artifact flag.
    flagged = [r for r in ok if r["verdict"] != "fine"]
    show("2. Genomes that produce artifacts (flagged by the sanity check)", [describe(r) for r in flagged])

    # 3. Which configurations give bad MOS?
    bad = [r for r in ok if r["mos"] < BAD]
    show(f"3. Bad MOS (below {BAD})", [describe(r) for r in bad])

    # 4. Which configurations are consistently good?
    by_config = defaultdict(list)
    for r in ok:
        by_config[r["genes"]].append(r)
    consistent = []
    for group in by_config.values():
        good_voices = {r["voice"] for r in group if r["mos"] >= GOOD}
        if len(good_voices) >= 2:
            consistent.append(f"good for {len(good_voices)} voices, mean MOS {mean(r['mos'] for r in group):.2f}: "
                              f"{group[0]['markup']}")
    show(f"4. Good (MOS >= {GOOD}) for at least 2 voices", consistent)

    # 5. How often does a genome beat its plain line, per number of features?
    by_size = defaultdict(lambda: [0, 0])
    for r in ok:
        if r["above_baseline"] is None:
            continue
        by_size[r["features"]][1] += 1
        if r["above_baseline"] > 0:
            by_size[r["features"]][0] += 1
    show("5. Beats the plain line, by number of features",
         [f"{k} features: {wins} of {total} ({wins / total:.0%})" for k, (wins, total) in sorted(by_size.items())])

    # 6. Which genomes can we exclude?
    excluded = {(r["voice"], r["run"], r["genome"]): r for r in bad + flagged}
    show("6. Can be excluded (bad MOS or flagged as artifacts)", [describe(r) for r in excluded.values()])


def main():
    parser = argparse.ArgumentParser(description="Answer the genome questions from the listener ratings.")
    parser.add_argument("--csv", metavar="FILE", help="write one row per rated genome as CSV, and a manifest.json beside it")
    args = parser.parse_args()

    rows = rated_genomes()
    if args.csv:
        csv_path = Path(args.csv)
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        manifest_path = csv_path.parent / "manifest.json"
        write_manifest(manifest_path, rows)
        print(f"{len(rows)} rows written to {csv_path}, manifest written to {manifest_path}")
    answers(rows)


if __name__ == "__main__":
    main()
"""report.py | The six genome questions, answered in text from the evidence snapshot (evidence_snapshot.py).

    python main_backend.py genome-report            # coverage, provenance and the six answers, printed
    python main_backend.py genome-report --csv FILE # also one row per genome as CSV, the snapshot as
                                                    # evidence.json and a manifest.json next to it

Every printout and file starts with what the numbers rest on: the snapshot_id, the newest rating used,
the code commit, the aggregation rule, and the coverage check (genomes in the rated runs against genomes
with a score, listing any without one). A report made while rating is still going on therefore says so:
on 11 Nov at 18:40, 227 of the 240 genomes of mos-diverse-v1 had a score; the last 13 were rated later.

A winner is named only when the snapshot's WINNER_RULE allows it; otherwise the answer says "not enough
evidence" or "no clear winner", with the number of listeners. A genome is only recommended after three
artifact checks (sanity, audio, listeners) and when it can still be rendered (see evidence_snapshot.py).

GOOD and BAD are working choices for questions 3 and 4, not measured thresholds; they are printed with
the answers. Test accounts' scores (Test user N) are not counted: ratings.mos leaves them out.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

from backend.genome import evidence_snapshot as E
from backend.genome import search as S      # tests patch S.list_runs

GOOD = 4.0
BAD = 2.5
MIN_LISTENERS = E.MIN_LISTENERS
FIELDS = ["snapshot_id", "voice", "run", "genome", "features", "gene_types", "mos", "n", "listeners", "low", "high",
          "above_baseline", "baseline_mos", "sanity", "audio_check", "artifact_heard", "artifact_answers",
          "unusable", "safe", "blocked_by", "recommendable", "markup"]


def rated_genomes(snapshot=None):
    """One row per rated genome, from the snapshot (built from search.list_runs() when not given)."""
    snapshot = snapshot or E.build(S.list_runs())
    plain = {v["voice"]: v["plain_mos"] for v in snapshot["voices"]}
    rows = []
    for g in snapshot["genomes"]:
        if not g["rated"]:
            continue
        gone = next((b.split(": ", 1)[1] for b in g["blocked_by"] if b.startswith("can no longer be rendered")), "")
        rows.append({**g, "snapshot_id": snapshot["provenance"]["snapshot_id"],
                     "above_baseline": g["above_plain"], "baseline_mos": plain.get(g["voice"]), "unusable": gone,
                     "verdict": g["sanity"], "listeners": ";".join(g["listeners"]), "gene_types": ";".join(g["gene_types"]),
                     "audio_check": "ok" if g["audio_check"]["ok"] else ",".join(g["audio_check"]["failed"]),
                     "blocked_by": "; ".join(g["blocked_by"])})
    return rows


def write_manifest(path, rows, snapshot):
    p, c = snapshot["provenance"], snapshot["coverage"]
    manifest = {
        "snapshot_id": p["snapshot_id"], "created": p["created"], "code_commit": p["code_commit"],
        "ratings_up_to": p["ratings_up_to"], "rating_log_sha256": p["rating_log_sha256"], "runs": p["runs"],
        "genomes_in_runs": c["genomes_in_runs"], "genomes_in_report": len(rows),
        "genomes_without_a_score": c["genomes_unrated"], "scores_used": c["scores_used"],
        "listeners_per_genome": c["listeners_per_genome"], "rule": p["rule"], "winner_rule": p["winner_rule"],
        "settings": {**p["settings"], "GOOD": GOOD, "BAD": BAD}, "caveats": snapshot["caveats"],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def show(title, lines):
    print(f"\n{title}")
    for line in lines or ["  (none)"]:
        print(f"  {line}")


def describe(r):
    note = f"  [can no longer be rendered: {r['unusable']}]" if r.get("unusable") else ""
    who = f"{r['n']} listener{'s' if r['n'] != 1 else ''}"
    spread = f", 95% interval {r['low']}-{r['high']}" if r.get("low") is not None else ", no interval"
    return f"{r['voice']}, genome {r['genome']} in {r['run']}: MOS {r['mos']:.2f} ({who}{spread})  {r['markup']}{note}"


def header(snapshot, rows):
    p, c = snapshot["provenance"], snapshot["coverage"]
    print(f"Evidence snapshot {p['snapshot_id']} (made {p['created']}, code {p['code_commit'][:10]}, "
          f"ratings up to {p['ratings_up_to']})")
    status = "complete" if c["complete"] else f"INCOMPLETE: {len(c['genomes_unrated'])} without a score"
    print(f"Coverage: {c['genomes_rated']} of {c['genomes_in_runs']} genomes in {c['runs_with_ratings']} rated runs have "
          f"a score ({status}); {len(rows)} report rows; {c['scores_used']} scores used "
          f"({c['test_account_scores_left_out']} test-account scores left out).")
    for g in c["genomes_unrated"]:
        print(f"  no score yet: {g['voice']}, genome {g['genome']} in {g['run']}")
    print("Listeners per genome: " + ", ".join(f"{k} listener(s): {v} genomes" for k, v in c["listeners_per_genome"].items()))
    print(f"Rule: {p['rule']}")
    for line in snapshot["caveats"]:
        print(f"Caveat: {line}")


def answers(rows, snapshot):
    # 1. Which genomes work best for each voice?
    lines = []
    for v in snapshot["voices"]:
        top = v["candidates"][0] if v["candidates"] else None
        lead = (f"; highest safe: genome {top['genome']}, MOS {top['mos']} ({top['n']} listener(s))" if top else "")
        lines.append(f"{v['voice']}: {v['status'].replace('_', ' ')}{lead}. {v['reason']}")
    show(f"1. Best genome per voice ({E.WINNER_RULE})", lines)

    # 2. Which genomes produce artifacts? Sanity check, audio check and listeners.
    show("2. Genomes that fail an artifact check (sanity, audio, or a listener heard one)",
         [f"{describe(r)}\n      blocked by: {r['blocked_by']}" for r in rows
          if r["sanity"] != "fine" or r["audio_check"] != "ok" or r["artifact_heard"]])

    # 3. Which configurations give bad MOS?
    show(f"3. Bad MOS (below {BAD}; a working choice)", [describe(r) for r in rows if r["mos"] < BAD])

    # 4. Which configurations are consistently good?
    by_config = defaultdict(list)
    for r in rows:
        by_config[r["markup"]].append(r)
    consistent = []
    for group in by_config.values():
        good_voices = {r["voice"] for r in group if r["mos"] >= GOOD}
        if len(good_voices) >= 2:
            consistent.append(f"good for {len(good_voices)} voices, mean MOS {mean(r['mos'] for r in group):.2f}: "
                              f"{group[0]['markup']}")
    show(f"4. Good (MOS >= {GOOD}) for at least 2 voices", consistent)

    # 5. How often does a genome beat its plain line, per number of features?
    by_size = defaultdict(lambda: [0, 0, 0])
    for r in rows:
        if r["above_baseline"] is None:
            continue
        by_size[r["features"]][1] += 1
        by_size[r["features"]][2] += r["n"]
        if r["above_baseline"] > 0:
            by_size[r["features"]][0] += 1
    show("5. Beats the plain line (same listener), by number of features",
         [f"{k} features: {wins} of {total} genomes ({wins / total:.0%}), {scores} scores"
          + ("  [fewer than 20 genomes]" if total < 20 else "")
          for k, (wins, total, scores) in sorted(by_size.items())])

    # 6. Which genomes can we exclude?
    excluded = {(r["voice"], r["run"], r["genome"]): r for r in rows if r["mos"] < BAD or not r["safe"]}
    show("6. Can be excluded (bad MOS, an artifact check failed, or no longer renderable)",
         [describe(r) for r in excluded.values()])
    print(f"\nRecommendable now (all checks passed and at least {MIN_LISTENERS} listeners): "
          f"{sum(1 for r in rows if r['recommendable'])} of {len(rows)} genomes.")


def main():
    parser = argparse.ArgumentParser(description="Answer the genome questions from the evidence snapshot.")
    parser.add_argument("--csv", metavar="FILE", help="write one row per rated genome as CSV, with evidence.json and "
                                                      "manifest.json beside it")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")     # the markup's pause marks on a Windows console

    snapshot = E.build(S.list_runs())
    rows = rated_genomes(snapshot)
    header(snapshot, rows)
    print(f"Snapshot written to {E.write(snapshot)}")
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        E.write(snapshot, csv_path.parent / "evidence.json")
        write_manifest(csv_path.parent / "manifest.json", rows, snapshot)
        print(f"{len(rows)} rows written to {csv_path}, with evidence.json and manifest.json beside it")
    answers(rows, snapshot)


if __name__ == "__main__":
    main()

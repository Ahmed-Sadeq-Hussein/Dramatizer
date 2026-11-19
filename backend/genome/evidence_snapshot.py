"""evidence_snapshot.py | The analysis step: what the listener ratings show, with how sure it is and where it came from.

    python main_backend.py genome-report --csv FILE     # writes this snapshot as evidence.json beside the CSV

This is the one place that turns ratings into conclusions. The assistant (chatbot, MCP, API) does not
decide anything: it fetches this snapshot (GET /evidence/snapshot) and explains it, citing its snapshot_id.

What a snapshot holds:

    provenance   snapshot_id (a hash of the contents), when it was made, the code commit, the newest rating
                 used (ratings_up_to), hashes of the rating log and the runs' ratings, the runs, every
                 setting, and the aggregation RULE in words
    coverage     genomes in the rated runs against genomes with a score, and which ones have none, so a
                 report with fewer rows than the runs have genomes is explained, not silent; how many
                 listeners heard each genome; test accounts' scores left out
    genomes      one row per genome: score, listeners, interval, artifact checks, whether it may be recommended
    voices       per voice: a winner only when the evidence allows one (WINNER_RULE), else why not
    caveats      what limits every conclusion, in words

Artifact checks, all three must pass before a score is used to recommend a genome:
    1. the sanity check (words, voice, pace kept; search.py)
    2. the audio check (clipping, bursts, gaps, empty sound; artifact_check.py)
    3. no listener said they heard an artifact (when the question was asked)
A genome that uses a gene type disabled since, or a speed outside 0.85-1.15, cannot be rendered again
(editor.unusable) and is never recommended either.
"""
import hashlib
import json
import math
import statistics as st
import subprocess
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from backend import model, safe_files
from backend.genome import artifact_check
from backend.genome import ratings as R
from backend.genome import search as S
from backend.genome.editor import unusable

SCHEMA_VERSION = 1
SNAPSHOT_DIR = model.PROJECT_DIR / "history" / "evidence"      # git-ignored: it names the listeners
MIN_LISTENERS = 2           # a winner needs at least this many listeners on it and on the runner-up
CONFIDENCE = 0.95
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
RULE = ("A genome's score is the mean of its listeners' 0-5 scores (test accounts left out). 'Above plain' is "
        "each listener's genome score minus that same listener's score for the plain line of the voice, averaged "
        "over the listeners. The interval is a 95% t-interval over the listeners' scores; with one listener there "
        "is no interval.")
WINNER_RULE = (f"A voice has a winner only when its top safe genome was scored by at least {MIN_LISTENERS} "
               "listeners and the lower end of its 95% interval is above both the plain line and the next-best safe "
               "genome. Otherwise the answer is 'not enough evidence' or 'no clear winner', with the counts.")


def _interval(values):
    """(low, high) 95% t-interval of the mean, or (None, None) with fewer than 2 values."""
    if len(values) < 2:
        return None, None
    m, sd = st.mean(values), st.stdev(values)
    t = T95.get(len(values) - 1, 1.96)
    half = t * sd / math.sqrt(len(values))
    return round(m - half, 2), round(m + half, 2)


def _commit():
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=model.PROJECT_DIR, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _genome_row(run, genome, ratings, plain, audio):
    clip = R.clip_key(genome["number"])
    score = R.mos(ratings, clip)
    scores = score["scores"] if score else {}
    above = [s - plain["scores"][who] for who, s in scores.items() if plain and who in plain["scores"]]
    low, high = _interval(list(scores.values()))
    sound = audio.get(clip) or {"ok": False, "failed": ["not checked"]}
    heard = (score or {}).get("artifact_heard", 0)
    answered = (score or {}).get("artifact_answers", 0)
    blocked = []
    if genome.get("verdict") != "fine":
        blocked.append(f"sanity check: {genome.get('verdict') or 'not run'}")
    if not sound["ok"]:
        blocked.append("audio check: " + ", ".join(sound["failed"]))
    if heard:
        blocked.append(f"{heard} of {answered} listeners heard an artifact")
    gone = unusable(genome["genes"])
    if gone:
        blocked.append(f"can no longer be rendered: {gone}")
    return {
        "voice": run["voice"], "run": run["id"], "genome": genome["number"], "clip": clip,
        "markup": genome.get("markup"), "features": len(genome["genes"]),
        "gene_types": sorted({g["feature"] for g in genome["genes"]}),
        "rated": bool(scores), "listeners": sorted(scores), "n": len(scores),
        "scores": scores, "mos": score["mos"] if score else None, "low": low, "high": high,
        "above_plain": round(st.mean(above), 2) if above else None,
        "sanity": genome.get("verdict"), "audio_check": sound,
        "artifact_heard": heard, "artifact_answers": answered,
        "safe": not blocked, "blocked_by": blocked,
        "recommendable": not blocked and len(scores) >= MIN_LISTENERS,
    }


def _voice_verdict(voice, rows, plain_mos):
    rated = [r for r in rows if r["rated"]]
    safe = sorted((r for r in rated if r["safe"]), key=lambda r: (-r["mos"], -r["n"], r["genome"]))
    out = {"voice": voice, "runs": sorted({r["run"] for r in rows}), "genomes_rated": len(rated),
           "safe_genomes": len(safe), "plain_mos": plain_mos,
           "listeners": sorted({who for r in rated for who in r["listeners"]}),
           "candidates": [{k: r[k] for k in ("run", "genome", "mos", "n", "low", "high", "above_plain", "listeners", "markup")}
                          for r in safe[:3]]}
    if not rated:
        return {**out, "status": "no_ratings", "reason": "No genome of this voice has a score."}
    if not safe:
        return {**out, "status": "no_safe_genome",
                "reason": f"All {len(rated)} rated genomes fail an artifact check or can no longer be rendered."}
    top, rest = safe[0], safe[1:]
    runner = rest[0] if rest else None
    tied = [r["genome"] for r in safe if r["mos"] == top["mos"]]
    if top["n"] < MIN_LISTENERS or (runner and runner["n"] < MIN_LISTENERS):
        few = min(r["n"] for r in [top] + ([runner] if runner else []))
        return {**out, "status": "not_enough_evidence", "tied_at_top": tied if len(tied) > 1 else [],
                "reason": (f"The top safe genomes were scored by {few} listener(s); a winner needs at least "
                           f"{MIN_LISTENERS} on each, so there is no interval to compare."
                           + (f" {len(tied)} genomes share the top score." if len(tied) > 1 else ""))}
    beats_plain = plain_mos is None or top["low"] > plain_mos
    beats_next = runner is None or top["low"] > runner["mos"]
    if beats_plain and beats_next:
        return {**out, "status": "winner", "winner": out["candidates"][0],
                "reason": f"Genome {top['genome']}: 95% interval {top['low']}-{top['high']} over {top['n']} listeners, "
                          f"above the plain line ({plain_mos}) and the next safe genome ({runner['mos'] if runner else 'none'})."}
    return {**out, "status": "no_clear_winner",
            "reason": f"Genome {top['genome']} scores highest ({top['mos']}, {top['n']} listeners), but its 95% interval "
                      f"({top['low']}-{top['high']}) does not clear "
                      + ("the plain line" if not beats_plain else f"the next safe genome ({runner['mos']})") + "."}


def build(runs=None):
    """The snapshot of every run that has ratings (runs: [(folder, run)], default search.list_runs())."""
    runs = sorted(((f, r) for f, r in (runs if runs is not None else S.list_runs()) if not r.get("sealed")),
                  key=lambda item: item[1]["id"])        # sealed: the Report evaluation, analysed on its own
    genomes, voices, plains, run_ids = [], defaultdict(list), {}, set()
    test_scores = 0
    # Second-listener runs (second_listen.py) hold the same audio as an original run: their scores are added to
    # the original genomes, not counted as new genomes. A listener already in the original keeps their first score.
    originals = {r["id"] for _, r in runs if not r.get("second_listen_of")}
    extra, repeated = defaultdict(list), []
    for folder, run in runs:
        if run.get("second_listen_of") in originals:
            more = R.load(folder)
            if more["scores"]:
                extra[run["second_listen_of"]].append(more)
                run_ids.add(run["id"])
    for folder, run in runs:
        if run.get("second_listen_of"):
            continue
        ratings = R.load(folder)
        for more in extra.get(run["id"], []):
            ratings = {**ratings, "scores": {**ratings["scores"]}, "artifacts": {**ratings.get("artifacts", {})}}
            for who, clips in more["scores"].items():
                if who in ratings["scores"]:
                    repeated.append(f"{who} in {run['id']}")
                    continue
                ratings["scores"][who] = clips
                if who in more.get("artifacts", {}):
                    ratings["artifacts"][who] = more["artifacts"][who]
        if not ratings["scores"]:
            continue
        run_ids.add(run["id"])
        test_scores += sum(len(clips) for who, clips in ratings["scores"].items() if R.is_test_rater(who))
        plain = R.mos(ratings, "baseline")
        plains[run["id"]] = plain
        audio = artifact_check.check_run(folder, run)
        for genome in run["genomes"]:
            row = _genome_row(run, genome, ratings, plain, audio)
            genomes.append(row)
            voices[run["voice"]].append(row)

    rated = [g for g in genomes if g["rated"]]
    per_listener = defaultdict(list)
    for g in rated:
        for who, s in g["scores"].items():
            per_listener[who].append(s)
    log = [e for e in R.read_log() if e["run"] in run_ids]
    coverage = {
        "runs_with_ratings": len(run_ids) - sum(len(v) for v in extra.values()),
        "second_listener_runs": sum(len(v) for v in extra.values()),
        "genomes_in_runs": len(genomes),
        "genomes_rated": len(rated),
        "genomes_unrated": [{"voice": g["voice"], "run": g["run"], "genome": g["genome"]} for g in genomes if not g["rated"]],
        "plain_lines_rated": sum(1 for p in plains.values() if p),
        "scores_used": sum(g["n"] for g in rated) + sum(p["n"] for p in plains.values() if p),
        "test_account_scores_left_out": test_scores,
        "listeners_per_genome": {str(k): v for k, v in sorted(Counter(g["n"] for g in rated).items())},
        "listeners": {who: {"genome_scores": len(s), "mean": round(st.mean(s), 2)} for who, s in sorted(per_listener.items())},
    }
    coverage["complete"] = not coverage["genomes_unrated"]
    voice_rows = []
    for voice, rows in sorted(voices.items()):
        plain_scores = [plains[r]["mos"] for r in {g["run"] for g in rows} if plains.get(r)]
        voice_rows.append(_voice_verdict(voice, rows, round(st.mean(plain_scores), 2) if plain_scores else None))

    caveats = []
    one = coverage["listeners_per_genome"].get("1", 0)
    if one:
        caveats.append(f"{one} of {len(rated)} rated genomes were heard by one listener only: their score is that "
                       "listener's opinion, with no interval, and cannot name a winner.")
    if not any(g["artifact_answers"] for g in rated):
        caveats.append("No listener answered the artifact question for these genomes (it was added after they were "
                       "rated), so artifacts are judged by the automatic checks alone.")
    means = [v["mean"] for v in coverage["listeners"].values()]
    if len(means) > 1 and max(means) - min(means) >= 1:
        caveats.append("Listeners use the scale differently (mean genome score "
                       + ", ".join(f"{w} {v['mean']}" for w, v in coverage["listeners"].items())
                       + "); where each voice had a different listener, comparing voices also compares listeners.")
    if not coverage["complete"]:
        caveats.append(f"{len(coverage['genomes_unrated'])} of {len(genomes)} genomes have no score yet; "
                       "answers cover the rated ones only.")
    if repeated:
        caveats.append(f"{len(repeated)} second-listener runs were rated by a listener who had already rated the "
                       "original (" + "; ".join(repeated[:5]) + "): only their first scores are counted.")

    body = {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "code_commit": _commit(),
            "ratings_up_to": max((e["time"] for e in log), default=None),
            "rating_log_entries": len(log),
            "rating_log_sha256": _sha("".join(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n" for e in log)),
            "runs": sorted(run_ids),
            "batches": sorted({r.get("batch") for _, r in runs if r["id"] in run_ids and r.get("batch")}),
            "rule": RULE, "winner_rule": WINNER_RULE,
            "settings": {"MIN_LISTENERS": MIN_LISTENERS, "CONFIDENCE": CONFIDENCE,
                         "artifact_limits": artifact_check.LIMITS},
        },
        "coverage": coverage, "voices": voice_rows, "genomes": genomes, "caveats": caveats,
    }
    snapshot_id = _sha(json.dumps(body, sort_keys=True, ensure_ascii=False))[:16]
    body["provenance"] = {"snapshot_id": snapshot_id, "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                          **body["provenance"]}
    return body


def write(snapshot, path=None):
    """Write the snapshot (default history/evidence/evidence-<id>.json and latest.json); returns the path."""
    path = Path(path) if path else SNAPSHOT_DIR / f"evidence-{snapshot['provenance']['snapshot_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_files.write_json(path, snapshot)
    if path.parent == SNAPSHOT_DIR:
        safe_files.write_json(SNAPSHOT_DIR / "latest.json", snapshot)
    return path


def summary(snapshot, voice=None):
    """The snapshot without the per-genome rows, for the assistant: provenance, coverage, voices, caveats."""
    voices = [v for v in snapshot["voices"] if voice is None or v["voice"].lower() == voice.lower()]
    if voice is None:                    # every voice: short, so it fits an assistant's tool result
        voices = [{"voice": v["voice"], "status": v["status"], "plain_mos": v["plain_mos"],
                   "highest_safe": {k: v["candidates"][0][k] for k in ("genome", "mos", "n")} if v["candidates"] else None,
                   **({"winner": v["winner"]} if "winner" in v else {})} for v in voices]
    status = Counter(v["status"] for v in snapshot["voices"])
    provenance = {k: v for k, v in snapshot["provenance"].items() if k != "runs"}
    provenance["runs"] = len(snapshot["provenance"]["runs"])      # the ids are in the snapshot file
    return {"provenance": provenance, "coverage": snapshot["coverage"], "caveats": snapshot["caveats"],
            "voice_status": dict(status), "voices": voices,
            "recommendable_genomes": sum(g["recommendable"] for g in snapshot["genomes"]),
            "blocked_by_artifact_checks": sum(1 for g in snapshot["genomes"] if g["rated"] and not g["safe"])}

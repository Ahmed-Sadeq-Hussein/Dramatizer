"""compare_search.py | Milestone 2: does a random genome search beat the baseline? (MILESTONE_2.md)

    python main_backend.py compare-search --run          # search every voice and line of baseline v2, then compare
    python main_backend.py compare-search                # compare what is finished; render nothing
    python main_backend.py compare-search --run --voices 2   # a small trial on the first 2 voices

For every voice and line of a finished baseline (output/baseline/<baseline>/), a
random genome search is run (backend/genome/search.py): 10 random genomes of 3
genes, genome seed 0, the baseline's render seed. The baseline's own render is
the search's plain clip, copied in, because search.render_plan with an empty plan
and the same seed is exactly how the baseline made it. The same genome seed on
the same line gives the same 10 genomes in every voice, so voices are compared
on the same genomes.

Then:

    1. every rendered genome is measured like the baseline (WER against the line,
       similarity to the reference, SQUIM quality), so the two are comparable
    2. the comparison: share of genomes judged fine, why the others failed, per
       gene type, and the fine genomes' metrics against the baseline's (mean
       difference per voice and line, 95 % bootstrap interval over voices,
       Wilcoxon signed-rank test on the paired voice-and-line means)
    3. the verdicts decided in MILESTONE_2.md before running: does the search
       work as a guard?
    4. 10 blind listening pairs (A and B in random order, the key kept apart),
       chosen before anyone listens, for LISTENING_PROTOCOL.md

Everything goes in output/milestone2/<name>/ (never committed): runs.json (which
search folder belongs to which voice and line), runs/ (the searches), measured.json
(saved after every clip, so a stopped run continues), summary.json, record.json
and listening/. The comparison is logged in MLflow like the baseline.
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import baseline as base  # noqa: E402

OUT_DIR = PROJECT / "output" / "milestone2"
NAME = "random-v2"
BASELINE = "v2"
GENES = 3
GENOME_SEED = 0
LISTENING_PAIRS = 10
LISTENING_SEED = 0
SHORT_LINES = (0, 1)        # under 6 s of speech: the weaker similarity limit applies (MILESTONE_2.md, Risks)
METRICS = ["wer", "similarity", "pesq", "stoi", "si_sdr"]
LIMITATION = ("Random search only (10 genomes of 3 genes per voice and line, one genome seed, one render seed); "
              "automatic judges (Whisper base, Resemblyzer, SQUIM) cannot measure expressiveness, so 'beats the "
              "baseline' needs the blind listening pairs; validation voices only.")

_save, _load = base._save, base._load


# --------------------------------------------------------------------------- #
# The searches
# --------------------------------------------------------------------------- #

def key_of(voice, line):
    return f"{voice['name']}/line{line}"


def jobs_of(plan, voices=None):
    chosen = plan["voices"][:voices] if voices else plan["voices"]
    return [(voice, line) for voice in chosen for line in range(len(plan["lines"]))]


def prepare(out_dir, plan, baseline_dir, jobs, genes, genome_seed):
    """One search folder per voice and line, made once (runs.json); the baseline's render is the plain clip."""
    from backend.genome.search import load_run, new_run, save_run

    path = out_dir / "runs.json"
    runs = _load(path, {})
    for voice, line in jobs:
        key = key_of(voice, line)
        if key in runs:
            continue
        plain = base.clip_path(baseline_dir, voice, line)
        if not plain.is_file():
            raise SystemExit(f"The baseline has no render for {key}: run the baseline first.")
        run_dir = new_run(voice["reference_audio"], voice["reference_text"], text=plan["lines"][line],
                          voice=f"{voice['name']}_line{line}", genes=genes, seed=genome_seed,
                          render_seed=plan["render_seed"], origin=voice.get("origin"), runs_dir=out_dir / "runs")
        shutil.copyfile(plain, run_dir / "baseline.wav")
        run = load_run(run_dir)
        run["baseline"].update(audio="baseline.wav", copied_from=plain.relative_to(baseline_dir).as_posix(),
                              baseline=baseline_dir.name)
        save_run(run_dir, run)
        runs[key] = run_dir.relative_to(out_dir).as_posix()
        _save(path, runs)
    return runs


def search_all(out_dir, runs, jobs):
    from backend.genome.search import load_run, run_search, summary

    todo = [key_of(v, i) for v, i in jobs if load_run(out_dir / runs[key_of(v, i)])["status"] != "finished"]
    print(f"{len(jobs) - len(todo)} of {len(jobs)} searches already finished; running {len(todo)}.", flush=True)
    for number, key in enumerate(todo, start=1):
        started = time.time()
        run = run_search(out_dir / runs[key])
        counts = summary(run)
        print(f"  {number}/{len(todo)} {key}: {counts['fine']} fine, {counts['terrible']} terrible"
              + (f", {counts['failed']} failed" if counts["failed"] else "")
              + f" ({time.time() - started:.0f} s)", flush=True)


def measure_genomes(out_dir, plan, runs, jobs):
    """Every rendered genome measured the way the baseline's clips were (baseline.measure_all)."""
    from backend.genome.search import load_run
    from backend.metrics.measurements import measure

    path = out_dir / "measured.json"
    measured = _load(path, {})
    for voice, line in jobs:
        run_dir = out_dir / runs[key_of(voice, line)]
        run = load_run(run_dir)
        for genome in run["genomes"]:
            name = f"{key_of(voice, line)}/genome_{genome['number']:02d}"
            clip = run_dir / (genome.get("audio") or "")
            if name in measured or genome["verdict"] not in ("fine", "terrible") or not clip.is_file():
                continue
            metrics = measure(voice["reference_audio"], clip, plan["lines"][line],
                              reference_text=voice["reference_text"])["metrics"]
            measured[name] = {"wer": metrics["wer"], "heard": metrics["heard"], "similarity": metrics["similarity"],
                              "pesq": metrics["quality"]["pesq"], "stoi": metrics["quality"]["stoi"],
                              "si_sdr": metrics["quality"]["si_sdr"], "seconds": metrics["seconds"]}
            _save(path, measured)
        print(f"  measured {key_of(voice, line)}", flush=True)
    return measured


# --------------------------------------------------------------------------- #
# The comparison
# --------------------------------------------------------------------------- #

def rows_of(out_dir, plan, runs, jobs, measured):
    """One row per judged genome, with its verdict, reasons, genes and metrics."""
    from backend.genome.search import load_run

    rows = []
    for voice, line in jobs:
        run = load_run(out_dir / runs[key_of(voice, line)])
        for genome in run["genomes"]:
            if genome["verdict"] is None:
                continue
            sanity = genome.get("sanity") or {}
            rows.append({
                "voice": voice["name"], "dataset": voice["dataset"], "line": line, "genome": genome["number"],
                "verdict": genome["verdict"], "error": genome.get("error"),
                "features": [gene["feature"] for gene in genome["genes"]], "markup": genome["markup"],
                "speed": genome["speed"],
                "wer_ok": sanity.get("wer_ok"), "similarity_ok": sanity.get("similarity_ok"),
                "pace_ok": sanity.get("pace_ok"), "sanity_similarity": sanity.get("similarity"),
                "audio": (out_dir / runs[key_of(voice, line)] / genome["audio"]).as_posix() if genome.get("audio") else None,
                **(measured.get(f"{key_of(voice, line)}/genome_{genome['number']:02d}") or {}),
            })
    return rows


def _share(rows, test):
    return sum(1 for r in rows if test(r)) / len(rows) if rows else None


def judged_shares(rows):
    """Share fine, and why the rest failed (a genome can fail for more than one reason)."""
    judged = [r for r in rows if r["verdict"] in ("fine", "terrible")]
    terrible = [r for r in judged if r["verdict"] == "terrible"]
    return {
        "genomes": len(rows),
        "judged": len(judged),
        "failed_to_render": sum(r["verdict"] == "failed" for r in rows),
        "fine": _share(judged, lambda r: r["verdict"] == "fine"),
        "terrible_because": {
            "words": _share(terrible, lambda r: r["wer_ok"] is False),
            "similarity": _share(terrible, lambda r: r["similarity_ok"] is False),
            "pace": _share(terrible, lambda r: r["pace_ok"] is False),
        },
    }


def by_feature(rows):
    """For each gene type: in how many judged genomes it was, and how often those were fine."""
    counts = {}
    for row in rows:
        if row["verdict"] not in ("fine", "terrible"):
            continue
        for feature in set(row["features"]):
            entry = counts.setdefault(feature, {"genomes": 0, "fine": 0})
            entry["genomes"] += 1
            entry["fine"] += row["verdict"] == "fine"
    return {feature: {**c, "fine_share": c["fine"] / c["genomes"]}
            for feature, c in sorted(counts.items(), key=lambda item: -item[1]["genomes"])}


def paired(rows, baseline_results, metric):
    """Per voice and line: mean of the fine genomes minus the baseline's clip. [(voice, difference)]"""
    plain = {(r["voice"], r["line"]): r[metric] for r in baseline_results.values()}
    groups = {}
    for row in rows:
        if row["verdict"] == "fine" and row.get(metric) is not None:
            groups.setdefault((row["voice"], row["line"]), []).append(row[metric])
    return [(voice, float(np.mean(values)) - plain[(voice, line)])
            for (voice, line), values in sorted(groups.items()) if plain.get((voice, line)) is not None]


def difference_of(pairs, rng):
    """Mean difference, a 95 % bootstrap interval over voices, and a Wilcoxon signed-rank test."""
    if not pairs:
        return None
    from scipy.stats import wilcoxon

    by_voice = {}
    for voice, value in pairs:
        by_voice.setdefault(voice, []).append(value)
    voices = list(by_voice.values())
    means = [np.mean([v for clips in (voices[i] for i in rng.integers(0, len(voices), len(voices))) for v in clips])
             for _ in range(base.BOOTSTRAP)]
    values = [value for _, value in pairs]
    try:
        p_value = float(wilcoxon(values).pvalue) if any(values) else 1.0
    except ValueError:
        p_value = None
    return {"mean": float(np.mean(values)), "low": float(np.percentile(means, 2.5)),
            "high": float(np.percentile(means, 97.5)), "pairs": len(values), "voices": len(voices),
            "wilcoxon_p": p_value}


def verdicts(shares, fine_means, baseline_summary):
    """The rules written in MILESTONE_2.md before the run."""
    inside = {}
    for metric in ("wer", "similarity"):
        interval = (baseline_summary.get("all") or {}).get(metric)
        value = fine_means.get(metric)
        if interval and value is not None:
            # WER may be lower than the interval and similarity higher: only the bad side counts
            inside[metric] = value <= interval["high"] if metric == "wer" else value >= interval["low"]
    works = bool(shares["fine"] is not None and shares["fine"] >= 0.5 and inside and all(inside.values()))
    return {
        "works_as_a_guard": works,
        "fine_share_at_least_half": shares["fine"] is not None and shares["fine"] >= 0.5,
        "fine_genomes_within_baseline_interval": inside,
        "beats_the_baseline": "not decided: needs the blind listening pairs (listening/)",
    }


def compare(out_dir, plan, baseline_dir, runs, jobs, measured):
    baseline_results = _load(baseline_dir / "results.json", {})
    baseline_summary = _load(baseline_dir / "summary.json", {})
    rows = rows_of(out_dir, plan, runs, jobs, measured)
    rng = np.random.default_rng(0)
    fine = [r for r in rows if r["verdict"] == "fine"]

    groups = {"all": rows, "short lines (0, 1)": [r for r in rows if r["line"] in SHORT_LINES],
              "longer lines": [r for r in rows if r["line"] not in SHORT_LINES]}
    for dataset in sorted({r["dataset"] for r in rows}):
        groups[dataset] = [r for r in rows if r["dataset"] == dataset]

    fine_means = {m: float(np.mean([r[m] for r in fine if r.get(m) is not None]))
                  for m in METRICS if any(r.get(m) is not None for r in fine)}
    summary = {
        "shares": {name: judged_shares(group) for name, group in groups.items()},
        "by_line": {str(line): judged_shares([r for r in rows if r["line"] == line])
                    for line in sorted({r["line"] for r in rows})},
        "by_feature": by_feature(rows),
        "fine_means": fine_means,
        "baseline_means": {m: (baseline_summary.get("all") or {}).get(m) for m in METRICS},
        "fine_minus_baseline": {m: difference_of(paired(rows, baseline_results, m), rng) for m in METRICS},
        "pairs_with_a_fine_genome": len({(r["voice"], r["line"]) for r in fine}),
        "pairs": len(jobs),
    }
    summary["verdicts"] = verdicts(summary["shares"]["all"], fine_means, baseline_summary)
    return summary, rows


# --------------------------------------------------------------------------- #
# Blind listening pairs
# --------------------------------------------------------------------------- #

def best_fine(rows_of_pair):
    """The best fine genome of a voice and line, by a rule fixed before listening:
    fewest word errors against the line, then most like the reference."""
    fine = [r for r in rows_of_pair if r["verdict"] == "fine" and r.get("wer") is not None]
    return min(fine, key=lambda r: (r["wer"], -r["similarity"])) if fine else None


def listening_pairs(out_dir, plan, baseline_dir, rows, count=LISTENING_PAIRS, seed=LISTENING_SEED):
    """
    `count` voice-and-line pairs, drawn at random among those with a fine genome
    (made once: a second call keeps the first draw). Each pair folder has A.wav and
    B.wav, the plain render and the best fine genome in random order; key.json says
    which is which and is not opened until the ratings are in.
    """
    folder = out_dir / "listening"
    key_path = folder / "key.json"
    if key_path.is_file():
        return _load(key_path, {})

    candidates = sorted({(r["voice"], r["line"]) for r in rows if r["verdict"] == "fine"})
    rng = random.Random(seed)
    chosen = rng.sample(candidates, min(count, len(candidates)))
    voices = {v["name"]: v for v in plan["voices"]}
    key = {"seed": seed, "rule": best_fine.__doc__.split(":", 1)[1].strip(), "pairs": []}
    for number, (voice, line) in enumerate(chosen, start=1):
        genome = best_fine([r for r in rows if r["voice"] == voice and r["line"] == line])
        plain = base.clip_path(baseline_dir, voices[voice], line)
        order = ["baseline", "genome"]
        rng.shuffle(order)
        pair_dir = folder / f"pair_{number:02d}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        files = {"baseline": plain, "genome": Path(genome["audio"])}
        for letter, which in zip("AB", order):
            shutil.copyfile(files[which], pair_dir / f"{letter}.wav")
        (pair_dir / "line.txt").write_text(plan["lines"][line], encoding="utf-8")
        key["pairs"].append({"pair": number, "voice": voice, "line": line, "A": order[0], "B": order[1],
                             "genome": genome["genome"], "markup": genome["markup"]})
    _save(key_path, key)
    (folder / "README.txt").write_text(
        "Blind pairs for LISTENING_PROTOCOL.md. In each pair_NN folder, listen to A.wav and B.wav (the line is in\n"
        "line.txt) and write down which one is more expressive while still natural, before opening key.json.\n",
        encoding="utf-8")
    return key


# --------------------------------------------------------------------------- #
# Showing, recording, logging
# --------------------------------------------------------------------------- #

def show(summary, key):
    print("\nShare of genomes judged fine (and why the others failed)")
    for name, s in summary["shares"].items():
        if s["fine"] is None:
            continue
        why = s["terrible_because"]
        reasons = ", ".join(f"{k} {v:.0%}" for k, v in why.items() if v is not None)
        print(f"  {name:20s} {s['fine']:5.0%} fine of {s['judged']:4d}" + (f"   failed for: {reasons}" if reasons else "")
              + (f"   ({s['failed_to_render']} did not render)" if s["failed_to_render"] else ""))
    print("\nBy line:")
    for line, s in summary["by_line"].items():
        print(f"  line {line}: {s['fine']:.0%} fine" if s["fine"] is not None else f"  line {line}: -")
    print("\nBy gene type (share of genomes with that gene that were fine):")
    for feature, c in summary["by_feature"].items():
        print(f"  {feature:22s} {c['fine_share']:5.0%} of {c['genomes']}")
    print("\nFine genomes minus the baseline (per voice and line; 95 % interval over voices; Wilcoxon p)")
    for metric, d in summary["fine_minus_baseline"].items():
        if d:
            p = f"{d['wilcoxon_p']:.3g}" if d["wilcoxon_p"] is not None else "-"
            print(f"  {metric:10s} {d['mean']:+.3f} ({d['low']:+.3f} to {d['high']:+.3f}), p = {p}, {d['pairs']} pairs")
    v = summary["verdicts"]
    print(f"\nWorks as a guard: {'yes' if v['works_as_a_guard'] else 'no'} "
          f"(fine share at least half: {v['fine_share_at_least_half']}; "
          f"fine genomes within the baseline's interval: {v['fine_genomes_within_baseline_interval']})")
    print(f"Beats the baseline: {v['beats_the_baseline']}")
    if key:
        print(f"Listening pairs: {len(key['pairs'])} in output/milestone2/.../listening/ (key.json holds the answers)")


def record_of(out_dir, baseline_dir, genes, genome_seed, jobs):
    from backend.genome.search import SCHEMA_VERSION
    from backend.metrics.reward import CONFIG, measuring_models

    baseline_record = _load(baseline_dir / "record.json", {})
    return {
        "name": out_dir.name,
        "made": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": base._git("rev-parse", "HEAD"),
        "git_dirty": bool(base._git("status", "--porcelain")),
        "baseline": baseline_dir.name,
        "baseline_git_commit": baseline_record.get("git_commit"),
        "baseline_plan_sha256": base.sha256(baseline_dir / "plan.json"),
        "method": "random",
        "genes_per_genome": genes,
        "genome_seed": genome_seed,
        "genomes_per_search": 10,
        "searches": len(jobs),
        "run_schema_version": SCHEMA_VERSION,
        "sanity_limits": {"words": CONFIG.sanity_wer_words, "pace": CONFIG.sanity_pace_min,
                          "similarity": CONFIG.sanity_similarity, "similarity_short": CONFIG.sanity_similarity_short,
                          "long_seconds": CONFIG.sanity_long_seconds},
        "judges": measuring_models(),
        "voice_model_weights": baseline_record.get("voice_model_weights"),
        "voice_model_code_commit": baseline_record.get("voice_model_code_commit"),
        "device": baseline_record.get("device"),
        "limitation": LIMITATION,
    }


def log_to_mlflow(out_dir, record, summary):
    from track import log_run

    metrics = {}
    for name, s in summary["shares"].items():
        if s["fine"] is not None:
            label = name.split(" (")[0].replace(" ", "_").replace("-", "_")
            metrics[f"{label}_fine_share"] = s["fine"]
    for reason, value in summary["shares"]["all"]["terrible_because"].items():
        if value is not None:
            metrics[f"terrible_because_{reason}"] = value
    for metric, value in summary["fine_means"].items():
        metrics[f"fine_{metric}"] = value
    for metric, d in summary["fine_minus_baseline"].items():
        if d:
            metrics[f"diff_{metric}"] = d["mean"]
            if d["wilcoxon_p"] is not None:
                metrics[f"diff_{metric}_p"] = d["wilcoxon_p"]
    metrics["works_as_a_guard"] = float(summary["verdicts"]["works_as_a_guard"])
    metrics["pairs_with_a_fine_genome"] = summary["pairs_with_a_fine_genome"]
    params = {key: record[key] for key in ("git_commit", "git_dirty", "baseline", "baseline_plan_sha256", "method",
                                           "genes_per_genome", "genome_seed", "genomes_per_search", "searches",
                                           "voice_model_weights", "device")}
    params.update({f"limit_{k}": v for k, v in record["sanity_limits"].items()})
    params.update({f"judge_{k}": v for k, v in record["judges"].items()})
    artifacts = [out_dir / name for name in ("summary.json", "record.json", "runs.json", "measured.json")]
    if (out_dir / "listening" / "key.json").is_file():
        artifacts.append(out_dir / "listening" / "key.json")
    log_run(f"search-{out_dir.name}", params, metrics, artifacts=artifacts, limitations=LIMITATION)


def main(name=NAME, baseline=BASELINE, run=False, voices=None, genes=GENES, genome_seed=GENOME_SEED, log=True):
    baseline_dir = base.OUT_DIR / baseline
    plan = _load(baseline_dir / "plan.json", None)
    if plan is None or not (baseline_dir / "results.json").is_file():
        raise SystemExit(f"Baseline {baseline} is not finished: run python main_backend.py baseline --name {baseline}")
    out_dir = OUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = jobs_of(plan, voices)
    print(f"Random search against baseline {baseline}: {len(jobs)} voice-and-line searches, {genes} genes, "
          f"genome seed {genome_seed}. Saved in {out_dir}", flush=True)

    if run:
        runs = prepare(out_dir, plan, baseline_dir, jobs, genes, genome_seed)
        search_all(out_dir, runs, jobs)
    else:
        runs = _load(out_dir / "runs.json", {})
        jobs = [job for job in jobs if key_of(*job) in runs]
        if not jobs:
            raise SystemExit("No searches yet: run with --run.")
    measured = measure_genomes(out_dir, plan, runs, jobs)
    summary, rows = compare(out_dir, plan, baseline_dir, runs, jobs, measured)
    key = listening_pairs(out_dir, plan, baseline_dir, rows) if voices is None else None
    record = record_of(out_dir, baseline_dir, genes, genome_seed, jobs)
    _save(out_dir / "summary.json", summary)
    _save(out_dir / "rows.json", rows)
    _save(out_dir / "record.json", record)
    show(summary, key)
    if log:
        try:
            log_to_mlflow(out_dir, record, summary)
            print("\nLogged in MLflow (history/eval/mlflow.db).")
        except ModuleNotFoundError as error:
            print(f"\nNot logged in MLflow ({error}).")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 2: random genome search against the baseline.")
    parser.add_argument("--name", default=NAME, help="the comparison's folder in output/milestone2")
    parser.add_argument("--baseline", default=BASELINE, help="the baseline run in output/baseline")
    parser.add_argument("--run", action="store_true", help="render and judge the searches (otherwise compare only)")
    parser.add_argument("--voices", type=int, help="only the first N voices of the baseline (a trial; no listening pairs)")
    parser.add_argument("--genes", type=int, default=GENES)
    parser.add_argument("--genome-seed", type=int, default=GENOME_SEED)
    parser.add_argument("--no-log", action="store_true", help="do not record the comparison in MLflow")
    args = parser.parse_args()
    main(args.name, args.baseline, args.run, args.voices, args.genes, args.genome_seed, not args.no_log)

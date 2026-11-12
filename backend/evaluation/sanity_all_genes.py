"""sanity_all_genes.py | The sanity check of every gene type on its own, on the baseline's voices and lines.

    python main_backend.py sanity-all-genes --run              # render and judge everything not done yet
    python main_backend.py sanity-all-genes --run --voices 1   # a small trial on the first voice
    python main_backend.py sanity-all-genes                    # only write rows.json from what is finished

Why: Milestone 2's random-v2 search drew its 10 genomes once (genome seed 0) and used them on all
72 voices and lines, so only 14 of the 17 gene types were ever checked (never hissing or either
pause), and each gene type was checked inside 1 to 5 fixed three-gene combinations, so a failure
could not be pinned on one gene type.

Here every genome has exactly one gene, and every voice and line gets one genome of each gene type:

    14 sound and word genes (emphasis, breath, ..., laughing speech), at a random word
    pause (dramatic) and pause (realization), at a random word
    speed, at a random rate from 0.7 to 1.3

The word and the rate are drawn per voice and line from a fixed seed, so a rerun draws the same
genomes. Each is rendered with the baseline's render seed and judged by the same sanity check as
random-v2 (backend/metrics/reward.sanity_check: words dropped or changed, voice drift, broken pace)
against the baseline's own plain render, copied in.

12 voices x 6 lines x 17 gene types = 1224 renders, about 10 s each on the RTX 3080 (3.5 h). Every
clip is saved as it is judged, so a stopped run continues where it stopped.

Everything goes in OUTPUT_DIR/milestone2/sanity-all-genes/: runs.json, runs/ (one search folder
per voice and line), rows.json (one row per judged genome) and record.json (when, what, how).
The evidence corpus (backend/rag/corpus.py) reads rows.json as sanity records.
"""
import argparse
import json
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import baseline as base  # noqa: E402

NAME = "sanity-all-genes"
OUT_DIR = base.model.OUTPUT_DIR / "milestone2" / NAME
BASELINE_DIR = base.model.PROJECT_DIR / "output" / "baseline" / "v2"
GENE_SEED = 0
LIMITATION = ("One gene per genome, one random word or rate per voice and line, one render seed; the sanity check "
              "measures faults (words, voice, pace), not expressiveness; validation voices only.")

_save, _load = base._save, base._load


def gene_types():
    """The 17 gene types: (name in words, how to make one gene of it from a random generator)."""
    from backend.genome.genome_factory import PAUSE_KINDS, SPEED_RANGE, TARGETED_FEATURES

    out = [(f.replace("_", " "), lambda rng, f=f: {"feature": f, "position": round(rng.random(), 3)})
           for f in TARGETED_FEATURES]
    out += [(f"pause ({kind})", lambda rng, kind=kind: {"feature": "pause", "position": round(rng.random(), 3), "kind": kind})
            for kind in PAUSE_KINDS]
    out.append(("speed", lambda rng: {"feature": "speed", "rate": round(rng.uniform(*SPEED_RANGE), 2)}))
    return out


def key_of(voice, line):
    return f"{voice['name']}/line{line}"


def genomes_for(text, job_number, seed=GENE_SEED):
    """One single-gene genome per gene type, drawn from this voice and line's own generator."""
    from backend.genome.candidate import Candidate
    from backend.speech.tts_formatter import process_tts_string

    rng = random.Random(seed * 100_000 + job_number)
    out = []
    for number, (name, make) in enumerate(gene_types(), start=1):
        genome = Candidate([make(rng)])
        plan = process_tts_string(text, genome.resolve(text))
        out.append({"number": number, "gene_type": name, "genes": genome.operations, "described": genome.describe(text),
                    "operations": genome.resolve(text), "markup": plan.text, "speed": plan.speed,
                    "warnings": plan.warnings, "audio": None, "sanity": None, "verdict": None})
    return out


def prepare(out_dir, plan, jobs):
    """One search folder per voice and line, made once; the baseline's render is the plain clip."""
    from backend.genome.search import load_run, new_run, save_run

    path = out_dir / "runs.json"
    runs = _load(path, {})
    for number, (voice, line) in enumerate(jobs):
        key = key_of(voice, line)
        if key in runs:
            continue
        plain = base.clip_path(BASELINE_DIR, voice, line)
        if not plain.is_file():
            raise SystemExit(f"Baseline v2 has no render for {key} ({plain}).")
        text = plan["lines"][line]
        run_dir = new_run(voice["reference_audio"], voice["reference_text"], text=text,
                          voice=f"{voice['name']}_line{line}", genes=1, seed=GENE_SEED,
                          render_seed=plan["render_seed"], origin=voice.get("origin"), runs_dir=out_dir / "runs")
        shutil.copyfile(plain, run_dir / "baseline.wav")
        run = load_run(run_dir)
        run["method_label"] = "Each gene type alone"
        run["genomes"] = genomes_for(text, number)
        run["baseline"].update(audio="baseline.wav", copied_from=plain.relative_to(BASELINE_DIR).as_posix(),
                               baseline=BASELINE_DIR.name)
        save_run(run_dir, run)
        runs[key] = run_dir.relative_to(out_dir).as_posix()
        _save(path, runs)
    return runs


def run_all(out_dir, runs, jobs, max_minutes=None):
    from backend.genome.search import load_run, run_search, summary

    deadline = time.time() + 60 * max_minutes if max_minutes else None
    todo = [key_of(v, i) for v, i in jobs if load_run(out_dir / runs[key_of(v, i)])["status"] != "finished"]
    print(f"{len(jobs) - len(todo)} of {len(jobs)} voices and lines already finished; running {len(todo)}.", flush=True)
    started_all = time.time()
    for number, key in enumerate(todo, start=1):
        if deadline and time.time() > deadline:
            print(f"Time limit reached; {len(todo) - number + 1} left. Run again to continue.", flush=True)
            return False
        started = time.time()
        counts = summary(run_search(out_dir / runs[key]))
        left = (time.time() - started_all) / number * (len(todo) - number) / 60
        print(f"  {number}/{len(todo)} {key}: {counts['fine']} fine, {counts['terrible']} failed the check"
              + (f", {counts['failed']} did not render" if counts["failed"] else "")
              + f" ({time.time() - started:.0f} s, about {left:.0f} min left)", flush=True)
    return True


def rows_of(out_dir, runs, jobs):
    from backend.genome.search import load_run

    rows = []
    for voice, line in jobs:
        key = key_of(voice, line)
        if key not in runs:
            continue
        run_dir = out_dir / runs[key]
        for genome in load_run(run_dir)["genomes"]:
            if genome["verdict"] is None:
                continue
            s = genome.get("sanity") or {}
            rows.append({"voice": voice["name"], "dataset": voice["dataset"], "line": line, "genome": genome["number"],
                         "gene_type": genome["gene_type"], "features": [g["feature"] for g in genome["genes"]],
                         "genes": genome["genes"], "markup": genome["markup"], "speed": genome["speed"],
                         "verdict": genome["verdict"], "error": genome.get("error"),
                         "wer_ok": s.get("wer_ok"), "similarity_ok": s.get("similarity_ok"), "pace_ok": s.get("pace_ok"),
                         "wer": s.get("wer"), "similarity": s.get("similarity"), "pace": s.get("pace"),
                         "heard": s.get("test_heard"), "seconds": genome.get("seconds"),
                         "audio": (run_dir / genome["audio"]).as_posix() if genome.get("audio") else None})
    return rows


def record_of(out_dir, plan, jobs):
    from backend.metrics.reward import CONFIG

    path = out_dir / "record.json"
    record = _load(path, {})
    if not record:
        import torch
        record = {"name": NAME, "made": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "method": "each gene type alone",
                  "baseline": BASELINE_DIR.name, "genes_per_genome": 1, "gene_seed": GENE_SEED,
                  "gene_types": [name for name, _ in gene_types()], "voices": len({v["name"] for v, _ in jobs}),
                  "lines": len(plan["lines"]), "render_seed": plan["render_seed"],
                  "sanity_limits": {"words": CONFIG.sanity_wer_words, "pace": CONFIG.sanity_pace_min,
                                    "similarity": CONFIG.sanity_similarity,
                                    "similarity_short": CONFIG.sanity_similarity_short,
                                    "long_seconds": CONFIG.sanity_long_seconds},
                  "judges": {"wer": "whisper-base", "similarity": "resemblyzer"},
                  "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                  "limitation": LIMITATION}
        _save(path, record)
    return record


def report(rows):
    judged = [r for r in rows if r["verdict"] in ("fine", "terrible")]
    if not judged:
        return
    by = {}
    for r in judged:
        by.setdefault(r["gene_type"], []).append(r["verdict"] == "terrible")
    failed = sum(r["verdict"] == "terrible" for r in judged)
    print(f"\n{len(judged)} judged, {failed} failed the sanity check ({100 * failed / len(judged):.0f}%):")
    for name, items in sorted(by.items(), key=lambda kv: -sum(kv[1]) / len(kv[1])):
        print(f"  {name:22s} {100 * sum(items) / len(items):5.1f}%  ({sum(items)} of {len(items)})")


def main(run=False, voices=None, max_minutes=None, out_dir=OUT_DIR):
    plan = json.loads((BASELINE_DIR / "plan.json").read_text(encoding="utf-8"))
    chosen = plan["voices"][:voices] if voices else plan["voices"]
    jobs = [(v, line) for v in chosen for line in range(len(plan["lines"]))]
    out_dir.mkdir(parents=True, exist_ok=True)
    record_of(out_dir, plan, jobs)
    runs = prepare(out_dir, plan, jobs)
    if run:
        run_all(out_dir, runs, jobs, max_minutes)
    rows = rows_of(out_dir, runs, jobs)
    _save(out_dir / "rows.json", rows)
    print(f"{len(rows)} rows -> {out_dir / 'rows.json'}")
    report(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The sanity check of every gene type on its own.")
    parser.add_argument("--run", action="store_true", help="render and judge (GPU); otherwise only write rows.json")
    parser.add_argument("--voices", type=int, help="only the first N voices of baseline v2 (a trial)")
    parser.add_argument("--max-minutes", type=float, help="stop after this long; run again to continue")
    args = parser.parse_args()
    main(args.run, args.voices, args.max_minutes)

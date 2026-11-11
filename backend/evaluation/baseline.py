"""baseline.py | The reproducible baseline: plain CosyVoice 3 renders of fixed lines in validation voices, measured and checked.

    python main_backend.py baseline                  # 12 voices x 6 lines, render seed 0
    python main_backend.py baseline --measure-only   # render nothing; measure and report what is there
    python main_backend.py baseline --no-log         # do not record the run in MLflow

    # The script baseline: every line of a script in every validation voice of the voice manifest
    python main_backend.py baseline --name script-v1 --script backend/data/scripts/context-script-v1.json --voices all

The baseline is what every later change is compared against (SCOPE.md): the voice
model CosyVoice 3 with its default settings, no genome, no cleaning of the
reference. Each line is spoken in each voice once, by the same renderer the genome
search uses for its own baseline clip, so a search result and this baseline are
made the same way.

    voices      an even share of VCTK, CREMA-D and LibriTTS-R, from the validation
                voices only (never the final test), each with its longest neutral clip;
                or, with --voices all, every validation voice in the voice manifest
    lines       LINES below, or the lines of a script file (--script): written for
                this, not taken from any dataset, no digits
    measured    WER (Whisper), speaker similarity to the reference (Resemblyzer),
                estimated quality (SQUIM: PESQ, STOI, SI-SDR), speaking rate
    reported    per dataset and overall, with a 95 % bootstrap interval over voices

Known-answer checks, where the right answer is known before looking:

    1. a reference clip compared with itself has similarity 1
    2. the recogniser hears the reference recordings' own scripts (their WER is its error)
    3. each render is most similar to its own voice, not to another voice in the run
    4. rendering the same voice, line and seed again gives the same audio

A script baseline (--script) renders every validation voice x every line. Its voice
folders are named after the reference clip (<dataset>_<clip id>), and
baseline_manifest.json says for every audio which line it speaks and which reference
clip it was cloned from (dataset, speaker, clip, transcript, length, checksum, split).

Every run goes in OUTPUT_DIR/baseline/<name>/ (OUTPUT_DIR is on the D drive, outside
the repository, see backend/model.py): plan.json (the
voices, lines and seeds, made once and reused), the renders, results.json (saved
after every clip, so a stopped run continues), checks.json, summary.json and
record.json (git commit, data hashes, models, versions, seeds). The run is then
logged in MLflow (history/eval/mlflow.db) with track.log_run.
"""
import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from backend import model  # noqa: E402

OUT_DIR = model.OUTPUT_DIR / "baseline"


NAME = "v2"             # v1 had only the first five lines (its plan.json keeps them)
VOICES = 12             # 4 per dataset
PICK_SEED = 0
RENDER_SEED = 0
# From short to long. Plain words and no digits, because the renderer runs without
# the text normaliser (as the search does).
LINES = [
    "I can't believe we actually pulled that off.",
    "Please close the door on your way out.",
    "The meeting has been moved to Thursday afternoon, so we still have time to prepare.",
    "If you had told me this morning how the day would end, I would never have believed you.",
    "We should have left the house an hour ago, but nobody could find the keys, and now the last train is gone.",
    # Added for v2: the shared comparison sentence. It reads as praise, sarcasm or a
    # threat depending on delivery (stress, pauses, laughter, pace), so it shows
    # whether a model or genome brings out an intended reading. 21 words, about 7 s.
    "Well, you really did it this time, and I honestly didn't think you had it in you, so I'll be waiting.",
]
BOOTSTRAP = 2000
LIMITATION = ("Plain CosyVoice 3 on validation voices only; automatic judges (Whisper base, Resemblyzer, SQUIM), "
              "no listening test; one render seed; small sample, so differences of a few hundredths are noise.")
# .env values that are safe to record (no secrets, no paths)
SETTINGS = ["DEVICE", "WER_MODEL", "SIMILARITY_MODEL", "PERQ_MODEL", "VOICE_SEED"]


def _save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _load(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# The plan, the renders, the measurements
# --------------------------------------------------------------------------- #

def read_script(path):
    """The lines of a script file (backend/data/scripts/), its name and its hash."""
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT / path
    script = json.loads(path.read_text(encoding="utf-8"))
    return [line["text"] for line in script["lines"]], {"name": script["name"], "file": path.relative_to(PROJECT).as_posix(),
                                                       "sha256": sha256(path),
                                                       "line_ids": [line["id"] for line in script["lines"]]}


def every_validation_voice():
    """The chosen clip of every validation voice in the voice manifest, in the manifest's order."""
    from backend.voice_data.split_voice_data import read_library, voice_clips

    chosen = voice_clips("validation")
    by_id = {(f"{clip.dataset}/{clip.speaker}", clip.clip_id): clip for clip in read_library()}
    return [by_id[(voice, clip_id)] for voice, clip_id in chosen.items()]


def plan_of(out_dir, voices, pick_seed, render_seed, script=None):
    """The voices, lines and seed of this baseline: made once, then reused, so a restart renders the same set."""
    path = out_dir / "plan.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))

    from backend.voice_data.Voice_Importer import get_clip
    from backend.voice_data.split_voice_data import pick_even, voice_origin

    lines, script_info = read_script(script) if script else (LINES, None)
    plan = {"lines": lines, "script": script_info, "render_seed": render_seed,
            "pick_seed": None if voices == "all" else pick_seed, "voices": []}
    clips = every_validation_voice() if voices == "all" else pick_even(voices, seed=pick_seed, split="validation")
    for clip in clips:
        origin = voice_origin(clip)
        if origin["split"] != "validation":
            raise RuntimeError(f"{clip.dataset}/{clip.speaker} is not a validation speaker")
        audio, text = get_clip(clip)
        # A script baseline names each voice after its reference clip, so a folder says which audio it was cloned from
        name = f"{clip.dataset}_{clip.clip_id}" if script else f"{clip.dataset}_{clip.speaker}"
        plan["voices"].append({"name": name, "dataset": clip.dataset,
                               "reference_audio": str(audio), "reference_text": text, "origin": origin})
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(path, plan)
    return plan


def write_baseline_manifest(out_dir, plan):
    """
    baseline_manifest.json: every audio of the run, the line it speaks and the reference
    clip it was cloned from, with the voice manifest's details of that clip. Rewritten
    after rendering, so the render checksums are those of the files on disk.
    """
    from backend.voice_data.split_voice_data import SPLITS_FILE, load_voice_manifest

    voices = {v["voice"]: v for v in load_voice_manifest()["voices"]}
    line_ids = (plan.get("script") or {}).get("line_ids") or [f"line{i}" for i in range(len(plan["lines"]))]
    audios = []
    for voice in plan["voices"]:
        origin = voice["origin"]
        known = voices.get(f"{origin['dataset']}/{origin['speaker']}", {})
        reference = {
            "voice": f"{origin['dataset']}/{origin['speaker']}",
            "dataset": origin["dataset"],
            "speaker": origin["speaker"],
            "clip_id": origin["clip_id"],
            "transcript": voice["reference_text"],
            "speech_seconds": known.get("speech_seconds"),
            "sha256": known.get("sha256"),
            "split": origin["split"],
        }
        for line, text in enumerate(plan["lines"]):
            path = clip_path(out_dir, voice, line)
            audios.append({
                "audio": path.relative_to(out_dir).as_posix(),
                "line": line,
                "line_id": line_ids[line],
                "text": text,
                "render_seed": plan["render_seed"],
                "rendered": path.is_file(),
                "sha256": sha256(path) if path.is_file() else None,
                "reference": reference,
            })
    _save(out_dir / "baseline_manifest.json", {
        "name": out_dir.name,
        "what": "Plain CosyVoice 3 renders (no genome): each audio, the line it speaks and the reference clip it was cloned from.",
        "script": plan.get("script"),
        "voice_manifest": SPLITS_FILE.name,
        "voice_manifest_sha256": sha256(SPLITS_FILE),
        "voices": len(plan["voices"]),
        "lines": len(plan["lines"]),
        "audios": audios,
    })


def clip_path(out_dir, voice, line):
    return out_dir / "renders" / voice["name"] / f"line{line}.wav"


def render(voice, text, path, seed):
    """One plain render, written to a .part file first so a stopped run never leaves half a clip."""
    from backend.genome.search import render_plan
    from backend.speech.tts_formatter import process_tts_string

    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    render_plan(voice["reference_audio"], voice["reference_text"], process_tts_string(text, []),
                path.with_suffix(".part.wav"), seed)
    os.replace(path.with_suffix(".part.wav"), path)
    return time.time() - started


def render_all(out_dir, plan):
    jobs = [(v, i) for v in plan["voices"] for i in range(len(plan["lines"]))]
    todo = [job for job in jobs if not clip_path(out_dir, *job).is_file()]
    print(f"{len(jobs) - len(todo)} of {len(jobs)} clips already made; making {len(todo)}.", flush=True)
    times = _load(out_dir / "render_seconds.json", {})
    for number, (voice, line) in enumerate(todo, start=1):
        path = clip_path(out_dir, voice, line)
        seconds = render(voice, plan["lines"][line], path, plan["render_seed"])
        times[path.relative_to(out_dir).as_posix()] = seconds
        _save(out_dir / "render_seconds.json", times)
        print(f"  {number}/{len(todo)} {voice['name']} line {line} ({seconds:.1f} s)", flush=True)


def measure_all(out_dir, plan):
    """The metrics of every render, kept in results.json after each one (and in measurements/, as every clip is)."""
    from backend.metrics.measurements import measure

    path = out_dir / "results.json"
    results = _load(path, {})
    times = _load(out_dir / "render_seconds.json", {})
    for voice in plan["voices"]:
        for line, text in enumerate(plan["lines"]):
            clip = clip_path(out_dir, voice, line)
            key = clip.relative_to(out_dir).as_posix()
            if key in results or not clip.is_file():
                continue
            metrics = measure(voice["reference_audio"], clip, text, reference_text=voice["reference_text"])["metrics"]
            results[key] = {
                "voice": voice["name"], "dataset": voice["dataset"], "line": line, "words": len(text.split()),
                "wer": metrics["wer"], "heard": metrics["heard"], "similarity": metrics["similarity"],
                "pesq": metrics["quality"]["pesq"], "stoi": metrics["quality"]["stoi"],
                "si_sdr": metrics["quality"]["si_sdr"], "words_per_second": metrics["words_per_second"],
                "rate_vs_reference": metrics["rate_vs_reference"], "seconds": metrics["seconds"],
                "render_seconds": times.get(key),
            }
            _save(path, results)
            print(f"  measured {key}: WER {metrics['wer']:.3f}, similarity {metrics['similarity']:.3f}", flush=True)
    return results


# --------------------------------------------------------------------------- #
# Known-answer checks
# --------------------------------------------------------------------------- #

def known_answer_checks(out_dir, plan, results, render_again=True):
    """The checks whose right answer is known in advance; saved in checks.json."""
    from backend.metrics.Similarity import speaker_similarity
    from backend.metrics.WER import transcribe, wer_of

    path = out_dir / "checks.json"
    saved = _load(path, {})
    checks = []

    # 1. A reference against itself: the similarity judge must give 1
    selfsame = [speaker_similarity(v["reference_audio"], v["reference_audio"]) for v in plan["voices"]]
    checks.append({"name": "a reference compared with itself has similarity 1",
                   "passed": min(selfsame) > 0.999, "value": min(selfsame)})

    # 2. The recogniser on the real recordings: their WER is the recogniser's own error
    reference_wer = saved.get("reference_wer") or {
        v["name"]: wer_of(v["reference_text"], transcribe(v["reference_audio"])) for v in plan["voices"]}
    saved["reference_wer"] = reference_wer
    _save(path, saved)
    median = float(np.median(list(reference_wer.values())))
    checks.append({"name": "the recogniser hears the reference recordings (median WER at most 0.10)",
                   "passed": median <= 0.10, "value": median,
                   "detail": {"mean": float(np.mean(list(reference_wer.values()))), "per_voice": reference_wer}})

    # 3. Each render is closer to its own voice than to any other voice in the run
    if results:
        own = 0
        for key, row in results.items():
            scores = {v["name"]: speaker_similarity(out_dir / key, v["reference_audio"]) for v in plan["voices"]}
            own += max(scores, key=scores.get) == row["voice"]
        share = own / len(results)
        checks.append({"name": "each render is most like its own voice (at least 90 % of renders)",
                       "passed": share >= 0.90, "value": share})

    # 4. Reproducibility: the first clip rendered again with the same seed (reused once it is made)
    if plan["voices"]:
        voice, line = plan["voices"][0], 0
        first = clip_path(out_dir, voice, line)
        again = out_dir / "reproduce" / first.name
        if first.is_file() and (again.is_file() or render_again):
            if not again.is_file():
                render(voice, plan["lines"][line], again, plan["render_seed"])
            from backend.audio_io import load_mono
            a, b = load_mono(first, 16000), load_mono(again, 16000)
            same_length = len(a) == len(b)
            difference = float(np.max(np.abs(a - b))) if same_length else None
            checks.append({"name": "the same voice, line and seed render the same audio",
                           "passed": same_length and difference <= 1e-3,
                           "value": difference, "detail": {"identical_file": sha256(first) == sha256(again),
                                                           "samples": [len(a), len(b)]}})

    _save(path, {**saved, "checks": checks})
    return checks


# --------------------------------------------------------------------------- #
# The summary and the run record
# --------------------------------------------------------------------------- #

METRICS = ["wer", "similarity", "pesq", "stoi", "si_sdr", "words_per_second", "rate_vs_reference", "render_seconds"]


def _interval(rows, metric, rng):
    """Mean, and a 95 % bootstrap interval that resamples voices (clips of one voice are not independent)."""
    by_voice = {}
    for row in rows:
        if row.get(metric) is not None:
            by_voice.setdefault(row["voice"], []).append(row[metric])
    if not by_voice:
        return None
    voices = list(by_voice.values())
    means = []
    for _ in range(BOOTSTRAP):
        sample = [voices[i] for i in rng.integers(0, len(voices), len(voices))]
        means.append(np.mean([value for clips in sample for value in clips]))
    values = [value for clips in voices for value in clips]
    return {"mean": float(np.mean(values)), "low": float(np.percentile(means, 2.5)),
            "high": float(np.percentile(means, 97.5)), "n": len(values), "voices": len(voices)}


def summarise(results):
    rows = list(results.values())
    groups = {"all": rows}
    for dataset in sorted({r["dataset"] for r in rows}):
        groups[dataset] = [r for r in rows if r["dataset"] == dataset]
    rng = np.random.default_rng(0)
    summary = {}
    for name, group in groups.items():
        summary[name] = {metric: _interval(group, metric, rng) for metric in METRICS}
        summary[name]["wer_zero"] = sum(r["wer"] == 0 for r in group) / len(group)
    summary["by_line"] = {str(line): {"wer": float(np.mean([r["wer"] for r in rows if r["line"] == line])),
                                      "similarity": float(np.mean([r["similarity"] for r in rows if r["line"] == line]))}
                          for line in sorted({r["line"] for r in rows})}
    return summary


def _git(*args, cwd=PROJECT):
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return None


def _optional(name):
    """A setting from .env, or None when it is not set (the run record must not fail on a missing one)."""
    value = os.environ.get(name, "").strip()
    return value or None


def run_record(plan, out_dir):
    """Everything needed to repeat the run, without secrets or private paths."""
    import torch

    from checks import MANIFEST, SPLIT
    from backend.metrics.reward import measuring_models

    versions = {}
    for package in ["torch", "torchaudio", "transformers", "openai-whisper", "resemblyzer", "jiwer", "numpy", "mlflow"]:
        try:
            from importlib.metadata import version
            versions[package] = version(package)
        except Exception:
            versions[package] = None

    cosyvoice_code = _optional("VOICE_MODEL_DIR")
    weights = _optional("COSYVOICE_MODEL_DIR")
    return {
        "name": out_dir.name,
        "made": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "manifest_sha256": sha256(MANIFEST) if MANIFEST.is_file() else None,
        "split_sha256": sha256(SPLIT),
        "plan_sha256": sha256(out_dir / "plan.json"),
        "voice_model": "CosyVoice 3",
        "voice_model_weights": Path(weights).name if weights else None,
        "voice_model_code_commit": _git("rev-parse", "HEAD", cwd=cosyvoice_code) if cosyvoice_code else None,
        "judges": measuring_models(),
        "render_seed": plan["render_seed"],
        "pick_seed": plan["pick_seed"],
        "voices": [v["name"] for v in plan["voices"]],
        "lines": plan["lines"],
        "script": plan.get("script"),
        "settings": {key: _optional(key) for key in SETTINGS},
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
        "limitation": LIMITATION,
    }


def show(summary, checks, lines=LINES):
    print("\nBaseline (mean, and 95 % interval over voices)")
    print(f"  {'':11s} {'WER':>22s} {'similarity':>22s} {'PESQ':>18s} {'WER 0':>6s}")
    for name, s in summary.items():
        if name == "by_line":
            continue
        cell = lambda m, f: f"{s[m]['mean']:{f}} ({s[m]['low']:{f}}-{s[m]['high']:{f}})" if s[m] else "-"
        print(f"  {name:11s} {cell('wer', '.3f'):>22s} {cell('similarity', '.3f'):>22s} {cell('pesq', '.2f'):>18s} "
              f"{s['wer_zero']:6.0%}")
    print("\nBy line (all voices):")
    for line, numbers in summary["by_line"].items():
        print(f"  line {line} ({len(lines[int(line)].split())} words): WER {numbers['wer']:.3f}, "
              f"similarity {numbers['similarity']:.3f}")
    print("\nKnown-answer checks:")
    for check in checks:
        value = f"{check['value']:.4g}" if isinstance(check.get("value"), float) else check.get("value")
        print(f"  {'ok  ' if check['passed'] else 'FAIL'}  {check['name']}  ({value})")


def log_to_mlflow(out_dir, record, summary, checks):
    from track import log_run

    metrics = {}
    for group in ("all", "VCTK", "CREMA-D", "LibriTTS-R"):
        for metric in ("wer", "similarity", "pesq", "stoi", "si_sdr", "words_per_second", "render_seconds"):
            value = (summary.get(group) or {}).get(metric)
            if value:
                metrics[f"{group}_{metric}".replace("-", "_")] = value["mean"]
        if group in summary:
            metrics[f"{group}_wer_zero".replace("-", "_")] = summary[group]["wer_zero"]
    metrics["checks_passed"] = sum(c["passed"] for c in checks)
    metrics["checks_total"] = len(checks)
    params = {key: record[key] for key in ("git_commit", "git_dirty", "manifest_sha256", "split_sha256", "plan_sha256",
                                           "voice_model", "voice_model_weights", "voice_model_code_commit",
                                           "render_seed", "pick_seed", "device", "cuda")}
    params.update({f"judge_{k}": v for k, v in record["judges"].items()})
    params["voices"] = len(record["voices"])
    params["lines"] = len(record["lines"])
    if record.get("script"):
        params["script"] = record["script"]["name"]
        params["script_sha256"] = record["script"]["sha256"]
    log_run(f"baseline-{out_dir.name}", params, metrics,
            artifacts=[out_dir / name for name in ("plan.json", "results.json", "checks.json", "summary.json", "record.json",
                                                    "baseline_manifest.json") if (out_dir / name).is_file()],
            limitations=LIMITATION)


def main(name=NAME, voices=VOICES, pick_seed=PICK_SEED, render_seed=RENDER_SEED, measure_only=False, log=True,
         script=None):
    out_dir = OUT_DIR / name
    plan = plan_of(out_dir, voices, pick_seed, render_seed, script)
    write_baseline_manifest(out_dir, plan)
    print(f"Baseline {name}: {len(plan['voices'])} validation voices x {len(plan['lines'])} lines, "
          f"render seed {plan['render_seed']}. Saved in {out_dir}", flush=True)
    if not measure_only:
        render_all(out_dir, plan)
        write_baseline_manifest(out_dir, plan)
    results = measure_all(out_dir, plan)
    if not results:
        raise SystemExit("Nothing measured yet: render first.")
    checks = known_answer_checks(out_dir, plan, results, render_again=not measure_only)
    summary = summarise(results)
    record = run_record(plan, out_dir)
    _save(out_dir / "summary.json", summary)
    _save(out_dir / "record.json", record)
    show(summary, checks, plan["lines"])
    if log:
        try:
            log_to_mlflow(out_dir, record, summary, checks)
            print("\nLogged in MLflow (history/eval/mlflow.db).")
        except ModuleNotFoundError as error:
            # The run is complete in out_dir either way; MLflow only collects it with the others
            print(f"\nNot logged in MLflow ({error}). Install it (requirements.txt), then run with "
                  "--measure-only to log this run without rendering again.")
    return summary, checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The reproducible baseline: plain renders, measured and checked.")
    parser.add_argument("--name", default=NAME, help="the run's folder, in OUTPUT_DIR/baseline")
    parser.add_argument("--voices", default=str(VOICES),
                        help="validation voices, an even share per dataset; or 'all' for every validation voice")
    parser.add_argument("--script", help="a script file (backend/data/scripts/) whose lines are rendered instead of LINES")
    parser.add_argument("--pick-seed", type=int, default=PICK_SEED, help="which voices are picked")
    parser.add_argument("--render-seed", type=int, default=RENDER_SEED, help="the seed of every render")
    parser.add_argument("--measure-only", action="store_true", help="render nothing; measure what is there")
    parser.add_argument("--no-log", action="store_true", help="do not record the run in MLflow")
    args = parser.parse_args()
    voices = "all" if args.voices == "all" else int(args.voices)
    main(args.name, voices, args.pick_seed, args.render_seed, args.measure_only, not args.no_log, args.script)

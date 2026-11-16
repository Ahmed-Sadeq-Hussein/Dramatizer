"""artifact_check.py | An automatic check of a rendered clip for faults a listener would hear, before its
score is used to recommend it.

    python main_backend.py artifact-check                 # every rated run; prints what failed
    python main_backend.py artifact-check --run RUN_ID    # one run

The sanity check (search.py) asks whether the words, the voice and the pace survived. It cannot hear a
click, a burst or a gap. This check reads the audio itself, frame by frame (20 ms), and fails a clip on:

    unreadable     the file is missing, empty, or holds NaN / infinite samples
    clipping       more than CLIP_SHARE of the samples at full scale (|x| >= CLIP_LEVEL)
    burst          its loudest 20 ms frame is more than BURST_DB louder than the loudest frame of the
                   plain line of the same run (an inserted sound far louder than the voice itself)
    gap            silence (below SILENT_DB) longer than GAP_SECONDS between the first and last sound
    too_short      less than MIN_VOICED_SECONDS of sound in the whole clip

The limits are working choices; they are written into every result, so a change shows. BURST_DB was set
on the 24 rated voices of mos-diverse-v1 (16 Nov 2026): a genome's loudest frame is a median 0.7 dB and at
most 7.8 dB above its plain line's, so 6 dB (twice the amplitude) flags only the clear outliers. A loudness
measure against the clip's own median was tried first and dropped: the plain lines alone reach 18 dB there. Results are cached in
<run folder>/artifact_check.json, keyed by each file's size and time, and rebuilt when a file changes.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from backend import safe_files

VERSION = 2
CACHE_FILE = "artifact_check.json"
FRAME_SECONDS = 0.02
CLIP_LEVEL = 0.99
CLIP_SHARE = 0.001          # 0.1 % of the samples
BURST_DB = 6.0
SILENT_DB = -50.0
GAP_SECONDS = 2.5           # the longest pause gene asks for about 1 s
MIN_VOICED_SECONDS = 0.5
LIMITS = {"clip_level": CLIP_LEVEL, "clip_share": CLIP_SHARE, "burst_db": BURST_DB, "silent_db": SILENT_DB,
          "gap_seconds": GAP_SECONDS, "min_voiced_seconds": MIN_VOICED_SECONDS, "frame_seconds": FRAME_SECONDS}


def _db(x):
    return 20 * np.log10(np.maximum(x, 1e-10))


def check_samples(samples, rate, plain_peak_db=None):
    """{"ok", "failed": [reasons], measured values} for one clip's samples (mono or channels-last).

    plain_peak_db: the loudest frame of the run's plain line ("peak_db" of its result); None skips the burst test.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.size == 0 or not np.all(np.isfinite(x)):
        return {"ok": False, "failed": ["unreadable"]}
    frame = max(1, int(rate * FRAME_SECONDS))
    n = x.size // frame
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1)) if n else np.array([0.0])
    level = _db(rms)
    voiced = level > SILENT_DB
    out = {"seconds": round(x.size / rate, 2), "clipped_share": round(float(np.mean(np.abs(x) >= CLIP_LEVEL)), 5),
           "voiced_seconds": round(float(voiced.sum() * FRAME_SECONDS), 2)}
    failed = []
    if out["clipped_share"] > CLIP_SHARE:
        failed.append("clipping")
    if out["voiced_seconds"] < MIN_VOICED_SECONDS:
        failed.append("too_short")
    out["peak_db"] = round(float(level.max()), 1)
    if plain_peak_db is not None:
        out["louder_than_plain_db"] = round(out["peak_db"] - plain_peak_db, 1)
        if out["louder_than_plain_db"] > BURST_DB:
            failed.append("burst")
    if voiced.any():
        first, last = np.flatnonzero(voiced)[[0, -1]]
        longest = run = 0
        for v in voiced[first:last + 1]:
            run = 0 if v else run + 1
            longest = max(longest, run)
        out["longest_gap_seconds"] = round(longest * FRAME_SECONDS, 2)
        if out["longest_gap_seconds"] > GAP_SECONDS:
            failed.append("gap")
    return {"ok": not failed, "failed": failed, **out}


def check_file(path, plain_peak_db=None):
    import soundfile as sf
    try:
        samples, rate = sf.read(str(path), dtype="float64", always_2d=False)
    except (OSError, RuntimeError, ValueError):
        return {"ok": False, "failed": ["unreadable"]}
    return check_samples(samples, rate, plain_peak_db)


def check_run(folder, run):
    """{clip key ("baseline", "genome_01", ...): result} for every clip of a run, cached in the run folder."""
    folder = Path(folder)
    cache_path = folder / CACHE_FILE
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if cache.get("version") != VERSION or cache.get("limits") != LIMITS:
            cache = None
    except (OSError, ValueError):
        cache = None
    cache = cache or {"version": VERSION, "limits": LIMITS, "clips": {}}
    clips = {"baseline": "baseline.wav", **{f"genome_{g['number']:02d}": g.get("audio") or f"genome_{g['number']:02d}.wav"
                                            for g in run.get("genomes", [])}}
    changed = False
    results = {}
    plain_peak = None
    for key, name in clips.items():                 # the plain line first: the others are measured against it
        path = folder / name
        stamp = [path.stat().st_size, path.stat().st_mtime_ns] if path.is_file() else None
        old = cache["clips"].get(key)
        if old and old.get("stamp") == stamp and old.get("plain_peak_db") == plain_peak:
            results[key] = old["result"]
        else:
            results[key] = check_file(path, plain_peak) if stamp else {"ok": False, "failed": ["unreadable"]}
            cache["clips"][key] = {"stamp": stamp, "plain_peak_db": plain_peak, "result": results[key]}
            changed = True
        if key == "baseline":
            plain_peak = results[key].get("peak_db")
    if changed:
        try:
            safe_files.write_json(cache_path, cache)
        except OSError:
            pass                                    # a read-only folder: the result is still returned
    return results


def main(argv=None):
    from backend.genome import search as S

    parser = argparse.ArgumentParser(description="Check rendered clips for clipping, bursts, gaps and empty audio.")
    parser.add_argument("--run", help="one run id (default: every run with ratings)")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    total = failed = 0
    for folder, run in S.list_runs():
        if args.run and run["id"] != args.run:
            continue
        if not args.run and not (folder / "ratings.json").is_file():
            continue
        for key, result in check_run(folder, run).items():
            total += 1
            if not result["ok"]:
                failed += 1
                print(f"{run['id']}/{key}: {', '.join(result['failed'])}  "
                      f"{ {k: v for k, v in result.items() if k not in ('ok', 'failed')} }")
    print(f"{failed} of {total} clips failed the artifact check (limits: {LIMITS}).")


if __name__ == "__main__":
    main()

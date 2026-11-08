"""calibrate_sanity.py | How much do two plain renders of the same line differ by chance? That sets the sanity check's limits.

The sanity check judges a genome's clip against a baseline: the same line in the
same voice with no genes. Even with no genes, two renders with different seeds
are not identical: the recogniser may hear a word differently, the pace moves a
little, the voice a little. A limit tighter than that chance difference would
call a genome terrible for nothing it did.

So this renders each line with no genes under several seeds, for validation
voices only (never the final test), and compares every pair of renders of the
same voice and line, exactly as the sanity check compares a clip with its
baseline:

    words        how many words are heard differently (fillers such as "uh" left out)
    pace         one render's speaking time over the other's
    similarity   speaker similarity between the two

    python -m backend.metrics.calibrate_sanity                      # 10 voices x 2 lines x 5 seeds
    python -m backend.metrics.calibrate_sanity --voices 20 --seeds 6
    python -m backend.metrics.calibrate_sanity --measure-only       # only measure what is rendered

Renders go in output/calibration/sanity/ (never committed). Every clip is saved
as soon as it is made, and every transcript too, so a stopped run continues
where it was when started again. summary.json holds the results and the
suggested limits; they are put in RewardConfig by hand.
"""
import argparse
import itertools
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from backend import model

OUT_DIR = model.PROJECT_DIR / "output" / "calibration" / "sanity"
# A short line and a long one: a limit in words matters most on the short one.
# Written without digits, because the search renders without the text normaliser.
LINES = [
    "I can't believe we actually pulled that off.",
    "We should have left the house an hour ago, but nobody could find the keys, and now the last train is gone.",
]
VOICES = 10
SEEDS = 5
PICK_SEED = 0
CHANCE = 0.95           # a limit should let this share of plain pairs through


def _save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def plan_of(out_dir, voices, seeds, pick_seed):
    """The voices, lines and seeds of this calibration: made once, then reused, so a restart renders the same set."""
    path = out_dir / "plan.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))

    from backend.voice_data.Voice_Importer import get_clip
    from backend.voice_data.split_voice_data import pick_even, voice_origin

    plan = {"lines": LINES, "seeds": list(range(seeds)), "pick_seed": pick_seed, "voices": []}
    for clip in pick_even(voices, seed=pick_seed, split="validation"):
        audio, text = get_clip(clip)
        plan["voices"].append({"name": f"{clip.dataset}_{clip.speaker}", "reference_audio": str(audio),
                               "reference_text": text, "origin": voice_origin(clip)})
    out_dir.mkdir(parents=True, exist_ok=True)
    _save(path, plan)
    return plan


def clip_path(out_dir, voice, line, seed):
    return out_dir / "renders" / voice["name"] / f"line{line}_seed{seed}.wav"


def render_all(out_dir, plan):
    """Render every voice x line x seed that is not on disk yet. Returns how many were made."""
    from backend.genome.search import render_plan
    from backend.speech.tts_formatter import process_tts_string

    jobs = [(v, i, s) for v in plan["voices"] for i in range(len(plan["lines"])) for s in plan["seeds"]]
    todo = [job for job in jobs if not clip_path(out_dir, *job).is_file()]
    print(f"{len(jobs) - len(todo)} of {len(jobs)} clips already made; making {len(todo)}.", flush=True)

    for number, (voice, line, seed) in enumerate(todo, start=1):
        path = clip_path(out_dir, voice, line, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        # The search's own renderer, with no genes: exactly how a baseline is made
        render_plan(voice["reference_audio"], voice["reference_text"], process_tts_string(plan["lines"][line], []),
                    path.with_suffix(".part.wav"), seed)
        os.replace(path.with_suffix(".part.wav"), path)
        print(f"  {number}/{len(todo)} {voice['name']} line {line} seed {seed} ({time.time() - started:.1f} s)", flush=True)
    return len(todo)


def measure_all(out_dir, plan, config):
    """What is heard in each clip and how long it speaks, saved after each one."""
    from backend.metrics.reward import speech_seconds
    from backend.metrics.WER import transcribe

    path = out_dir / "heard.json"
    heard = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    clips = [clip_path(out_dir, v, i, s) for v in plan["voices"] for i in range(len(plan["lines"])) for s in plan["seeds"]]

    for clip in clips:
        key = clip.relative_to(out_dir).as_posix()
        if key in heard or not clip.is_file():
            continue
        heard[key] = {"heard": transcribe(clip), "speech_seconds": speech_seconds(clip, config)}
        _save(path, heard)
    return heard


def compare_all(out_dir, plan, heard):
    """Every ordered pair of renders of one voice and line: (baseline, test) as the sanity check sees them."""
    from backend.metrics.Similarity import speaker_similarity
    from backend.metrics.WER import word_differences

    pairs = []
    for voice in plan["voices"]:
        for line in range(len(plan["lines"])):
            clips = [clip_path(out_dir, voice, line, s) for s in plan["seeds"]]
            clips = [c for c in clips if c.relative_to(out_dir).as_posix() in heard]
            for base, test in itertools.permutations(clips, 2):
                b, t = heard[base.relative_to(out_dir).as_posix()], heard[test.relative_to(out_dir).as_posix()]
                words = word_differences(b["heard"], t["heard"])
                pairs.append({
                    "voice": voice["name"], "line": line, "baseline": base.name, "test": test.name,
                    "errors": words["errors"], "words": words["words"], "wer": words["wer"],
                    "pace": b["speech_seconds"] / t["speech_seconds"],
                    "baseline_seconds": b["speech_seconds"],
                    "similarity": speaker_similarity(test, base),
                })
    return pairs


def _share(values, test):
    return sum(map(test, values)) / len(values) if values else None


def summarise(plan, pairs, config):
    """The numbers, how the current limits do on them, and suggested limits."""
    from backend.metrics.reward import sanity_words_allowed

    summary = {"pairs": len(pairs), "voices": len(plan["voices"]), "seeds": len(plan["seeds"]), "lines": {}}

    for line, text in enumerate(plan["lines"]):
        chosen = [p for p in pairs if p["line"] == line]
        if not chosen:
            continue
        errors = [p["errors"] for p in chosen]
        words = round(np.median([p["words"] for p in chosen]))
        # The fewest words that let CHANCE of the plain pairs through
        needed = next(k for k in range(max(errors) + 1) if _share(errors, lambda e: e <= k) >= CHANCE)
        summary["lines"][line] = {
            "text": text,
            "words": words,
            "errors": {str(k): errors.count(k) for k in range(max(errors) + 1) if errors.count(k)},
            "identical": _share(errors, lambda e: e == 0),
            "words_needed": needed,
            "wer_needed": needed / words if words else None,
            "pass_now": _share(chosen, lambda p: p["errors"] <= sanity_words_allowed(p["words"], config)),
            "pass_with_5_percent": _share(chosen, lambda p: p["wer"] <= 0.05),
        }

    # The similarity limit depends on how much the baseline speaks (sanity_long_seconds)
    long_sims = [p["similarity"] for p in pairs if p["baseline_seconds"] >= config.sanity_long_seconds]
    short_sims = [p["similarity"] for p in pairs if p["baseline_seconds"] < config.sanity_long_seconds]
    long_low = float(np.percentile(long_sims, 100 * (1 - CHANCE))) if long_sims else None
    short_low = float(np.percentile(short_sims, 100 * (1 - CHANCE))) if short_sims else None
    summary["similarity_by_length"] = {
        f"{config.sanity_long_seconds:g} s or more": {"pairs": len(long_sims), "low": long_low,
                                                      "pass_now": _share(long_sims, lambda v: v >= config.sanity_similarity)},
        f"under {config.sanity_long_seconds:g} s": {"pairs": len(short_sims), "low": short_low,
                                                    "pass_now": _share(short_sims, lambda v: v >= config.sanity_similarity_short)},
    }

    pace = [p["pace"] for p in pairs]
    similarity = [p["similarity"] for p in pairs]
    pace_low = float(np.percentile(pace, 100 * (1 - CHANCE)))
    similarity_low = float(np.percentile(similarity, 100 * (1 - CHANCE)))
    summary["pace"] = {"min": min(pace), "low": pace_low, "median": float(np.median(pace)), "max": max(pace),
                       "pass_now": _share(pace, lambda v: v >= config.sanity_pace_min)}
    summary["similarity"] = {"min": min(similarity), "low": similarity_low, "median": float(np.median(similarity))}

    # Suggestions: just loose enough for CHANCE of the plain pairs, rounded outwards
    lines = summary["lines"].values()
    summary["suggested"] = {
        "sanity_wer_words": max(1, max(line["words_needed"] for line in lines)),
        # Only lines where the floor of whole words is not already enough need a share
        "sanity_wer_limit": max([math.ceil(100 * line["wer_needed"]) / 100 for line in lines
                                 if line["words_needed"] > config.sanity_wer_words and line["wer_needed"]]
                                or [config.sanity_wer_limit]),
        "sanity_pace_min": math.floor(20 * pace_low) / 20,
        "sanity_similarity": math.floor(100 * long_low) / 100 if long_low is not None else None,
        "sanity_similarity_short": math.floor(100 * short_low) / 100 if short_low is not None else None,
    }
    return summary


def show(summary, config):
    print(f"\n{summary['pairs']} pairs of plain renders ({summary['voices']} validation voices, {summary['seeds']} seeds each)")
    for line in summary["lines"].values():
        print(f"\nLine of {line['words']} words: {line['text']}")
        print(f"  words heard differently: {', '.join(f'{k}: {n}' for k, n in line['errors'].items())}")
        print(f"  identical transcripts: {line['identical']:.0%}; {CHANCE:.0%} of pairs differ by at most {line['words_needed']} word(s)")
        print(f"  pass the current word limit: {line['pass_now']:.0%}; would pass a flat 5% WER: {line['pass_with_5_percent']:.0%}")
    p, s = summary["pace"], summary["similarity"]
    print(f"\nPace: min {p['min']:.2f}, {100 - CHANCE * 100:.0f}th percentile {p['low']:.2f}, median {p['median']:.2f}; "
          f"pass the current {config.sanity_pace_min:g}: {p['pass_now']:.0%}")
    print(f"Similarity: min {s['min']:.3f}, {100 - CHANCE * 100:.0f}th percentile {s['low']:.3f}, median {s['median']:.3f}")
    for length, numbers in summary["similarity_by_length"].items():
        if numbers["pairs"]:
            print(f"  baseline speech {length}: {numbers['pairs']} pairs, {100 - CHANCE * 100:.0f}th percentile "
                  f"{numbers['low']:.3f}, pass the current limit: {numbers['pass_now']:.0%}")
    print("\nSuggested limits (just loose enough for chance alone):")
    for key, value in summary["suggested"].items():
        if value is not None:
            print(f"  {key} = {value:g}   (now {getattr(config, key):g})")


def main(voices=VOICES, seeds=SEEDS, pick_seed=PICK_SEED, measure_only=False, out_dir=OUT_DIR):
    from backend.metrics.reward import CONFIG

    out_dir = Path(out_dir)
    plan = plan_of(out_dir, voices, seeds, pick_seed)
    if not measure_only:
        render_all(out_dir, plan)
    heard = measure_all(out_dir, plan, CONFIG)
    pairs = compare_all(out_dir, plan, heard)
    if not pairs:
        raise SystemExit("Nothing to compare yet: render first.")
    summary = summarise(plan, pairs, CONFIG)
    _save(out_dir / "pairs.json", pairs)
    _save(out_dir / "summary.json", summary)
    show(summary, CONFIG)
    print(f"\nSaved in {out_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure how much plain renders differ by chance, to set the sanity limits.")
    parser.add_argument("--voices", type=int, default=VOICES, help="validation voices, an even share per dataset")
    parser.add_argument("--seeds", type=int, default=SEEDS, help="renders of each line per voice")
    parser.add_argument("--pick-seed", type=int, default=PICK_SEED, help="which voices are picked")
    parser.add_argument("--measure-only", action="store_true", help="render nothing; measure what is there")
    args = parser.parse_args()
    main(args.voices, args.seeds, args.pick_seed, args.measure_only)

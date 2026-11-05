"""calibrate_similarity.py | Find where "same speaker" ends and "different speaker" begins.

similarity_reward in reward.py needs two limits: sim_full (at or above it, the
clip clearly sounds like the person) and sim_zero (at or below it, it clearly
does not). This script measures them instead of guessing:

    1. Takes several clips from each of many speakers in the voice library.
    2. Compares clips of the SAME speaker with each other, and clips of
       DIFFERENT speakers with each other, using Similarity.py.
    3. Puts sim_full where 95% of same-speaker pairs still pass, and sim_zero
       where 95% of different-speaker pairs already fail.

Two datasets answer two questions:
    VCTK     many accents, read neutrally: does it tell people apart?
    CREMA-D  the same actors in six emotions: does it still recognise a
             person who sounds angry, sad or happy? (What Dramatizer produces)

    python calibrate_similarity.py
    python calibrate_similarity.py --speakers 40 --clips 5

Writes output/calibration/similarity.json and similarity.png. Run it again
after changing the speaker encoder (SIMILARITY_MODEL in .env).
"""
import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np

import model
from Similarity import _embedding
from Voice_Importer import get_clip, read_crema_d, read_vctk

OUT_DIR = model.PROJECT_DIR / "output" / "calibration"

# Short clips give unreliable embeddings, and Dramatizer's references are full sentences
MIN_WORDS = 6


def pick_clips(clips, speakers, per_speaker, rng, spread_key=None):
    """
    A fixed random sample: `speakers` speakers, `per_speaker` clips each.
    With spread_key (e.g. "emotion"), clips are spread over its values.
    """
    by_speaker = {}
    for clip in clips:
        if len(clip.text.split()) >= MIN_WORDS or clip.dataset == "CREMA-D":
            by_speaker.setdefault(clip.speaker, []).append(clip)

    chosen = {}
    for speaker in rng.sample(sorted(s for s, c in by_speaker.items() if len(c) >= per_speaker), speakers):
        pool = by_speaker[speaker]

        if spread_key:
            # One clip per value of the key first (anger, happy, sad...), then the rest
            groups = {}
            for clip in rng.sample(pool, len(pool)):
                groups.setdefault(clip.details.get(spread_key), []).append(clip)
            ordered = [g[i] for i in range(max(map(len, groups.values()))) for g in groups.values() if i < len(g)]
            chosen[speaker] = ordered[:per_speaker]
        else:
            chosen[speaker] = rng.sample(pool, per_speaker)

    return chosen


def pair_similarities(chosen, max_different, rng):
    """Cosine similarities of all same-speaker pairs and a sample of different-speaker pairs."""
    vectors = {}
    for speaker, clips in chosen.items():
        vectors[speaker] = []
        for clip in clips:
            path, _ = get_clip(clip)
            e = _embedding(path)
            vectors[speaker].append(e / np.linalg.norm(e))

    same = [
        float(a @ b)
        for vs in vectors.values()
        for a, b in itertools.combinations(vs, 2)
    ]

    all_different = [
        (s1, i, s2, j)
        for s1, s2 in itertools.combinations(vectors, 2)
        for i in range(len(vectors[s1]))
        for j in range(len(vectors[s2]))
    ]
    sample = rng.sample(all_different, min(max_different, len(all_different)))
    different = [float(vectors[s1][i] @ vectors[s2][j]) for s1, i, s2, j in sample]

    return np.array(same), np.array(different)


def summary(same, different):
    """The numbers that decide the limits."""
    thresholds = np.linspace(0, 1, 1001)
    # Equal error rate: the threshold where a same pair is as likely to fail as a different pair to pass
    false_reject = np.array([(same < t).mean() for t in thresholds])
    false_accept = np.array([(different >= t).mean() for t in thresholds])
    eer_index = int(np.argmin(np.abs(false_reject - false_accept)))

    return {
        "same_pairs": int(len(same)),
        "different_pairs": int(len(different)),
        "same_mean": float(same.mean()),
        "same_p05": float(np.percentile(same, 5)),
        "different_mean": float(different.mean()),
        "different_p95": float(np.percentile(different, 95)),
        "eer_threshold": float(thresholds[eer_index]),
        "eer": float((false_reject[eer_index] + false_accept[eer_index]) / 2),
    }


def plot(results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(results), 1, figsize=(9, 3.2 * len(results)), sharex=True)
    axes = np.atleast_1d(axes)
    bins = np.linspace(0.2, 1.0, 81)

    for ax, (name, r) in zip(axes, results.items()):
        ax.hist(r["different"], bins=bins, alpha=0.6, color="#eb6834", label="different speakers", density=True)
        ax.hist(r["same"], bins=bins, alpha=0.6, color="#2a78d6", label="same speaker", density=True)
        s = r["summary"]
        # The two labels sit on opposite sides of their line, so they never overlap
        lines = sorted([(s["different_p95"], "95% of different below"), (s["same_p05"], "95% of same above")])
        for (x, label), side in zip(lines, ("right", "left")):
            ax.axvline(x, color="#14191d", lw=1, ls="--")
            pad = " " if side == "left" else ""
            ax.text(x, ax.get_ylim()[1] * 0.92, f"{pad}{label} {x:.2f}{' ' if side == 'right' else ''}",
                    fontsize=8, va="top", ha=side)
        ax.set_title(f"{name}: equal error rate {s['eer']:.1%} at {s['eer_threshold']:.2f}", fontsize=10, loc="left")
        ax.legend(fontsize=8, loc="upper left")
        ax.set_yticks([])

    axes[-1].set_xlabel("speaker similarity (cosine)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Calibrate the similarity reward limits.")
    parser.add_argument("--speakers", type=int, default=40, help="speakers per dataset")
    parser.add_argument("--clips", type=int, default=6, help="clips per speaker")
    parser.add_argument("--different", type=int, default=3000, help="different-speaker pairs to sample")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    library = Path(model.setting("VOICE_LIBRARY_DIR"))
    rng = random.Random(args.seed)
    results = {}

    for name, reader, folder, spread in [
        ("VCTK (neutral reading)", read_vctk, model.setting("VCTK_FOLDER", "VCTK"), None),
        ("CREMA-D (six emotions)", read_crema_d, model.setting("CREMA_D_FOLDER", "CREMA-D"), "emotion"),
    ]:
        clips = reader(library / folder)

        if name.startswith("VCTK"):
            clips = [c for c in clips if c.details.get("mic") == "mic1"]

        print(f"{name}: embedding {args.speakers} speakers x {args.clips} clips...")
        chosen = pick_clips(clips, args.speakers, args.clips, rng, spread)
        same, different = pair_similarities(chosen, args.different, rng)
        results[name] = {"same": same.tolist(), "different": different.tolist(), "summary": summary(same, different)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot(results, OUT_DIR / "similarity.png")

    report = {name: r["summary"] for name, r in results.items()}
    report["settings"] = vars(args) | {"min_words": MIN_WORDS, "model": model.setting("SIMILARITY_MODEL", "resemblyzer")}
    (OUT_DIR / "similarity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    for name, s in report.items():
        if name != "settings":
            print(f"\n{name}")
            for key, value in s.items():
                print(f"  {key:16s} {value:.3f}" if isinstance(value, float) else f"  {key:16s} {value}")

    print(f"\nWrote {OUT_DIR / 'similarity.json'} and similarity.png")


if __name__ == "__main__":
    main()

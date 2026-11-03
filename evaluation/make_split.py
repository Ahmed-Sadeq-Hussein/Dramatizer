"""make_split.py | Split the speakers into dev and test, once, with a fixed seed.


    Speakers are split, not clips, so no speaker is in both sets. The test set is
    written as locked: nothing in the project reads it until the final evaluation.
    The script refuses to run again if split.json exists, because a new split would
    change the test set after it may have been looked at.
"""
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
EVAL_DIR = PROJECT / "history" / "eval"
MANIFEST = EVAL_DIR / "manifest.csv"
SPLIT = EVAL_DIR / "split.json"
SEED = 42
TEST_FRACTION = 0.2


def speaker_key(row):
    """Speaker ids repeat across datasets, so the dataset is part of the key."""
    return f"{row['dataset']}/{row['speaker']}"


def make_split(speakers, seed=SEED, test_fraction=TEST_FRACTION):
    ordered = sorted(speakers)
    random.Random(seed).shuffle(ordered)
    n_test = max(1, round(len(ordered) * test_fraction))
    return {"dev": sorted(ordered[n_test:]), "test": sorted(ordered[:n_test])}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if SPLIT.exists() and "--force" not in sys.argv:
        sys.exit("split.json already exists. A new split would change the test set. "
                 "Use --force only if you have not looked at any test results.")

    with MANIFEST.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    parts = make_split({speaker_key(r) for r in rows})
    dev, test = set(parts["dev"]), set(parts["test"])
    clips = {
        "dev": sum(1 for r in rows if speaker_key(r) in dev),
        "test": sum(1 for r in rows if speaker_key(r) in test),
    }

    split = {
        "seed": SEED,
        "test_fraction": TEST_FRACTION,
        "manifest_sha256": sha256(MANIFEST),
        "dev": parts["dev"],
        "test": parts["test"],
        "clips": clips,
        "test_locked": True,
    }
    SPLIT.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"dev: {len(parts['dev'])} speakers, {clips['dev']} clips")
    print(f"test (locked): {len(parts['test'])} speakers, {clips['test']} clips")
    print(f"saved {SPLIT}")


if __name__ == "__main__":
    main()

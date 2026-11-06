"""checks.py | Deterministic checks on the manifest and the split.

    python backend/evaluation/checks.py

The split is the one made by split_voice_data.py, saved in
data/manifests/split-validation-test.json. It is locked there: that file is never
overwritten, so the checks here only read it.

Prints one line per check and exits with an error if any check fails.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
EVAL_DIR = PROJECT / "history" / "eval"
MANIFEST = EVAL_DIR / "manifest.csv"
SPLIT = PROJECT / "backend" / "data" / "manifests" / "split-validation-test.json"


def speaker_key(row):
    """Speaker ids repeat across datasets, so the dataset is part of the key."""
    return f"{row['dataset']}/{row['speaker']}"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_checks(rows, split):
    validation = set(split.get("validation", []))
    test = set(split.get("test", []))
    return [
        ("validation and test are both present", bool(validation) and bool(test)),
        ("validation and test share no speaker", not (validation & test)),
        ("every clip belongs to a speaker in the split", all(speaker_key(r) in validation | test for r in rows)),
        ("no clip has an empty script", all((r.get("text") or "").strip() for r in rows)),
    ]


def main():
    with MANIFEST.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))

    results = run_checks(rows, split)
    for name, ok in results:
        print(f"{'ok  ' if ok else 'FAIL'}  {name}")

    if not all(ok for _, ok in results):
        sys.exit(1)
    print(f"{len(results)} checks passed")


if __name__ == "__main__":
    main()

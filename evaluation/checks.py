"""checks.py | Deterministic checks on the manifest and the split. 

    Prints one line per check and exits with an error if any check fails.
"""
import csv
import json
import sys

from make_split import MANIFEST, SPLIT, sha256, speaker_key


def run_checks(rows, split, manifest_hash):
    dev, test = set(split["dev"]), set(split["test"])
    return [
        ("manifest is the one the split was made from", manifest_hash == split["manifest_sha256"]),
        ("dev and test share no speaker", not (dev & test)),
        ("every clip belongs to a speaker in the split", all(speaker_key(r) in dev | test for r in rows)),
        ("test set is marked locked", split.get("test_locked") is True),
        ("no clip has an empty script", all((r.get("text") or "").strip() for r in rows)),
    ]


def main():
    with MANIFEST.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))

    results = run_checks(rows, split, sha256(MANIFEST))
    for name, ok in results:
        print(f"{'ok  ' if ok else 'FAIL'}  {name}")

    if not all(ok for _, ok in results):
        sys.exit(1)
    print(f"{len(results)} checks passed")


if __name__ == "__main__":
    main()

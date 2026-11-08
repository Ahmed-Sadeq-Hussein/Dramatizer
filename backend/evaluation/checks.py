"""checks.py | Deterministic checks on the manifest and the split.

    python backend/evaluation/checks.py

The split is the one made by split_voice_data.py, saved in
data/manifests/split-validation-test.json. It is locked there: that file is never
overwritten, so the checks here only read it.

It also checks the data contract (backend/evaluation/DATA_CONTRACT.md): the
manifest's columns, the allowed value of every label, and that each dataset has
its own licence.

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


# The data contract, as DATA_CONTRACT.md writes it out. "" means the dataset has no such label.
FIELDS = ["dataset", "clip_id", "speaker", "text", "emotion", "intensity", "mic", "licence", "source"]
LICENCE = {"CREMA-D": "ODbL", "VCTK": "CC BY 4.0", "LibriTTS-R": "CC BY 4.0"}
ALLOWED = {
    "emotion": {"CREMA-D": {"anger", "disgust", "fear", "happy", "neutral", "sad"}},
    "intensity": {"CREMA-D": {"low", "medium", "high", "unspecified"}},
    "mic": {"VCTK": {"mic1", "mic2"}},
}


def contract_checks(rows, fields=None, audio_exists=None):
    """
    The data contract, one (name, passed) pair per rule. `fields` are the manifest's
    columns (the keys of the first row when not given). `audio_exists` is a function
    that says whether a source file is there; None skips that check (no files needed).
    """
    fields = list(fields if fields is not None else (rows[0].keys() if rows else []))

    def label_ok(row, label):
        allowed = ALLOWED[label].get(row.get("dataset"))
        value = row.get(label) or ""
        return value in allowed if allowed else value == ""

    keys = [(r.get("dataset"), r.get("clip_id")) for r in rows]
    results = [
        ("the manifest has exactly the contract's columns", fields == FIELDS),
        ("every clip comes from a known dataset", all(r.get("dataset") in LICENCE for r in rows)),
        ("every clip carries its dataset's licence", all(r.get("licence") == LICENCE.get(r.get("dataset")) for r in rows)),
        ("no clip is listed twice", len(keys) == len(set(keys))),
        ("every clip has a speaker", all((r.get("speaker") or "").strip() for r in rows)),
        ("emotion is an allowed label (CREMA-D only)", all(label_ok(r, "emotion") for r in rows)),
        ("intensity is an allowed label (CREMA-D only)", all(label_ok(r, "intensity") for r in rows)),
        ("mic is an allowed label (VCTK only)", all(label_ok(r, "mic") for r in rows)),
    ]
    if audio_exists is not None:
        results.append(("every clip's audio file exists", all(audio_exists(r.get("source") or "") for r in rows)))
    return results


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

    results = run_checks(rows, split) + contract_checks(rows, audio_exists=lambda p: Path(p).is_file())
    for name, ok in results:
        print(f"{'ok  ' if ok else 'FAIL'}  {name}")

    if not all(ok for _, ok in results):
        sys.exit(1)
    print(f"{len(results)} checks passed")


if __name__ == "__main__":
    main()

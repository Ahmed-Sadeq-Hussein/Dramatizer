"""checks.py | Deterministic checks on the manifest and the split.

    python backend/evaluation/checks.py

The split is the voice manifest made by split_voice_data.py, saved in
data/manifests/voice-split-v2.json: every voice, the one clip it uses, and its
split. It is locked there: that file is never overwritten, so the checks here
only read it.

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
SPLIT = PROJECT / "backend" / "data" / "manifests" / "voice-split-v2.json"
SPLITS = ("validation", "test")
NEUTRAL = {"", "neutral"}


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
    unused = {item["speaker"] for item in split.get("unused", [])}
    return [
        ("validation and test are both present", bool(validation) and bool(test)),
        ("validation and test share no speaker", not (validation & test)),
        ("every clip's speaker is a validation voice, a test voice or listed as unused",
         all(speaker_key(r) in validation | test | unused for r in rows)),
        ("no clip has an empty script", all((r.get("text") or "").strip() for r in rows)),
    ]


def voice_checks(rows, split):
    """
    The voice manifest: each voice is listed once with one clip, in the split its
    lists say, and that clip is a usable one of its own speaker. One (name, passed) pair per rule.
    """
    voices = split.get("voices", [])
    rules = split.get("rules", {})
    by_clip = {(r.get("dataset"), r.get("clip_id")): r for r in rows}
    keys = [v.get("voice") for v in voices]
    listed = {name: set(split.get(name, [])) for name in SPLITS}
    unused = {item["speaker"] for item in split.get("unused", [])}
    fraction = rules.get("test_fraction", 0.2)

    def own_clip(v):
        row = by_clip.get((v.get("dataset"), v.get("clip_id")))
        return row is not None and speaker_key(row) == v.get("voice")

    def usable(v):
        row = by_clip.get((v.get("dataset"), v.get("clip_id")))
        seconds = v.get("speech_seconds")
        return (row is not None and (row.get("emotion") or "") in NEUTRAL and seconds is not None
                and rules.get("min_seconds", 3.0) <= seconds <= rules.get("max_seconds", 25.0))

    shares_ok = True
    for dataset in {v.get("dataset") for v in voices}:
        mine = [v for v in voices if v.get("dataset") == dataset]
        tested = sum(v.get("split") == "test" for v in mine)
        shares_ok &= tested == max(1, round(len(mine) * fraction))

    return [
        ("every voice is listed once, with one clip", len(keys) == len(set(keys)) and all(v.get("clip_id") for v in voices)),
        ("the split's lists hold exactly the voices, each in its own split",
         all(listed[name] == {v.get("voice") for v in voices if v.get("split") == name} for name in SPLITS)),
        ("no unused speaker is also a voice", not (unused & set(keys))),
        ("every voice's clip is in the manifest and is its own speaker's", all(own_clip(v) for v in voices)),
        ("every voice's clip is neutral, within the length limits", all(usable(v) for v in voices)),
        ("each dataset's test share is 20 %, rounded, at least one voice", bool(voices) and shares_ok),
    ]


def main():
    with MANIFEST.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))

    results = run_checks(rows, split) + voice_checks(rows, split) + contract_checks(rows, audio_exists=lambda p: Path(p).is_file())
    for name, ok in results:
        print(f"{'ok  ' if ok else 'FAIL'}  {name}")

    if not all(ok for _, ok in results):
        sys.exit(1)
    print(f"{len(results)} checks passed")


if __name__ == "__main__":
    main()

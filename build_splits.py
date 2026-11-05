"""build_splits.py | Which speakers and lines are for training, validation and the final test.

Writes data/manifests/splits-v1.json from the data manifest (build_manifest.py).
The same manifest gives the same file, byte for byte.

    python build_splits.py

Speakers: 70% train, 15% validation, 15% test, split by person and never by clip,
so nobody's voice is on both sides. The same id can be two people in two datasets
(1089 is in CREMA-D and in LibriTTS-R), so a speaker is always dataset + id.
Each dataset is split on its own, and so are the speakers with and without a
neutral clip, because only those with one can be cloned for the baseline.

Lines: held-out sentences for the clones to say, half validation and half test,
with the same mix of lengths. They were written for this project and are in none
of the datasets. A line in a dataset would be one the speaker's own recording
could be compared with, and one the emphasis test (emphasis_testset.json) already
tuned a choice on would not be held out.

Who goes where comes from a hash of the name, not from a random generator, so it
cannot change with a library version. Test speakers and lines stay unused until
the end of the project (PLAN.md section 5).
"""
import hashlib
import json
from collections import defaultdict

from build_manifest import MANIFEST
from WER import normalise

SPLITS_FILE = MANIFEST.parent / "splits-v1.json"
EMPHASIS_LINES = MANIFEST.parent.parent.parent / "emphasis_testset.json"
SALT = "dramatizer-splits-v1"
VALIDATION_SHARE = 0.15
TEST_SHARE = 0.15

LINES = {
    "short": [
        "Nobody told me you were coming.",
        "Give me back that letter.",
        "Why is the door still open?",
        "We have to leave tonight.",
        "I never asked for any of this.",
        "You promised you would wait for me.",
    ],
    "medium": [
        "The lights went out just as the train reached the bridge.",
        "She folded the note twice and slipped it under the door.",
        "If you walk away now, there is no coming back.",
        "I can still hear the rain on the roof of that old house.",
        "Tell your brother the boat leaves before the sun comes up.",
        "Do you really think anyone in this town will believe you?",
        "He laughed so hard that the whole kitchen went quiet.",
        "All I wanted was one honest answer from somebody in this family.",
    ],
    "long": [
        "By the time the doctor arrived, the fire had burned down to ash and the children had fallen asleep by the window.",
        "I have kept your secret for almost ten years, but I am tired, and I will not lie to them again.",
        "When the music finally stopped, everyone turned to look at the old man waiting alone beside the empty stage.",
        "You can shout at me all you like, but the money is gone and shouting will not bring it back.",
        "We walked along the river until the road disappeared, and then we sat down on the stones and waited for morning.",
        "Before you open that box, I need you to understand that nothing in your life will ever be the same.",
    ],
}


def _hashed(names):
    """The names in a fixed order that looks random and does not depend on the order given."""
    return sorted(names, key=lambda name: hashlib.sha256(f"{SALT}|{name}".encode("utf-8")).hexdigest())


def load_manifest():
    """The manifest, after checking its clips are still the ones its hash promises."""
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    lines = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in manifest["clips"]]
    if hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest() != manifest["clips_sha256"]:
        raise ValueError(f"{MANIFEST} was changed by hand. Run build_manifest.py again.")
    return manifest


def split_speakers(clips):
    """{split: {dataset: [speaker, ...]}}, every speaker in exactly one split."""
    groups = defaultdict(set)
    neutral = {(r["dataset"], r["speaker"]) for r in clips if r["emotion"] in (None, "neutral")}
    for r in clips:
        key = (r["dataset"], r["speaker"])
        groups[(r["dataset"], key in neutral)].add(r["speaker"])

    splits = {name: defaultdict(list) for name in ("train", "validation", "test")}
    for (dataset, _), speakers in sorted(groups.items()):
        ordered = _hashed(f"{dataset}|{speaker}" for speaker in speakers)
        validation = round(len(ordered) * VALIDATION_SHARE)
        test = round(len(ordered) * TEST_SHARE)
        for i, name in enumerate(ordered):
            split = "validation" if i < validation else "test" if i < validation + test else "train"
            splits[split][dataset].append(name.split("|", 1)[1])

    return {split: {dataset: sorted(speakers[dataset]) for dataset in sorted(speakers)} for split, speakers in splits.items()}


def split_lines():
    """{split: [{"line_id", "length", "text"}, ...]}, half of each length in each split."""
    splits = {"validation": [], "test": []}
    for length, lines in LINES.items():
        for i, text in enumerate(_hashed(lines)):
            line_id = f"{length}-{lines.index(text) + 1:02d}"
            splits["validation" if i % 2 == 0 else "test"].append({"line_id": line_id, "length": length, "text": text})
    return {split: sorted(lines, key=lambda line: line["line_id"]) for split, lines in splits.items()}


def check_lines_are_new(clips):
    """Refuses a held-out line that is already in a dataset or in the emphasis test."""
    emphasis = json.loads(EMPHASIS_LINES.read_text(encoding="utf-8"))["items"]
    used = {normalise(r["transcript"]) for r in clips} | {normalise(item["line"]) for item in emphasis}
    repeated = [text for lines in LINES.values() for text in lines if normalise(text) in used]
    if repeated:
        raise ValueError(f"These lines are not held out: {repeated}")


def counts(speakers, clips):
    """Speakers, clips and hours in one split, per dataset."""
    result = {}
    for dataset, names in speakers.items():
        chosen = set(names)
        subset = [r for r in clips if r["dataset"] == dataset and r["speaker"] in chosen]
        result[dataset] = {
            "speakers": len(chosen),
            "speakers_with_a_neutral_clip": len({r["speaker"] for r in subset if r["emotion"] in (None, "neutral")}),
            "clips": len(subset),
            "hours_of_speech": round(sum(r["speech_seconds"] for r in subset) / 3600, 2),
        }
    return result


def build():
    manifest = load_manifest()
    clips = manifest["clips"]
    check_lines_are_new(clips)

    speakers = split_speakers(clips)
    result = {
        "splits": SPLITS_FILE.stem,
        "manifest": manifest["manifest"],
        "clips_sha256": manifest["clips_sha256"],
        "rules": {
            "salt": SALT,
            "speaker": "dataset + speaker id; the same id in two datasets is two people",
            "shares": {"train": 1 - VALIDATION_SHARE - TEST_SHARE, "validation": VALIDATION_SHARE, "test": TEST_SHARE},
            "split_separately": ["dataset", "has a neutral clip"],
            "test": "unused until the end of the project",
        },
        "counts": {split: counts(names, clips) for split, names in speakers.items()},
        "speakers": speakers,
        "lines": split_lines(),
    }

    SPLITS_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return result


if __name__ == "__main__":
    result = build()

    print(f"Wrote {SPLITS_FILE}")
    for split, datasets in result["counts"].items():
        total = sum(c["speakers"] for c in datasets.values())
        neutral = sum(c["speakers_with_a_neutral_clip"] for c in datasets.values())
        print(f"  {split:10s} {total:3d} speakers, {neutral:3d} with a neutral clip, {sum(c['clips'] for c in datasets.values()):5d} clips")
    for split, lines in result["lines"].items():
        print(f"  {split:10s} {len(lines):3d} lines")

"""split_voice_data.py | The voices of the datasets, the one clip each voice uses, and the split between validation and the final test.

A voice is one speaker (dataset + id) who has at least one usable clip: neutral,
with 3-25 s of speech (Picky_voice.py), and for VCTK from microphone mic1 and not
one of the silent clips. Each voice uses ONE clip, its longest usable one,
because a longer reference gives a clone that is more like the speaker and is
heard more reliably (Picky_voice.py has the measurements). On a tie the clip id
decides, so the choice never depends on reading order.

The split is made over these voices, not over every speaker: a speaker with no
usable clip could never be used, and counting them would leave too few usable
voices in a test set (CREMA-D had 5 of 18). Each dataset is split on its own,
80 % validation and 20 % test, with seed 42. Speakers without a usable clip are
listed as unused, so every speaker is accounted for.

The result is the voice manifest, SPLITS_FILE: every voice, its clip (id, length,
transcript, checksum) and its split. It is made once and never overwritten:

    python main_backend.py split          # or: python -m backend.voice_data.split_voice_data

Picking takes the clips from the readers in Voice_Importer.py (read_crema_d,
read_vctk, read_libritts_r) and always gives each picked voice its longest clip.

One dataset: voices is either how many to take at random, or which ones.

    clips = read_crema_d("D:/Voices library/CREMA-D")
    pick_crema_d(clips, 10)                                 # 10 random actors, each with its longest clip
    pick_crema_d(clips, ["1001", "1002_DFA_NEU_XX"])        # a speaker's longest clip, or that exact clip
    pick_crema_d(clips, 10, min_seconds=2.5, max_seconds=6) # the longest clip within these limits

All datasets: an even share from each, read from the voice library. Custom voices
are never included.

    pick_even(30)                                           # 10 from each dataset
    pick_even(30, min_seconds=4, max_seconds=10, seed=1)

split= keeps a pick inside one split: "validation" or "test". The test voices
cannot reach a baseline by accident, because the split is locked in a file.

    pick_even(30, split="validation")
    pick_crema_d(clips, 5, split="validation")
"""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from backend import model
from backend.voice_data.build_manifest import MANIFEST, NO_SPEECH, SOURCES, VCTK_MIC
from backend.voice_data.Picky_voice import MAX_SECONDS, MIN_SECONDS, measure, select_clips

SPLITS_FILE = MANIFEST.parent / "voice-split-v2.json"
SPLIT_NAME = "voice-split-v2"
SPLIT_SEED = 42
TEST_FRACTION = 0.2
SPLITS = ("validation", "test")


def usable_clips(dataset, clips):
    """The clips of one dataset that may be a reference at all: VCTK from one microphone, without the silent clips."""
    if dataset == "VCTK":
        return [clip for clip in clips if clip.details.get("mic") == VCTK_MIC and clip.clip_id not in NO_SPEECH]
    return list(clips)


def longest_clips(clips, min_seconds=MIN_SECONDS, max_seconds=MAX_SECONDS):
    """{speaker: [(seconds, clip), ...]}: the neutral clips within the limits, longest first, then by clip id."""
    usable = select_clips(clips, min_seconds, max_seconds, duplicate_actor=True, emotions=False)
    seconds = measure(usable)
    by_speaker = defaultdict(list)
    for clip in usable:
        by_speaker[clip.speaker].append((seconds[clip.clip_id], clip))
    for found in by_speaker.values():
        found.sort(key=lambda pair: (-pair[0], pair[1].clip_id))
    return dict(by_speaker)


# --------------------------------------------------------------------------- #
# The voice manifest and its split
# --------------------------------------------------------------------------- #

def load_voice_manifest(path=SPLITS_FILE):
    """The whole voice manifest (see the top of this file), checked for leakage."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Make it once: python main_backend.py split")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    check_no_leakage({name: manifest[name] for name in SPLITS})
    return manifest


def load_splits(path=SPLITS_FILE):
    """{"validation": [voice keys], "test": [voice keys]} from the voice manifest. A voice key is dataset/speaker."""
    manifest = load_voice_manifest(path)
    return {name: manifest[name] for name in SPLITS}


def voice_clips(split=None, path=SPLITS_FILE):
    """{voice key: clip id} of the chosen clip of every voice, or of one split's voices."""
    return {voice["voice"]: voice["clip_id"] for voice in load_voice_manifest(path)["voices"]
            if split is None or voice["split"] == split}


def _speakers_in(split):
    """The voice keys of one split. The key is dataset/id, so the same id in another dataset is another person."""
    splits = load_splits()
    if split not in splits:
        raise ValueError(f"Unknown split {split!r}: use one of {list(splits)}.")
    return set(splits[split])


def split_of(dataset, speaker):
    """Which split a voice is in: "validation", "test", or None (an unused speaker, or a custom voice)."""
    key = f"{dataset}/{speaker}"
    for split, keys in load_splits().items():
        if key in keys:
            return split
    return None


def voice_origin(clip):
    """Where a cloned voice comes from: its dataset, speaker, clip, the clip's labels and the voice's split."""
    return {
        "dataset": clip.dataset,
        "speaker": clip.speaker,
        "clip_id": clip.clip_id,
        "details": {key: str(value) for key, value in clip.details.items()},
        "split": split_of(clip.dataset, clip.speaker),
    }


def check_no_leakage(splits):
    """Stops with an error if a voice is in two splits."""
    seen = {}
    for name, keys in splits.items():
        for key in keys:
            if key in seen:
                raise ValueError(f"{key} is in both {seen[key]} and {name}")
            seen[key] = name


def split_voices(keys):
    """
    {"validation": [voice keys], "test": [voice keys]}, split inside each dataset, so
    every dataset gets 20 % test (at least one voice) and the rest validation.
    """
    by_dataset = defaultdict(list)
    for key in keys:
        by_dataset[key.split("/", 1)[0]].append(key)

    splits = {"validation": [], "test": []}
    for dataset in sorted(by_dataset):
        ordered = sorted(by_dataset[dataset])
        random.Random(SPLIT_SEED).shuffle(ordered)
        n_test = test_count(len(ordered))
        splits["test"] += ordered[:n_test]
        splits["validation"] += ordered[n_test:]

    splits = {name: sorted(found) for name, found in splits.items()}
    check_no_leakage(splits)
    return splits


def test_count(voices):
    """How many of a dataset's voices go to the final test: 20 %, rounded, at least one."""
    return max(1, round(voices * TEST_FRACTION))


def _clip_checksums():
    """{(dataset, clip id): sha256 of the audio} from the clip manifest, voices-v1.json."""
    clips = json.loads(MANIFEST.read_text(encoding="utf-8"))["clips"]
    return {(clip["dataset"], clip["clip_id"]): clip["sha256"] for clip in clips}


def build_voice_manifest(clips):
    """The voice manifest for every clip of the library (see the top of this file). Nothing is saved."""
    checksums = _clip_checksums()
    by_dataset = defaultdict(list)
    for clip in clips:
        by_dataset[clip.dataset].append(clip)

    voices, unused = [], []
    for dataset in sorted(by_dataset):
        everyone = sorted({clip.speaker for clip in by_dataset[dataset]})
        longest = longest_clips(usable_clips(dataset, by_dataset[dataset]))
        for speaker in everyone:
            if speaker not in longest:
                unused.append({"speaker": f"{dataset}/{speaker}",
                               "why": f"no neutral clip with {MIN_SECONDS:g}-{MAX_SECONDS:g} s of speech"})
                continue
            seconds, clip = longest[speaker][0]
            voices.append({
                "voice": f"{dataset}/{speaker}",
                "dataset": dataset,
                "speaker": speaker,
                "clip_id": clip.clip_id,
                "speech_seconds": seconds,
                "usable_clips": len(longest[speaker]),
                "transcript": clip.text,
                "sha256": checksums.get((dataset, clip.clip_id)),
                "details": {key: str(value) for key, value in sorted(clip.details.items())},
            })

    splits = split_voices([voice["voice"] for voice in voices])
    where = {key: name for name, keys in splits.items() for key in keys}
    for voice in voices:
        voice["split"] = where[voice["voice"]]

    counts = {}
    for dataset in sorted(by_dataset):
        mine = [voice for voice in voices if voice["dataset"] == dataset]
        counts[dataset] = {name: sum(voice["split"] == name for voice in mine) for name in SPLITS}
        counts[dataset]["unused speakers"] = sum(item["speaker"].startswith(dataset + "/") for item in unused)
        counts[dataset]["mean seconds"] = round(sum(voice["speech_seconds"] for voice in mine) / max(1, len(mine)), 2)
    counts["All"] = {name: sum(counts[d][name] for d in by_dataset) for name in (*SPLITS, "unused speakers")}

    return {
        "name": SPLIT_NAME,
        "what": "Every voice (dataset + speaker) with a usable clip, the one clip it uses, and its split.",
        "rules": {
            "voice": "a speaker with at least one neutral clip of "
                     f"{MIN_SECONDS:g}-{MAX_SECONDS:g} s of speech (VCTK: mic {VCTK_MIC}, silent clips left out)",
            "clip": "the voice's longest usable clip; on a tie, the first clip id",
            "split": f"by voice, each dataset on its own, {round(100 * (1 - TEST_FRACTION))} % validation and "
                     f"{round(100 * TEST_FRACTION)} % test (rounded, at least 1), shuffled with seed {SPLIT_SEED}",
            "test": "locked: unused until the final evaluation",
            "min_seconds": MIN_SECONDS,
            "max_seconds": MAX_SECONDS,
            "seed": SPLIT_SEED,
            "test_fraction": TEST_FRACTION,
        },
        "clip_manifest": MANIFEST.name,
        "clip_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        "counts": counts,
        "validation": splits["validation"],
        "test": splits["test"],
        "unused": unused,
        "voices": voices,
    }


def save_voice_manifest(manifest, path=SPLITS_FILE):
    """Writes the voice manifest once. The test set is locked, so an existing file is never overwritten."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists. A new split would change the locked test set.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def make_splits():
    """Makes the voice manifest from the whole library and saves it. Run once, then never again."""
    manifest = build_voice_manifest(read_library())
    save_voice_manifest(manifest)
    return manifest


# --------------------------------------------------------------------------- #
# Picking voices
# --------------------------------------------------------------------------- #

def _pick(clips, dataset, voices, min_seconds, max_seconds, seed, split):
    if split is not None:
        allowed = _speakers_in(split)
        clips = [clip for clip in clips if f"{dataset}/{clip.speaker}" in allowed]

    what = f"a neutral clip of {min_seconds:g}-{max_seconds:g} s" + (f" in {split}" if split else "")
    by_speaker = longest_clips(clips, min_seconds, max_seconds)
    if not by_speaker:
        raise ValueError(f"{dataset}: no speaker has {what}.")
    by_id = {clip.clip_id: clip for found in by_speaker.values() for _, clip in found}

    def longest(speaker):
        return by_speaker[speaker][0][1]

    if isinstance(voices, int):
        if not 1 <= voices <= len(by_speaker):
            raise ValueError(f"{dataset}: asked for {voices} voices, but {len(by_speaker)} speakers have {what}.")
        chosen = [longest(speaker) for speaker in random.Random(seed).sample(sorted(by_speaker), voices)]
    else:
        chosen = []
        for name in _flat(voices):
            if name in by_id:
                chosen.append(by_id[name])
            elif name in by_speaker:
                chosen.append(longest(name))
            else:
                raise ValueError(f"{dataset}: {name!r} is not a speaker or clip with {what}.")

    return sorted(chosen, key=lambda c: (c.speaker, c.clip_id))


def _flat(voices):
    """The names in a list, or in a list of lists."""
    for item in voices:
        if isinstance(item, str):
            yield item
        else:
            yield from _flat(item)


def pick_crema_d(clips, voices, min_seconds=MIN_SECONDS, max_seconds=MAX_SECONDS, seed=0, split=None):
    """Neutral CREMA-D clips: a number of random actors with their longest clip each, or the named speakers and clips."""
    return _pick(usable_clips("CREMA-D", clips), "CREMA-D", voices, min_seconds, max_seconds, seed, split)


def pick_vctk(clips, voices, min_seconds=MIN_SECONDS, max_seconds=MAX_SECONDS, seed=0, split=None):
    """The same for VCTK. One microphone only, and without the clips that hold no speech."""
    return _pick(usable_clips("VCTK", clips), "VCTK", voices, min_seconds, max_seconds, seed, split)


def pick_libritts_r(clips, voices, min_seconds=MIN_SECONDS, max_seconds=MAX_SECONDS, seed=0, split=None):
    """The same for LibriTTS-R. Its clips have no emotion labels, so they all count as neutral."""
    return _pick(usable_clips("LibriTTS-R", clips), "LibriTTS-R", voices, min_seconds, max_seconds, seed, split)


PICKERS = {"VCTK": pick_vctk, "CREMA-D": pick_crema_d, "LibriTTS-R": pick_libritts_r}


def pick_even(number, min_seconds=MIN_SECONDS, max_seconds=MAX_SECONDS, seed=0, split=None, library=None):
    """
    `number` random voices, an even share from each dataset. When it does not divide
    evenly, the first datasets in PICKERS take one more. Custom voices are not in PICKERS.
    library is {dataset: clips} if they are already read; without it, they are read here.
    """
    share, extra = divmod(number, len(PICKERS))
    folder = Path(model.setting("VOICE_LIBRARY_DIR"))
    chosen = []

    for i, (name, picker) in enumerate(PICKERS.items()):
        wanted = share + (i < extra)
        if wanted:
            if library is None:
                setting, default = SOURCES[name]["folder"]
                clips = SOURCES[name]["reader"](folder / model.setting(setting, default))
            else:
                clips = library[name]
            chosen += picker(clips, wanted, min_seconds, max_seconds, seed, split)

    return chosen


def read_library():
    """Every clip in all the datasets in PICKERS, read the same way pick_even reads them."""
    folder = Path(model.setting("VOICE_LIBRARY_DIR"))
    clips = []
    for name in PICKERS:
        setting, default = SOURCES[name]["folder"]
        clips += SOURCES[name]["reader"](folder / model.setting(setting, default))
    return clips


def speaker_key(clip):
    """The one name of a clip's voice: dataset + id, because the same id can be two people in two datasets."""
    return f"{clip.dataset}/{clip.speaker}"


def group_by_speaker(clips):
    """{voice key: [clips]} for a list of clips from any of the readers."""
    groups = defaultdict(list)
    for clip in clips:
        groups[speaker_key(clip)].append(clip)
    return dict(groups)


def clips_in_split(clips, speakers):
    """The clips that belong to the given voice keys."""
    wanted = set(speakers)
    return [clip for clip in clips if speaker_key(clip) in wanted]


if __name__ == "__main__":
    result = make_splits()
    print(f"{'':12s} {'validation':>10s} {'test':>5s} {'unused speakers':>16s}")
    for dataset, numbers in result["counts"].items():
        print(f"{dataset:12s} {numbers['validation']:10d} {numbers['test']:5d} {numbers['unused speakers']:16d}")
    print(f"saved {SPLITS_FILE}")

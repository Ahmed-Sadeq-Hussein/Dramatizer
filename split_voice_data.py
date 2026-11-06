"""split_voice_data.py | Picks voices from the datasets, and keeps the train-free split between validation and the final test.

Picking takes the clips from the readers in Voice_Importer.py (read_crema_d,
read_vctk, read_libritts_r). A voice here is one clip that is neutral and whose
speech is between min_seconds and max_seconds long (measured by Picky_voice.py).

One dataset: voices is either how many to take at random, or which ones.

    clips = read_crema_d("D:/Voices library/CREMA-D")
    pick_crema_d(clips, 10)                                 # 10 random actors, one clip each
    pick_crema_d(clips, ["1001", "1002_DFA_NEU_XX"])        # a speaker's random clip, or that exact clip
    pick_crema_d(clips, 10, min_seconds=2.5, max_seconds=6)

All datasets: an even share from each, read from the voice library. Custom voices
are never included.

    pick_even(30)                                           # 10 from each dataset
    pick_even(30, min_seconds=4, max_seconds=10, seed=1)

split= keeps a pick inside one split: "validation" or "test". The test speakers
cannot reach a baseline by accident, because the split is locked in a file.

    pick_even(30, split="validation")
    pick_crema_d(clips, 5, split="validation")

The split is made once from the whole library, by make_splits(), and saved to
SPLITS_FILE. Each dataset is split on its own, 80 % validation and 20 % test, by
speaker (dataset + id), with seed 42. Run this file once to make the split:

    python split_voice_data.py

The same seed gives the same voices. The saved file is never overwritten.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import model
from build_manifest import MANIFEST, NO_SPEECH, SOURCES, VCTK_MIC
from Picky_voice import MAX_SECONDS, MIN_SECONDS, select_clips

SPLITS_FILE = MANIFEST.parent / "split-validation-test.json"
SPLIT_SEED = 42
TEST_FRACTION = 0.2


def _speakers_in(split, dataset):
    """The speaker keys of one split. The key is dataset/id, so the same id in another dataset is another person."""
    splits = load_splits(SPLITS_FILE)
    if split not in splits:
        raise ValueError(f"Unknown split {split!r}: use one of {list(splits)}.")
    return set(splits[split])


def split_of(dataset, speaker):
    """Which split a speaker is in: "validation", "test", or None (for example a custom voice)."""
    key = f"{dataset}/{speaker}"
    for split, keys in load_splits(SPLITS_FILE).items():
        if key in keys:
            return split
    return None


def voice_origin(clip):
    """Where a cloned voice comes from: its dataset, speaker, clip, the clip's labels and the speaker's split."""
    return {
        "dataset": clip.dataset,
        "speaker": clip.speaker,
        "clip_id": clip.clip_id,
        "details": {key: str(value) for key, value in clip.details.items()},
        "split": split_of(clip.dataset, clip.speaker),
    }


def _pick(clips, dataset, voices, min_seconds, max_seconds, seed, split):
    if split is not None:
        allowed = _speakers_in(split, dataset)
        clips = [clip for clip in clips if f"{dataset}/{clip.speaker}" in allowed]

    what = f"a neutral clip of {min_seconds:g}-{max_seconds:g} s" + (f" in {split}" if split else "")
    usable = select_clips(clips, min_seconds, max_seconds, duplicate_actor=True, emotions=False)
    if not usable:
        raise ValueError(f"{dataset}: no speaker has {what}.")

    by_id = {clip.clip_id: clip for clip in usable}
    by_speaker = defaultdict(list)
    for clip in sorted(usable, key=lambda c: c.clip_id):
        by_speaker[clip.speaker].append(clip)

    rng = random.Random(seed)

    if isinstance(voices, int):
        if not 1 <= voices <= len(by_speaker):
            raise ValueError(f"{dataset}: asked for {voices} voices, but {len(by_speaker)} speakers have {what}.")
        chosen = [rng.choice(by_speaker[speaker]) for speaker in rng.sample(sorted(by_speaker), voices)]
    else:
        chosen = []
        for name in _flat(voices):
            if name in by_id:
                chosen.append(by_id[name])
            elif name in by_speaker:
                chosen.append(rng.choice(by_speaker[name]))
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
    """Neutral CREMA-D clips: a number of random actors with one clip each, or the named speakers and clips."""
    return _pick(clips, "CREMA-D", voices, min_seconds, max_seconds, seed, split)


def pick_vctk(clips, voices, min_seconds=MIN_SECONDS, max_seconds=MAX_SECONDS, seed=0, split=None):
    """The same for VCTK. One microphone only, and without the clips that hold no speech."""
    clips = [clip for clip in clips if clip.details.get("mic") == VCTK_MIC and clip.clip_id not in NO_SPEECH]
    return _pick(clips, "VCTK", voices, min_seconds, max_seconds, seed, split)


def pick_libritts_r(clips, voices, min_seconds=MIN_SECONDS, max_seconds=MAX_SECONDS, seed=0, split=None):
    """The same for LibriTTS-R. Its clips have no emotion labels, so they all count as neutral."""
    return _pick(clips, "LibriTTS-R", voices, min_seconds, max_seconds, seed, split)


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
    """The one name of a clip's speaker: dataset + id, because the same id can be two people in two datasets."""
    return f"{clip.dataset}/{clip.speaker}"


def group_by_speaker(clips):
    """{speaker key: [clips]} for a list of clips from any of the readers."""
    groups = defaultdict(list)
    for clip in clips:
        groups[speaker_key(clip)].append(clip)
    return dict(groups)


def split_speakers(clips):
    """
    {"validation": [speaker keys], "test": [speaker keys]}, split inside each dataset,
    so every dataset gets about 80 % validation and 20 % test. 
    Applied per dataset. Pass every clip in the library.
    """
    by_dataset = defaultdict(list)
    for key in group_by_speaker(clips):
        by_dataset[key.split("/", 1)[0]].append(key)

    splits = {"validation": [], "test": []}
    for dataset in sorted(by_dataset):
        ordered = sorted(by_dataset[dataset])
        random.Random(SPLIT_SEED).shuffle(ordered)
        n_test = max(1, round(len(ordered) * TEST_FRACTION))
        splits["test"] += ordered[:n_test]
        splits["validation"] += ordered[n_test:]

    splits = {name: sorted(keys) for name, keys in splits.items()}
    check_no_leakage(splits)
    return splits


def check_no_leakage(splits):
    """Stops with an error if a speaker is in two splits."""
    seen = {}
    for name, keys in splits.items():
        for key in keys:
            if key in seen:
                raise ValueError(f"{key} is in both {seen[key]} and {name}")
            seen[key] = name


def save_splits(splits, path=SPLITS_FILE):
    """Writes the split once. The test set is locked, so an existing file is never overwritten."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists. A new split would change the locked test set.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2), encoding="utf-8")


def load_splits(path=SPLITS_FILE):
    """{"validation": [speaker keys], "test": [speaker keys]} from the saved file, checked for leakage."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run this file once: python split_voice_data.py")
    splits = json.loads(path.read_text(encoding="utf-8"))
    check_no_leakage(splits)
    return splits


def clips_in_split(clips, speakers):
    """The clips that belong to the given speaker keys."""
    wanted = set(speakers)
    return [clip for clip in clips if speaker_key(clip) in wanted]


def make_splits():
    """Makes the split from the whole library and saves it. Run once, then never again."""
    splits = split_speakers(read_library())
    save_splits(splits)
    return splits


if __name__ == "__main__":
    result = make_splits()
    for name, keys in result.items():
        print(f"{name:10s} {len(keys):4d} speakers")
    print(f"saved {SPLITS_FILE}")

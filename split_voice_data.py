"""split_voice_data.py | Picks voices from the datasets, and decides which are for training, validation and the final test.

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

split= keeps a pick inside one split of data/manifests/splits-v1.json, so test
speakers cannot reach a baseline by accident. Without it, every speaker can be picked.

    pick_even(30, split="validation")
    pick_crema_d(clips, 5, split="train")

The same seed gives the same voices. Making the splits is not written here yet: those
functions are empty for now, and build_splits.py writes the file.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import model
from build_manifest import MANIFEST, NO_SPEECH, SOURCES, VCTK_MIC
from Picky_voice import MAX_SECONDS, MIN_SECONDS, select_clips

SPLITS_FILE = MANIFEST.parent / "splits-v1.json"


def _speakers_in(split, dataset):
    """The speakers of one dataset in one split. The same id in another dataset is another person."""
    speakers = json.loads(SPLITS_FILE.read_text(encoding="utf-8"))["speakers"]
    if split not in speakers:
        raise ValueError(f"Unknown split {split!r}: use one of {list(speakers)}.")
    return set(speakers[split][dataset])


def _pick(clips, dataset, voices, min_seconds, max_seconds, seed, split):
    if split is not None:
        allowed = _speakers_in(split, dataset)
        clips = [clip for clip in clips if clip.speaker in allowed]

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


def speaker_key(clip):
    """The one name of a clip's speaker: dataset + id, because the same id can be two people in two datasets."""
    pass


def group_by_speaker(clips):
    """{speaker key: [clips]} for a list of clips from any of the readers."""
    pass


def split_speakers(clips):
    """{"train": [speaker keys], "validation": [...], "test": [...]}, every speaker in exactly one."""
    pass


def clips_in_split(clips, speakers):
    """The clips that belong to the given speaker keys."""
    pass


def check_no_leakage(splits):
    """Stops with an error if a speaker is in two splits, or in none."""
    pass


def save_splits(splits, path):
    """Writes the splits to a file, the same bytes for the same data."""
    pass


def load_splits(path):
    """Reads back what save_splits wrote."""
    pass

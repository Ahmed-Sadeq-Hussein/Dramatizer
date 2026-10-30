"""Picky_voice.py | Choose which clips of a dataset to use as reference voices.

One function, select_clips, decides. Four options switch its rules on and off,
because different jobs need different clips:

    select_clips(clips, min_seconds=3.0, max_seconds=25.0, duplicate_actor=True, emotions=False)

    min_seconds      shortest speech kept. WHY: a short reference gives a clone
                     that sounds less like the person. Measured on this
                     project's data (28 October 2026), cloning one sentence for
                     8 VCTK speakers, the clone's similarity to the speaker was:
                         1.0-1.8 s reference: mean 0.881, worst 0.760 (fails sim_full 0.80)
                         2.2-2.8 s reference: mean 0.892, worst 0.841
                         3.2 s and longer:    mean 0.922, worst 0.874
                     From 3 s every clone passed. Lower it only knowingly, e.g.
                     to 2.5 s to keep more CREMA-D actors.
    max_seconds      longest speech kept. WHY: CosyVoice refuses reference audio
                     longer than 30 s (an assert in cosyvoice/cli/frontend.py);
                     25 leaves a safe margin.
    duplicate_actor  True: every usable clip of every speaker (for browsing,
                     statistics, calibration). False: one clip per speaker, the
                     longest. WHY: a reference set or a test set needs each voice
                     once, or speakers with many clips dominate the results.
    emotions         False: only neutral clips (clips with no emotion label, like
                     VCTK, count as neutral). True: every emotion. WHY: a
                     reference spoken in anger or sadness pushes every clone toward
                     that emotion before the genome changes anything, so references
                     should be neutral; emotional clips are still wanted for
                     evaluation and calibration.

"Speech" means silence at the start and end is not counted. (The VCTK copy is
already trimmed, so its file length is used as it is.)

Use it from Python:

    from Voice_Importer import read_crema_d
    from Picky_voice import select_clips

    clips = read_crema_d("D:/Voices library/CREMA-D")
    references = select_clips(clips, duplicate_actor=False)        # one neutral clip per actor, 3-25 s
    everything = select_clips(clips, min_seconds=0, emotions=True)  # all clips, any length up to 25 s

or from the command line (the same four options):

    python Picky_voice.py crema-d "D:/Voices library/CREMA-D"
    python Picky_voice.py crema-d "D:/Voices library/CREMA-D" --min 2.5 --no-duplicate-actor
    python Picky_voice.py vctk "D:/Voices library/VCTK" --emotions --max 10
    python Picky_voice.py vctk "D:/Voices library/VCTK" --no-duplicate-actor --export "D:/Voices library/picked/VCTK"

--export writes FOLDER/<speaker>/reference.wav and reference.txt, the layout
Voice_model.py expects, using the longest selected clip of each speaker.

Measuring takes a while the first time (VCTK: every clip is read once), so
the lengths are saved in output/picky/ and reused.
"""
import argparse
import io
import json
import shutil
from collections import defaultdict
from pathlib import Path

import librosa
import pyarrow.parquet as pq
import soundfile as sf

import model
from Voice_Importer import get_clip, list_speakers, read_crema_d, read_vctk

# The defaults of the four options (see the top of this file for why)
MIN_SECONDS = 3.0
MAX_SECONDS = 25.0
DUPLICATE_ACTOR = True
EMOTIONS = False

# The emotion label that counts as neutral (CREMA-D: "neutral")
NEUTRAL = "neutral"

# Datasets whose clips have already had their leading and trailing silence removed
PRE_TRIMMED = {"VCTK"}

# How quiet counts as silence when trimming, in dB below the loudest part
TRIM_TOP_DB = 35

CACHE_DIR = model.PROJECT_DIR / "output" / "picky"


def _trimmed_seconds(samples, rate):
    trimmed, _ = librosa.effects.trim(samples, top_db=TRIM_TOP_DB)
    return len(trimmed) / rate


def measure(clips):
    """
    {clip_id: seconds of speech} for every clip, measured once and then read
    from the cache in output/picky/<dataset>_seconds.json.
    """
    if not clips:
        return {}

    dataset = clips[0].dataset
    cache_file = CACHE_DIR / f"{dataset}_seconds.json"
    cached = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}

    missing = [clip for clip in clips if clip.clip_id not in cached]

    if missing:
        print(f"Measuring {len(missing)} {dataset} clips (once; saved for next time)...")

        # VCTK clips live in parquet row groups: read each group's audio once
        by_group = defaultdict(list)
        for clip in missing:
            by_group[(clip.source, clip.row_group)].append(clip)

        measured = 0

        for (source, row_group), group in by_group.items():
            if row_group is None:
                for clip in group:
                    samples, rate = sf.read(str(clip.source), dtype="float32", always_2d=True)
                    cached[clip.clip_id] = round(_trimmed_seconds(samples.mean(axis=1), rate), 3)
            else:
                audio = pq.ParquetFile(source).read_row_group(row_group, columns=["audio"]).column("audio")
                for clip in group:
                    wav = io.BytesIO(audio[clip.row].as_py()["bytes"])
                    if dataset in PRE_TRIMMED:
                        cached[clip.clip_id] = round(sf.info(wav).duration, 3)
                    else:
                        samples, rate = sf.read(wav, dtype="float32", always_2d=True)
                        cached[clip.clip_id] = round(_trimmed_seconds(samples.mean(axis=1), rate), 3)

            # A progress line every 5000 clips
            if (measured + len(group)) // 5000 > measured // 5000:
                print(f"  {measured + len(group)}/{len(missing)} clips")
            measured += len(group)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cached), encoding="utf-8")

    return {clip.clip_id: cached[clip.clip_id] for clip in clips}


def is_neutral(clip):
    """True for a neutral clip, or one with no emotion label at all (VCTK)."""
    emotion = clip.details.get("emotion")
    return emotion is None or emotion == NEUTRAL


def select_clips(clips, min_seconds=MIN_SECONDS, max_seconds=MAX_SECONDS,
                 duplicate_actor=DUPLICATE_ACTOR, emotions=EMOTIONS):
    """
    The clips that pass every rule switched on (see the top of this file):

    min_seconds      shortest speech kept, in seconds
    max_seconds      longest speech kept, in seconds
    duplicate_actor  True: all usable clips per speaker. False: only each
                     speaker's longest usable clip
    emotions         True: any emotion. False: neutral clips only

    Returns the selected clips, longest first within each speaker, speakers in
    their original order.
    """
    if min_seconds > max_seconds:
        raise ValueError(f"min_seconds ({min_seconds}) is above max_seconds ({max_seconds}).")

    if not emotions:
        clips = [clip for clip in clips if is_neutral(clip)]

    seconds = measure(clips)

    by_speaker = defaultdict(list)
    for clip in clips:
        if min_seconds <= seconds[clip.clip_id] <= max_seconds:
            by_speaker[clip.speaker].append(clip)

    selected = []
    for found in by_speaker.values():
        found.sort(key=lambda c: seconds[c.clip_id], reverse=True)
        selected.extend(found if duplicate_actor else found[:1])

    return selected


def export_references(selected, folder):
    """
    Write folder/<speaker>/reference.wav and reference.txt, one per speaker:
    the longest of that speaker's selected clips.
    """
    folder = Path(folder)
    seconds = measure(selected)
    longest = {}

    for clip in selected:
        if clip.speaker not in longest or seconds[clip.clip_id] > seconds[longest[clip.speaker].clip_id]:
            longest[clip.speaker] = clip

    for speaker, clip in longest.items():
        path, text = get_clip(clip)
        target = folder / speaker
        target.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target / "reference.wav")
        (target / "reference.txt").write_text(text, encoding="utf-8")

    return folder


if __name__ == "__main__":
    readers = {"crema-d": read_crema_d, "vctk": read_vctk}

    parser = argparse.ArgumentParser(description="Choose which clips of a dataset to use as reference voices.")
    parser.add_argument("dataset", choices=readers)
    parser.add_argument("folder", help="the dataset's folder")
    parser.add_argument("--min", type=float, default=MIN_SECONDS,
                        help=f"shortest speech in seconds (default {MIN_SECONDS})")
    parser.add_argument("--max", type=float, default=MAX_SECONDS,
                        help=f"longest speech in seconds (default {MAX_SECONDS})")
    parser.add_argument("--duplicate-actor", action=argparse.BooleanOptionalAction, default=DUPLICATE_ACTOR,
                        help="keep every usable clip of a speaker; --no-duplicate-actor keeps only the longest")
    parser.add_argument("--emotions", action=argparse.BooleanOptionalAction, default=EMOTIONS,
                        help="keep every emotion; --no-emotions (the default) keeps only neutral clips")
    parser.add_argument("--export", default=None, metavar="FOLDER",
                        help="write each speaker's longest selected clip as FOLDER/<speaker>/reference.wav + .txt")
    args = parser.parse_args()

    clips = readers[args.dataset](args.folder)
    selected = select_clips(clips, args.min, args.max, args.duplicate_actor, args.emotions)

    speakers = list_speakers(clips)
    kept_speakers = {clip.speaker for clip in selected}

    print(f"\n{clips[0].dataset} with min {args.min:g} s, max {args.max:g} s, "
          f"duplicate actor {'on' if args.duplicate_actor else 'off'}, "
          f"emotions {'on' if args.emotions else 'off (neutral only)'}:")
    print(f"  {len(selected)} of {len(clips)} clips selected")
    print(f"  {len(kept_speakers)} of {len(speakers)} speakers have at least one")

    left_out = sorted(set(speakers) - kept_speakers)
    if left_out:
        print(f"  left out: {', '.join(left_out)}")

    if args.export:
        print(f"Exported to {export_references(selected, args.export)}")

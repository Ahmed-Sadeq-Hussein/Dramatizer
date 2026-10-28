"""Voice_Importer.py | Browse the voice datasets: list the clips, hear one, read its script.

Two datasets from the voice library are supported (see "Voice library.txt"):

    CREMA-D   91 actors, 12 sentences in 6 emotions, one wav file per clip
    VCTK      109 speakers with many accents, stored in parquet files

Each is read into the same shape, a list of Clip objects, so everything after
reading works the same for both.

Use it from Python:

    from Voice_Importer import read_crema_d, find_clips, show_clips, get_clip, play_clip

    clips = read_crema_d("D:/Voices library/CREMA-D")
    angry = find_clips(clips, speaker="1001", emotion="anger")
    show_clips(angry)
    audio_path, script = play_clip(angry[0])     # plays it, returns wav + text

or browse interactively, from the command line:

    python Voice_Importer.py crema-d "D:/Voices library/CREMA-D"
    python Voice_Importer.py vctk "D:/Voices library/VCTK"

VCTK clips live inside parquet files, so get_clip writes the chosen clip to a
wav file (with its script next to it as a .txt), in IMPORT_DIR from .env, or
imported_voices/ in this project.
"""
import argparse
import io
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow.parquet as pq
from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_DIR / ".env")

IMPORT_DIR = PROJECT_DIR / os.environ.get("IMPORT_DIR", "imported_voices")

# CREMA-D names its files ACTOR_SENTENCE_EMOTION_INTENSITY.wav, with codes
# for the last three. These are the codes from the dataset's documentation.
CREMA_D_SENTENCES = {
    "IEO": "It's eleven o'clock.",
    "TIE": "That is exactly what happened.",
    "IOM": "I'm on my way to the meeting.",
    "IWW": "I wonder what this is about.",
    "TAI": "The airplane is almost full.",
    "MTI": "Maybe tomorrow it will be cold.",
    "IWL": "I would like a new alarm clock.",
    "ITH": "I think I have a doctor's appointment.",
    "DFA": "Don't forget a jacket.",
    "ITS": "I think I've seen this before.",
    "TSI": "The surface is slick.",
    "WSI": "We'll stop in a couple of minutes.",
}

CREMA_D_EMOTIONS = {
    "NEU": "neutral",
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "anger",
    "FEA": "fear",
    "DIS": "disgust",
}

CREMA_D_INTENSITIES = {
    "LO": "low",
    "MD": "medium",
    "HI": "high",
    "XX": "unspecified",
    "X": "unspecified",  # one file in the dataset uses a single X
}


@dataclass
class Clip:
    """One recording from a dataset, whichever dataset it came from."""

    dataset: str            # "CREMA-D" or "VCTK"
    clip_id: str            # unique within the dataset, e.g. "1001_DFA_ANG_XX"
    speaker: str            # e.g. "1001" or "p247"
    text: str               # the script: the exact words spoken
    source: Path            # the wav file, or the parquet file that holds it
    details: dict = field(default_factory=dict)   # emotion, intensity, mic, ...
    row_group: int | None = None    # where in the parquet file (VCTK only)
    row: int | None = None


# --------------------------------------------------------------------------- #
# Reading a dataset folder
# --------------------------------------------------------------------------- #


def read_crema_d(folder):
    """
    Every clip in a CREMA-D folder. The wav files may sit in any sub-folder.
    Returns a list of Clip, sorted by speaker and clip id.
    """
    folder = Path(folder)
    wavs = sorted(folder.rglob("*.wav"))

    if not wavs:
        raise FileNotFoundError(f"No wav files found in {folder}. Is this the CREMA-D folder?")

    clips = []

    for wav in wavs:
        parts = wav.stem.split("_")

        # A file that does not follow the naming scheme is not a CREMA-D clip
        if len(parts) != 4 or parts[1] not in CREMA_D_SENTENCES:
            continue

        actor, sentence, emotion, intensity = parts

        clips.append(Clip(
            dataset="CREMA-D",
            clip_id=wav.stem,
            speaker=actor,
            text=CREMA_D_SENTENCES[sentence],
            source=wav,
            details={
                "emotion": CREMA_D_EMOTIONS.get(emotion, emotion),
                "intensity": CREMA_D_INTENSITIES.get(intensity, intensity),
            },
        ))

    return sorted(clips, key=lambda c: (c.speaker, c.clip_id))


def read_vctk(folder):
    """
    Every clip in a VCTK folder of parquet files (the Hugging Face copy).
    Only the text columns are read here, which takes a few seconds; the audio
    is read later, one clip at a time, by get_clip.
    Returns a list of Clip, sorted by speaker and clip id.
    """
    folder = Path(folder)
    files = sorted(folder.rglob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder}. Is this the VCTK folder?")

    clips = []

    for file in files:
        parquet = pq.ParquetFile(file)

        for row_group in range(parquet.num_row_groups):
            table = parquet.read_row_group(row_group, columns=["id", "speaker_id", "mic_id", "text"])

            for row, record in enumerate(table.to_pylist()):
                clips.append(Clip(
                    dataset="VCTK",
                    clip_id=record["id"],
                    speaker=record["speaker_id"],
                    text=record["text"],
                    source=file,
                    details={"mic": record["mic_id"]},
                    row_group=row_group,
                    row=row,
                ))

    return sorted(clips, key=lambda c: (c.speaker, c.clip_id))


# --------------------------------------------------------------------------- #
# Viewing, picking and listening
# --------------------------------------------------------------------------- #


def list_speakers(clips):
    """How many clips each speaker has, as {speaker: count}, sorted by speaker."""
    return dict(sorted(Counter(c.speaker for c in clips).items()))


def find_clips(clips, speaker=None, text=None, **details):
    """
    The clips that match every filter given:

    speaker   exact speaker id, e.g. "1001" or "p247"
    text      words that must appear in the script (any case)
    details   e.g. emotion="anger", intensity="high", mic="mic1"
    """
    found = []

    for clip in clips:
        if speaker is not None and clip.speaker != speaker:
            continue
        if text is not None and text.lower() not in clip.text.lower():
            continue
        if any(str(clip.details.get(key)) != str(value) for key, value in details.items()):
            continue
        found.append(clip)

    return found


def show_clips(clips, start=0, count=20):
    """Print a numbered table of clips, from position start, count of them."""
    shown = clips[start:start + count]

    if not shown:
        print("No clips to show.")
        return

    for number, clip in enumerate(shown, start=start + 1):
        details = ", ".join(str(v) for v in clip.details.values())
        print(f"{number:>6}  {clip.speaker:<6} {clip.clip_id:<20} [{details}]  {clip.text}")

    print(f"        showing {start + 1}-{start + len(shown)} of {len(clips)}")


def get_clip(clip):
    """
    The clip as a wav file and its script: returns (audio path, text).

    CREMA-D clips are already wav files. VCTK clips are written out of their
    parquet file into IMPORT_DIR/VCTK/<speaker>/<clip id>.wav, once, with the
    script beside it as <clip id>.txt, ready to use as a reference voice.
    """
    if clip.dataset != "VCTK":
        return clip.source, clip.text

    wav = IMPORT_DIR / "VCTK" / clip.speaker / f"{clip.clip_id}.wav"

    if not wav.exists():
        table = pq.ParquetFile(clip.source).read_row_group(clip.row_group, columns=["audio"])
        audio = table.column("audio")[clip.row].as_py()

        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(audio["bytes"])
        wav.with_suffix(".txt").write_text(clip.text, encoding="utf-8")

    return wav, clip.text


def play_clip(clip):
    """Play the clip, print its script, and return (audio path, text)."""
    path, text = get_clip(clip)

    print(f"\n{clip.dataset} {clip.clip_id} (speaker {clip.speaker})")
    print(f"Script: {text}")
    print(f"Audio:  {path}")

    if sys.platform == "win32":
        import winsound

        # Waits until the clip has finished playing
        winsound.PlaySound(str(path), winsound.SND_FILENAME)
    else:
        print("(Playback is built in on Windows only: open the audio file to listen.)")

    return path, text


def browse(clips):
    """
    Pick a clip interactively. Returns (audio path, text) of the clip you
    choose with "use", or None if you quit.
    """
    help_text = """
Commands:
  NUMBER        play that clip and show its script
  r             replay the last clip
  use           choose the last clip played, and stop browsing
  n / p         next / previous page
  s SPEAKER     only this speaker (s alone: list the speakers)
  t WORDS       only clips whose script contains these words
  d KEY VALUE   only clips with this detail, e.g. "d emotion anger"
                (filters add up: each one narrows the clips shown now)
  a             all clips again (remove every filter)
  h             this help
  q             quit without choosing
"""
    dataset = clips[0].dataset if clips else "?"
    print(f"\n{dataset}: {len(clips)} clips from {len(list_speakers(clips))} speakers")
    print(help_text)

    shown, start, last = clips, 0, None
    show_clips(shown, start)

    while True:
        command = input("\n> ").strip()
        word, _, rest = command.partition(" ")
        word = word.lower()

        if word == "q":
            return None

        elif word == "h":
            print(help_text)

        elif word.isdigit():
            number = int(word)
            if 1 <= number <= len(shown):
                last = shown[number - 1]
                play_clip(last)
            else:
                print(f"Pick a number from 1 to {len(shown)}.")

        elif word == "r":
            if last is None:
                print("Nothing played yet.")
            else:
                play_clip(last)

        elif word == "use":
            if last is None:
                print("Play a clip first, then type use.")
            else:
                return get_clip(last)

        elif word in ("n", "p"):
            start = start + 20 if word == "n" else start - 20
            start = max(0, min(start, max(0, len(shown) - 1) // 20 * 20))
            show_clips(shown, start)

        elif word == "s" and not rest:
            speakers = list_speakers(clips)
            print("  ".join(f"{s} ({n})" for s, n in speakers.items()))

        elif word in ("s", "t", "d", "a"):
            # Each filter narrows what is shown now; "a" starts over
            if word == "s":
                shown = find_clips(shown, speaker=rest)
            elif word == "t":
                shown = find_clips(shown, text=rest)
            elif word == "d":
                key, _, value = rest.partition(" ")
                shown = find_clips(shown, **{key: value})
            else:
                shown = clips

            start = 0
            show_clips(shown, start)

        else:
            print("Unknown command. Type h for help.")


if __name__ == "__main__":
    readers = {"crema-d": read_crema_d, "vctk": read_vctk}

    parser = argparse.ArgumentParser(description="Browse a voice dataset: list clips, hear one, read its script.")
    parser.add_argument("dataset", choices=readers, help="which dataset the folder holds")
    parser.add_argument("folder", help="the dataset's folder, e.g. \"D:/Voices library/VCTK\"")
    args = parser.parse_args()

    print(f"Reading {args.folder}...")
    chosen = browse(readers[args.dataset](args.folder))

    if chosen is not None:
        path, text = chosen
        print(f"\nChosen clip:\n  audio:  {path}\n  script: {text}")

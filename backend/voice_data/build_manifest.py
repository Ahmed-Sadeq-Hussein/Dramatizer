"""build_manifest.py | The data manifest: which clips this project uses, where they came from, and their terms.

Writes data/manifests/voices-v1.json: the sources and their licences, what each
field means, and one line per clip. The same data gives the same file, byte for
byte (there is no date in it), so its hash identifies the data version.

    python -m backend.voice_data.build_manifest            # reuses the checksums already computed
    python -m backend.voice_data.build_manifest --fresh    # reads every clip again

Which clips: 3-25 s of speech (why: Picky_voice.py), every emotion, every clip
of every speaker. From VCTK only one microphone, mic1: mic2 is the same reading
of the same sentence, and two copies would count as two clips. Two VCTK clips
are left out because they hold no speech (NO_SPEECH below).

Checksums are saved in output/picky/ every 1000 clips, so a stopped run carries
on where it stopped. VCTK's checksum is of the wav inside the parquet file.
"""
import argparse
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf

from backend import model
from backend.voice_data.Picky_voice import CACHE_DIR, measure, select_clips
from backend.voice_data.Voice_Importer import read_crema_d, read_libritts_r, read_vctk

MANIFEST = model.DATA_DIR / "manifests" / "voices-v1.json"
MIN_SECONDS = 3.0
MAX_SECONDS = 25.0
VCTK_MIC = "mic1"

# Almost silent files: no voice activity found and Whisper hears nothing. Scanned
# on 3 Nov 2026. The speech measure cannot see this, because it is relative to the
# loudest part of the same file, so a silent file counts as all speech.
NO_SPEECH = {"p341_101_mic1", "p345_292_mic1"}

SOURCES = {
    "VCTK": {
        "reader": read_vctk,
        "folder": ("VCTK_FOLDER", "VCTK"),
        "version": "0.92, the copy with leading and trailing silence trimmed",
        "downloaded_from": "https://huggingface.co/datasets/jspaulsen/vctk",
        "original": "https://datashare.ed.ac.uk/handle/10283/3443",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "terms": "Free for any use, including commercial. Credit the dataset.",
        "labels": "None. Speaker p315 has no transcripts and is not in this copy.",
    },
    "CREMA-D": {
        "reader": read_crema_d,
        "folder": ("CREMA_D_FOLDER", "CREMA-D"),
        "version": "the Hugging Face copy",
        "downloaded_from": "https://huggingface.co/datasets/myleslinder/crema-d",
        "original": "https://github.com/CheyneyComputerScience/CREMA-D",
        "license": "ODbL 1.0",
        "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
        "terms": "Free for any use, including commercial. Credit the dataset, and share changed versions "
                 "of the dataset itself under the same licence.",
        "labels": "emotion and intensity are what the actor was asked to perform, not what listeners hear.",
    },
    "LibriTTS-R": {
        "reader": read_libritts_r,
        "folder": ("LIBRITTS_R_FOLDER", "LibriTTS-R"),
        "version": "dev-clean and test-clean only (OpenSLR 141, checksums verified on download)",
        "downloaded_from": "https://www.openslr.org/141/",
        "original": "https://www.openslr.org/141/",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "terms": "Free for any use, including commercial. Credit the dataset.",
        "labels": "None. Audiobook readings restored by the Miipher model, so the sound is processed.",
    },
}

FIELDS = {
    "clip_id": "unique within the dataset",
    "dataset": "a key of sources",
    "speaker": "speaker id, unique within the dataset; no speaker is in two datasets",
    "transcript": "the words spoken",
    "speech_seconds": "length without the silence at the start and end",
    "sample_rate": "samples per second of the audio file",
    "emotion": "CREMA-D only, otherwise null: the emotion the actor was asked to perform",
    "intensity": "CREMA-D only, otherwise null: the intensity the actor was asked to perform",
    "mic": "VCTK only, otherwise null",
    "subset": "LibriTTS-R only, otherwise null: the part of the corpus",
    "path": "the audio, relative to the dataset's folder (VCTK: the parquet file that holds it)",
    "row_group": "VCTK only, otherwise null: where in the parquet file",
    "row": "VCTK only, otherwise null: where in the row group",
    "sha256": "checksum of the wav file's bytes",
}


def _describe(wav_bytes):
    """(sha256, sample rate) of one wav file's bytes."""
    return hashlib.sha256(wav_bytes).hexdigest(), sf.info(io.BytesIO(wav_bytes)).samplerate


def checksums(clips, fresh=False):
    """{clip_id: [sha256, sample rate]}, saved as it goes in output/picky/<dataset>_sha256.json."""
    cache_file = CACHE_DIR / f"{clips[0].dataset}_sha256.json"
    known = {} if fresh or not cache_file.exists() else json.loads(cache_file.read_text(encoding="utf-8"))
    missing = [clip for clip in clips if clip.clip_id not in known]

    if not missing:
        return known

    print(f"Checksums for {len(missing)} {clips[0].dataset} clips...", flush=True)

    def save():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        temporary = cache_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(known), encoding="utf-8")
        temporary.replace(cache_file)

    # VCTK clips live in parquet row groups: read each group's audio once
    by_group = defaultdict(list)
    for clip in missing:
        by_group[(clip.source, clip.row_group)].append(clip)

    done = 0
    for (source, row_group), group in by_group.items():
        audio = None if row_group is None else pq.ParquetFile(source).read_row_group(row_group, columns=["audio"]).column("audio")

        for clip in group:
            data = clip.source.read_bytes() if audio is None else audio[clip.row].as_py()["bytes"]
            known[clip.clip_id] = _describe(data)
            done += 1
            if done % 1000 == 0:
                save()
                print(f"  {done}/{len(missing)} clips", flush=True)

    save()
    return known


def clip_record(clip, folder, seconds, sums):
    sha256, sample_rate = sums[clip.clip_id]
    return {
        "clip_id": clip.clip_id,
        "dataset": clip.dataset,
        "speaker": clip.speaker,
        "transcript": clip.text,
        "speech_seconds": seconds[clip.clip_id],
        "sample_rate": sample_rate,
        "emotion": clip.details.get("emotion"),
        "intensity": clip.details.get("intensity"),
        "mic": clip.details.get("mic"),
        "subset": clip.details.get("split"),
        "path": clip.source.relative_to(folder).as_posix(),
        "row_group": clip.row_group,
        "row": clip.row,
        "sha256": sha256,
    }


def build(fresh=False):
    library = Path(model.setting("VOICE_LIBRARY_DIR"))
    records, sources = [], {}

    for name, source in SOURCES.items():
        setting, default = source["folder"]
        folder = library / model.setting(setting, default)
        print(f"Reading {name}...", flush=True)
        clips = source["reader"](folder)
        if name == "VCTK":
            clips = [clip for clip in clips if clip.details["mic"] == VCTK_MIC and clip.clip_id not in NO_SPEECH]

        selected = select_clips(clips, MIN_SECONDS, MAX_SECONDS, duplicate_actor=True, emotions=True)
        seconds = measure(selected)
        sums = checksums(selected, fresh)
        chosen = [clip_record(clip, folder, seconds, sums) for clip in selected]
        records += sorted(chosen, key=lambda r: (r["speaker"], r["clip_id"]))

        sources[name] = {
            **{key: value for key, value in source.items() if key not in ("reader", "folder")},
            "speakers": len({r["speaker"] for r in chosen}),
            "clips": len(chosen),
            "hours_of_speech": round(sum(r["speech_seconds"] for r in chosen) / 3600, 2),
        }

    lines = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records]
    header = {
        "manifest": MANIFEST.stem,
        "selection": {"min_speech_seconds": MIN_SECONDS, "max_speech_seconds": MAX_SECONDS,
                      "emotions": "all", "clips_per_speaker": "all", "vctk_mic": VCTK_MIC,
                      "left_out_no_speech": sorted(NO_SPEECH)},
        "clips_sha256": hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(),
        "sources": sources,
        "fields": FIELDS,
    }

    # One clip per line, so a change to the data shows up as a readable diff
    text = json.dumps(header, indent=2, ensure_ascii=False)[:-2] + ',\n  "clips": [\n    ' + ",\n    ".join(lines) + "\n  ]\n}\n"
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(text, encoding="utf-8", newline="\n")
    return header


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write the data manifest.")
    parser.add_argument("--fresh", action="store_true", help="read every clip again instead of reusing saved checksums")
    args = parser.parse_args()

    header = build(args.fresh)

    print(f"\nWrote {MANIFEST} ({MANIFEST.stat().st_size / 1e6:.1f} MB)")
    print(f"clips_sha256 {header['clips_sha256']}")
    for name, source in header["sources"].items():
        print(f"  {name:11s} {source['speakers']:4d} speakers  {source['clips']:6d} clips  "
              f"{source['hours_of_speech']:6.2f} h  {source['license']}")

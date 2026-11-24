"""voice_pack.py | The serving voice pack: the 176 validation voices' chosen clips, about 150 MB instead of 24 GB.

The site only ever offers the validation voices, one clip each (the voice manifest, voice-split-v2.json). The pack
holds exactly those clips, so a computer or container without the full voice library can still serve them:

    <dataset>/<speaker>/reference.wav     the clip (the original bytes when it is already a 16-bit wav)
    <dataset>/<speaker>/reference.txt     its script
    pack.json                             every clip: id, dataset, speaker, seconds, sha256, details; the manifest's
                                          sha256; the date; the licences
    README.md                             the credits the licences ask for

The 44 test voices are never in it: they stay locked until the final evaluation.

    python main_backend.py voice-pack build                 # from the full library, into VOICE_PACK_STORE/build
    python main_backend.py voice-pack check [FOLDER]        # every file against pack.json
    python main_backend.py voice-pack upload [FOLDER]       # to the PRIVATE dataset VOICE_PACK_REPO (one commit)
    python main_backend.py voice-pack pull REVISION         # that commit into VOICE_PACK_STORE/<revision>, checked

The commit's hash is the pack's revision. Set VOICE_PACK_DIR to a pulled pack and the site reads it instead of
the library (server.library); saved runs find their reference recording in it by clip id (reference_of).
"""
import argparse
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import soundfile as sf

from backend import model

PACK_FILE = "pack.json"
SPLIT = "validation"
DEFAULT_REPO = "Dramatizer/Serving_Voices"
DEFAULT_STORE = "D:/Dramatizer voice pack"


def store():
    return Path(model.setting("VOICE_PACK_STORE", DEFAULT_STORE))


def repo():
    return model.setting("VOICE_PACK_REPO", DEFAULT_REPO)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _file_sha(path):
    return _sha(Path(path).read_bytes())


class PackError(Exception):
    """The pack is missing, incomplete or changed."""


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


def _wav_bytes(audio_path):
    """The clip as 16-bit wav bytes: the file itself when it already is one (so its sha256 still matches the
    manifest), otherwise converted."""
    data = Path(audio_path).read_bytes()
    info = sf.info(io.BytesIO(data))
    if info.format == "WAV" and info.subtype == "PCM_16":
        return data, info
    samples, rate = sf.read(io.BytesIO(data), dtype="float32")
    buffer = io.BytesIO()
    sf.write(buffer, samples, rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue(), sf.info(io.BytesIO(buffer.getvalue()))


def build(out_dir, clips=None, manifest_path=None, now=None):
    """Write the pack of the validation voices into out_dir (emptied of an older pack first). clips: the library's
    clips (default: read the whole library). Returns pack.json's content."""
    from backend.voice_data import split_voice_data
    from backend.voice_data.build_manifest import SOURCES
    from backend.voice_data.Voice_Importer import get_clip

    manifest_path = Path(manifest_path or split_voice_data.SPLITS_FILE)
    manifest = split_voice_data.load_voice_manifest(manifest_path)
    voices = [voice for voice in manifest["voices"] if voice["split"] == SPLIT]
    locked = set(manifest["test"])
    if any(voice["voice"] in locked for voice in voices):
        raise PackError("A test voice is marked as validation in the manifest: not building.")
    wanted = {(voice["dataset"], voice["clip_id"]): voice for voice in voices}

    clips = split_voice_data.read_library() if clips is None else clips
    found = {(clip.dataset, clip.clip_id): clip for clip in clips if (clip.dataset, clip.clip_id) in wanted}
    missing = sorted(f"{d}/{c}" for d, c in set(wanted) - set(found))
    if missing:
        raise PackError(f"{len(missing)} validation clips are not in the library, e.g. {missing[:3]}.")

    out_dir = Path(out_dir)
    if (out_dir / PACK_FILE).exists():
        for old in out_dir.glob("*/*/reference.*"):
            old.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for key in sorted(wanted):
        voice, clip = wanted[key], found[key]
        if f"{clip.dataset}/{clip.speaker}" in locked:
            raise PackError(f"{clip.dataset}/{clip.speaker} is a test voice: not building.")
        audio, text = get_clip(clip)
        data, info = _wav_bytes(audio)
        folder = out_dir / clip.dataset / clip.speaker
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "reference.wav").write_bytes(data)
        (folder / "reference.txt").write_text(text, encoding="utf-8")
        entries.append({
            "voice": voice["voice"], "dataset": clip.dataset, "speaker": clip.speaker, "clip_id": clip.clip_id,
            "audio": f"{clip.dataset}/{clip.speaker}/reference.wav",
            "text": f"{clip.dataset}/{clip.speaker}/reference.txt",
            "seconds": round(info.frames / info.samplerate, 3), "sample_rate": info.samplerate,
            "sha256": _sha(data), "text_sha256": _sha(text.encode("utf-8")),
            "same_bytes_as_source": _sha(data) == voice.get("sha256"),
            "details": {k: str(v) for k, v in clip.details.items()},
        })

    datasets = sorted({entry["dataset"] for entry in entries})
    pack = {
        "name": "Dramatizer serving voices",
        "what": "The validation voices' chosen clips, one per voice. The test voices are not in it.",
        "made": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifest": manifest["name"],
        "manifest_sha256": _file_sha(manifest_path),
        "split": SPLIT,
        "count": len(entries),
        "seconds": round(sum(entry["seconds"] for entry in entries), 1),
        "licences": {name: {key: SOURCES[name][key] for key in ("license", "license_url", "original", "terms")}
                     for name in datasets},
        "clips": entries,
    }
    (out_dir / PACK_FILE).write_text(json.dumps(pack, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(readme(pack), encoding="utf-8")
    return pack


def readme(pack):
    lines = [f"# {pack['name']}", "", pack["what"], "",
             f"{pack['count']} voices, {pack['seconds'] / 60:.0f} minutes, from the voice manifest "
             f"{pack['manifest']} (sha256 {pack['manifest_sha256'][:12]}...). Made {pack['made']}.", "",
             "These are recordings of real people. Keep this dataset private.", "", "## Credits and licences", ""]
    for name, licence in pack["licences"].items():
        lines.append(f"- **{name}**: {licence['license']} ({licence['license_url']}), from {licence['original']}. "
                     f"{licence['terms']}")
    lines += ["", "The CREMA-D clips are a changed extract of the CREMA-D database and stay under ODbL 1.0.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Reading and checking
# --------------------------------------------------------------------------- #


def load(folder):
    """pack.json of the pack in folder."""
    path = Path(folder) / PACK_FILE
    if not path.is_file():
        raise PackError(f"No voice pack in {folder} ({PACK_FILE} is missing).")
    return json.loads(path.read_text(encoding="utf-8"))


def check(folder):
    """Every file of the pack against pack.json. Returns a list of problems (empty: the pack is whole)."""
    folder = Path(folder)
    try:
        pack = load(folder)
    except (PackError, ValueError) as error:
        return [str(error)]
    problems = []
    if pack.get("split") != SPLIT:
        problems.append(f"The pack holds the {pack.get('split')!r} split, not {SPLIT!r}.")
    if pack.get("count") != len(pack.get("clips", [])):
        problems.append(f"pack.json says {pack.get('count')} clips but lists {len(pack.get('clips', []))}.")
    for entry in pack.get("clips", []):
        for name, sha in ((entry["audio"], entry["sha256"]), (entry["text"], entry["text_sha256"])):
            path = folder / name
            if not path.is_file():
                problems.append(f"{name} is missing.")
            elif _file_sha(path) != sha:
                problems.append(f"{name} has changed (sha256 differs).")
    return problems


def read(folder):
    """The pack's clips as Voice_Importer Clips, their source the pack's own wav, so get_clip uses it as it is."""
    from backend.voice_data.Voice_Importer import Clip

    folder = Path(folder)
    clips = []
    for entry in load(folder)["clips"]:
        clips.append(Clip(dataset=entry["dataset"], clip_id=entry["clip_id"], speaker=entry["speaker"],
                          text=(folder / entry["text"]).read_text(encoding="utf-8").strip(),
                          source=folder / entry["audio"], details=dict(entry.get("details", {}))))
    return sorted(clips, key=lambda c: (c.dataset, c.speaker, c.clip_id))


def folder_in_use():
    """VOICE_PACK_DIR, or None when the site reads the full library."""
    value = model.setting("VOICE_PACK_DIR", "")
    return Path(value) if value else None


def reference_of(run):
    """A saved run's reference recording. With a pack in use it is found by clip id (the stored path is from the
    computer that made the run); without one, or when the pack does not have that clip, the stored path."""
    folder = folder_in_use()
    origin = run.get("origin") or {}
    if folder and origin.get("dataset") and origin.get("clip_id"):
        try:
            entries = load(folder)["clips"]
        except (PackError, OSError, ValueError):
            entries = []
        for entry in entries:
            if entry["dataset"] == origin["dataset"] and entry["clip_id"] == origin["clip_id"]:
                return str(folder / entry["audio"])
    return run.get("reference_audio")


# --------------------------------------------------------------------------- #
# Hugging Face
# --------------------------------------------------------------------------- #


def set_revision(folder, revision):
    """Record the revision (the Hugging Face commit) in the local pack.json; release_info reads it from there.
    pack.json is not in its own checksums, so this does not break check()."""
    path = Path(folder) / PACK_FILE
    pack = json.loads(path.read_text(encoding="utf-8"))
    pack["revision"] = revision
    path.write_text(json.dumps(pack, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def upload(folder, repo_id, token=None):
    """Upload the pack to a PRIVATE Hugging Face dataset as one commit; refuses a public or missing one, or a pack
    that fails check(). Returns the commit hash: the revision to pull."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

    problems = check(folder)
    if problems:
        raise PackError("Not uploaded, the pack is not whole: " + "; ".join(problems[:5]))
    api = HfApi(token=token)
    try:
        info = api.repo_info(repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        raise PackError(f"No dataset {repo_id} that this token can see. Create it as Private first "
                        f"(huggingface.co/new-dataset), and log in with a write token.") from None
    except HfHubHTTPError as error:
        raise PackError(f"Hugging Face refused: {error}") from None
    if not info.private:
        raise PackError(f"Not uploaded: {repo_id} is PUBLIC. These are recordings of real people; make it private "
                        f"(its Settings > Change visibility) and run this again.")
    pack = load(folder)
    commit = api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type="dataset",
                               commit_message=f"Voice pack: {pack['count']} validation voices, manifest "
                                              f"{pack['manifest_sha256'][:12]}, made {pack['made']}",
                               delete_patterns=["*/*/reference.*"])
    revision = commit.oid
    set_revision(folder, revision)
    return revision


def pull(revision, repo_id, dest=None, token=None):
    """Download one revision of the pack into dest (default VOICE_PACK_STORE/<revision>) and check every file.
    Returns the folder."""
    from huggingface_hub import snapshot_download

    if not revision or len(revision) < 7:
        raise PackError("Pull a pack by its revision: the commit hash that `voice-pack upload` printed.")
    dest = Path(dest or store() / revision)
    snapshot_download(repo_id, repo_type="dataset", revision=revision, local_dir=str(dest), token=token)
    problems = check(dest)
    if problems:
        raise PackError(f"The pack in {dest} does not match its pack.json: " + "; ".join(problems[:5]))
    set_revision(dest, revision)
    return dest


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def main(argv=None):
    parser = argparse.ArgumentParser(prog="voice-pack", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="action", required=True)
    one = sub.add_parser("build", help="make the pack from the full voice library")
    one.add_argument("--out", help="the folder (default VOICE_PACK_STORE/build)")
    one = sub.add_parser("check", help="check every file against pack.json")
    one.add_argument("folder", nargs="?", help="default VOICE_PACK_DIR, else VOICE_PACK_STORE/build")
    one = sub.add_parser("upload", help="upload to the private dataset (VOICE_PACK_REPO)")
    one.add_argument("folder", nargs="?", help="default VOICE_PACK_STORE/build")
    one.add_argument("--hf", help=f"OWNER/NAME (default {DEFAULT_REPO}, or VOICE_PACK_REPO)")
    one = sub.add_parser("pull", help="download one revision and check it")
    one.add_argument("revision")
    one.add_argument("--hf", help=f"OWNER/NAME (default {DEFAULT_REPO}, or VOICE_PACK_REPO)")
    one.add_argument("--to", help="the folder (default VOICE_PACK_STORE/<revision>)")
    args = parser.parse_args(argv)

    try:
        if args.action == "build":
            out = Path(args.out) if args.out else store() / "build"
            pack = build(out)
            size = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
            same = sum(entry["same_bytes_as_source"] for entry in pack["clips"])
            print(f"Built {pack['count']} voices ({pack['seconds'] / 60:.0f} min of audio, {size / 1e6:.0f} MB) in "
                  f"{out}. {same} of them are byte for byte the clip the voice manifest checksummed.")
            print(f"Try it: VOICE_PACK_DIR={out}. Upload it: python main_backend.py voice-pack upload")
        elif args.action == "check":
            folder = Path(args.folder) if args.folder else (folder_in_use() or store() / "build")
            problems = check(folder)
            for problem in problems[:20]:
                print("  " + problem)
            print(f"{folder}: " + (f"{len(problems)} problems" if problems else
                                   f"whole ({load(folder)['count']} voices, revision "
                                   f"{load(folder).get('revision') or 'not uploaded'})"))
            return 1 if problems else 0
        elif args.action == "upload":
            folder = Path(args.folder) if args.folder else store() / "build"
            revision = upload(folder, args.hf or repo())
            print(f"Uploaded. Revision {revision}\nPull it elsewhere: python main_backend.py voice-pack pull {revision}")
        elif args.action == "pull":
            folder = pull(args.revision, args.hf or repo(), args.to)
            print(f"Pulled and checked: {folder}\nUse it: VOICE_PACK_DIR={folder}")
    except PackError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""uploaded_voices.py | Voices people upload on the website: each one usable only by the account that uploaded it.

An account uploads a short recording of a voice and the words spoken in it. The voice then shows
under Dataset: Custom voices for that account only (and for admins, who run the service); no other
account can choose it, hear it or hear the renders made with it (server.py). Uploaded voices are never
used for genome searches, ratings, training or evaluation (agreement.py, section 2). The owner can
delete one at any time, and it is then deleted from this computer.

Where they are kept (outside the repository, like every voice):

    <UPLOADED_VOICES_DIR, default VOICES_DIR/uploaded>/<username>/<voice folder>/
        reference.wav       the recording, decoded and written again as 16-bit wav (never the uploaded bytes)
        reference.txt       the words spoken in it
        about.json          owner, name, when, length, and the consent the uploader gave

    from backend.voice_data import uploaded_voices as uv
    uv.save("ana", "Calm", "The words I said.", audio_bytes, "rec.mp3")   # -> the voice's about
    uv.clips("ana")             # [Clip], dataset "Custom voices"
    uv.remove("ana", "calm")
    uv.rename("ana", "calm", "Warm")    # a new name to show; the voice's ID ("ana Calm") never changes

A voice has an ID and a name. The ID ("speaker" in about.json: the account and the name it was uploaded
under) is fixed: renders made with it, who may hear them, and its folder all go by it. The name is only what
the page shows, and its owner may change it.
"""
import io
import json
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.voice_data.Voice_Importer import Clip

DATASET = "Custom voices"
MAX_BYTES = 8 * 1024 * 1024            # the uploaded file
MIN_SECONDS, MAX_SECONDS = 3.0, 30.0   # a reference voice needs a few seconds; more does not help the cloning
MAX_PER_ACCOUNT = 3
MAX_SCRIPT = 500
FORMATS = (".wav", ".mp3", ".flac", ".ogg")
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 -]{0,23}$")     # shown in the page and part of render file names
OWNER = re.compile(r"^[a-z0-9._-]{2,32}$")                  # accounts.USERNAME
FOLDER = re.compile(r"^[a-z0-9-]{1,24}$")


class UploadRefused(Exception):
    """An upload or removal that cannot be done. The message is safe to show."""


def root():
    configured = os.environ.get("UPLOADED_VOICES_DIR", "").strip()
    if configured:
        return Path(configured)
    voices = os.environ.get("VOICES_DIR", "").strip()
    return Path(voices) / "uploaded" if voices else None


def _owner_dir(owner, base=None):
    if not OWNER.match(str(owner or "")):
        raise UploadRefused("Only a logged-in account can have uploaded voices.")
    base = Path(base) if base else root()
    if base is None:
        raise UploadRefused("This server has no folder for uploaded voices (VOICES_DIR in .env).")
    return base / owner


def speaker_of(owner, name):
    """The voice's name in renders and in the page: the account and the name it was given, e.g. "ana Calm"."""
    return f"{owner.replace('.', '-')} {name}"


def folder_of(name):
    return "-".join(name.lower().split())


def _read(about_path):
    try:
        return json.loads(about_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def voices(owner=None, base=None):
    """The uploaded voices' about records: one account's, or (owner None) everyone's, by owner then name."""
    base = Path(base) if base else root()
    if base is None or not base.is_dir():
        return []
    owners = [owner] if owner else sorted(p.name for p in base.iterdir() if p.is_dir() and OWNER.match(p.name))
    found = []
    for who in owners:
        folder = _owner_dir(who, base)
        if not folder.is_dir():
            continue
        for voice in sorted(folder.iterdir()):
            about = _read(voice / "about.json")
            if about and about.get("owner") == who and (voice / "reference.wav").is_file():
                found.append({**about, "folder": voice.name})
    return found


def clips(owner=None, base=None):
    """The uploaded voices as Clips (dataset "Custom voices"), ready to clone."""
    base = Path(base) if base else root()
    out = []
    for about in voices(owner, base):
        voice = _owner_dir(about["owner"], base) / about["folder"]
        out.append(Clip(dataset=DATASET, clip_id=about["speaker"], speaker=about["speaker"],
                        text=(voice / "reference.txt").read_text(encoding="utf-8").strip(),
                        source=voice / "reference.wav",
                        details={"name": about.get("name") or about["speaker"], "uploaded": about["uploaded"][:10],
                                 "owner": about["owner"]}))
    return out


def _decode(data, filename):
    suffix = Path(str(filename or "")).suffix.lower()
    if suffix not in FORMATS:
        raise UploadRefused(f"Upload a {', '.join(f[1:] for f in FORMATS)} file.")
    if not data:
        raise UploadRefused("The file is empty.")
    if len(data) > MAX_BYTES:
        raise UploadRefused(f"The file is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    try:
        samples, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception:
        raise UploadRefused("That file could not be read as audio.")
    seconds = len(samples) / rate if rate else 0
    if not MIN_SECONDS <= seconds <= MAX_SECONDS:
        raise UploadRefused(f"The recording is {seconds:.1f} s long: it must be {MIN_SECONDS:g} to {MAX_SECONDS:g} s.")
    mono = samples.mean(axis=1)
    if float(np.max(np.abs(mono))) < 1e-3:
        raise UploadRefused("The recording is silent.")
    return mono, rate, seconds


def save(owner, name, script, data, filename, consent=True, base=None, now=None):
    """Keep an uploaded voice for `owner`. Returns its about record; raises UploadRefused with the reason."""
    if consent is not True:
        raise UploadRefused("Confirm that this is your own voice, or that the speaker has allowed you to upload it.")
    name = " ".join(str(name or "").split())
    if not NAME.match(name):
        raise UploadRefused("A voice name is 1 to 24 letters, digits, spaces or -, starting with a letter or digit.")
    script = " ".join(str(script or "").split())
    if not script or len(script) > MAX_SCRIPT:
        raise UploadRefused(f"Type the words spoken in the recording (up to {MAX_SCRIPT} characters).")
    folder = _owner_dir(owner, base)
    mine = voices(owner, base)
    if len(mine) >= MAX_PER_ACCOUNT:
        raise UploadRefused(f"An account can keep {MAX_PER_ACCOUNT} uploaded voices: delete one first.")
    slug = folder_of(name)
    if (not FOLDER.match(slug) or any(v["folder"] == slug or str(v.get("name", "")).lower() == name.lower() for v in mine)
            or (folder / slug).exists()):
        raise UploadRefused("You already have a voice with that name.")
    # Who may hear a render is decided by its voice's name, so no two voices may share one (usernames
    # "a.b" and "a-b" would otherwise both give "a-b <name>")
    if any(v["speaker"].lower() == speaker_of(owner, name).lower() for v in voices(None, base)):
        raise UploadRefused("That voice name cannot be used: choose another.")
    mono, rate, seconds = _decode(data, filename)

    target = folder / slug
    work = folder / f".{slug}.uploading"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        sf.write(str(work / "reference.wav"), mono, rate, subtype="PCM_16")
        (work / "reference.txt").write_text(script, encoding="utf-8")
        about = {"owner": owner, "name": name, "speaker": speaker_of(owner, name),
                 "uploaded": (now or datetime.now()).strftime("%Y-%m-%d %H:%M:%S"),
                 "seconds": round(seconds, 1), "original_format": Path(filename).suffix.lower()[1:],
                 "consent": "The uploader confirmed this is their own voice or that the speaker allowed it."}
        (work / "about.json").write_text(json.dumps(about, indent=2, ensure_ascii=False), encoding="utf-8")
        work.rename(target)                 # the voice appears whole, or not at all
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return {**about, "folder": slug}


def rename(owner, folder, name, base=None):
    """Give one of `owner`'s uploaded voices a new name to show. Its ID (speaker), folder and renders stay as they
    are. Returns its about record."""
    if not FOLDER.match(str(folder or "")):
        raise UploadRefused("No such voice.")
    mine = voices(owner, base)
    found = next((v for v in mine if v["folder"] == folder), None)
    if found is None:
        raise UploadRefused("No such voice.")
    name = " ".join(str(name or "").split())
    if not NAME.match(name):
        raise UploadRefused("A voice name is 1 to 24 letters, digits, spaces or -, starting with a letter or digit.")
    if any(v["folder"] != folder and str(v.get("name", "")).lower() == name.lower() for v in mine):
        raise UploadRefused("You already have a voice with that name.")
    path = _owner_dir(owner, base) / folder / "about.json"
    about = _read(path) or {}
    about["name"] = name
    work = path.with_suffix(".json.saving")
    work.write_text(json.dumps(about, indent=2, ensure_ascii=False), encoding="utf-8")
    work.replace(path)                      # the new name, whole, or the old one
    return {**about, "folder": folder}


def remove(owner, folder, base=None):
    """Delete one of `owner`'s uploaded voices from this computer. Returns its about record."""
    if not FOLDER.match(str(folder or "")):
        raise UploadRefused("No such voice.")
    found = next((v for v in voices(owner, base) if v["folder"] == folder), None)
    if found is None:
        raise UploadRefused("No such voice.")
    shutil.rmtree(_owner_dir(owner, base) / folder)
    return found

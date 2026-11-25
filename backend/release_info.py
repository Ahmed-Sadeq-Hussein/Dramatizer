"""release_info.py | What a running Dramatizer really is (Milestone 4, step 3): its commit, the voice pack and model
revisions it reads, which models it has loaded, and one hash of its settings.

GET /api/v1/version reports this from inside the server; deploy/release.py writes the same fields into
deploy/release.json for a candidate, from the files it will be started with; deploy/smoke.py compares the two.
Both sides compute the settings hash here, so they cannot drift apart.

Never a path, a password or a token: only revisions, hashes and names.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

# The settings that change what the server does, and that are never secret. Their values (paths included) are
# hashed, not shown. A secret (UI_PASSWORD, a token) must never be added here: the hash would not leak it, but
# release.json is made from deploy/container.env, which must not hold one either.
CONFIG_KEYS = (
    "API_PORT", "DEVICE", "VOICE_MODEL_DIR", "COSYVOICE_MODEL_DIR", "LLM_LIBRARY_DIR", "MODELS_DIR",
    "VOICE_LIBRARY_DIR", "VOICE_PACK_DIR", "VOICES_DIR", "OUTPUT_DIR", "IMPORT_DIR", "MEASUREMENTS_DIR",
    "MAX_QUEUE", "MAX_WAIT_SECONDS", "PRELOAD_VOICE_MODEL", "CHATBOT_PYTHON", "CHATBOT_MODEL", "CHATBOT_PRELOAD",
    "WER_MODEL", "SIMILARITY_MODEL", "PERQ_MODEL", "DENOISE_MODEL", "EMPHASIS_MODEL", "VOICE_SEED",
    "REGISTRATION", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
)


def config_sha256(values):
    """One sha256 over CONFIG_KEYS' values (missing = empty), in a fixed order."""
    text = "\n".join(f"{key}={str(values.get(key) or '').strip()}" for key in CONFIG_KEYS)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_env_file(path):
    """KEY=VALUE lines of an env file (as docker run --env-file reads them: no quotes, # starts a comment line).
    A list of files is read in order, a later file's value winning, as with several --env-file options."""
    if isinstance(path, (list, tuple)):
        values = {}
        for one in path:
            values.update(read_env_file(one))
        return values
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def commit():
    """The commit of this code: GIT_COMMIT (baked into the image) or, on the host, git's HEAD."""
    given = os.environ.get("GIT_COMMIT", "").strip()
    if given and given != "unknown":
        return given
    try:
        found = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, capture_output=True, text=True,
                               timeout=5, check=True)
        return found.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def voice_pack(folder):
    """{"revision", "manifest_sha256"} from the pack's pack.json, or None (no pack: the full library is used)."""
    pack = _read_json(Path(folder) / "pack.json") if folder else None
    if not isinstance(pack, dict):
        return None
    return {"revision": pack.get("revision"), "manifest_sha256": pack.get("manifest_sha256")}


def models(folder):
    """{name: {"repo", "revision"}} from the model store's models.json, or None (no store yet)."""
    store = _read_json(Path(folder) / "models.json") if folder else None
    if not isinstance(store, dict):
        return None
    entries = store.get("models", store)
    return {name: {"repo": entry.get("repo"), "revision": entry.get("revision")}
            for name, entry in sorted(entries.items()) if isinstance(entry, dict)}


def describe(values, loaded=()):
    """The fields /version shows and release.json holds, from these settings."""
    return {
        "commit": commit(),
        "voice_pack": voice_pack(values.get("VOICE_PACK_DIR")),
        "models": models(values.get("MODELS_DIR")),
        "config_sha256": config_sha256(values),
        "loaded": sorted(loaded),
    }


# The fields smoke.py compares between release.json and /version. "loaded" is not one: it changes as models load.
COMPARED = ("commit", "voice_pack", "models", "config_sha256")


def differences(expected, running):
    """[(field, expected, running)] for every compared field that differs. A field missing on one side differs."""
    return [(field, expected.get(field), running.get(field)) for field in COMPARED
            if field not in expected or field not in running or expected[field] != running[field]]

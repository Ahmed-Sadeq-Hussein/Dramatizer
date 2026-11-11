"""safe_files.py | Reading and replacing small JSON files that several requests use at once.

On Windows a file cannot be replaced while another thread has it open, and a
read can fail for a moment while it is being replaced (PermissionError). With
several people on the API at once (one logging in while another's page loads,
say) that turned a login or a saved score into a server error. These helpers
wait a moment and try again instead, and each writer uses its own temporary
file, so two writers never write into the same one.

    from backend.safe_files import read_json, read_text, write_json, replace

    data = read_json(path, default={})
    write_json(path, data)                 # a temporary file first, then swapped in
"""
import json
import os
import random
import time
import uuid
from pathlib import Path

TRIES = 100
WAIT = (0.01, 0.08)  # seconds between tries, at random so two waiters do not keep colliding: about 4 s in all


def _again(action):
    for attempt in range(TRIES):
        try:
            return action()
        except PermissionError:
            if attempt == TRIES - 1:
                raise
            time.sleep(random.uniform(*WAIT))


def read_text(path):
    return _again(lambda: Path(path).read_text(encoding="utf-8"))


def read_json(path, default=None):
    """The file's JSON, or `default` when there is no file (when default is not None)."""
    path = Path(path)
    if default is not None and not path.is_file():
        return default
    return json.loads(read_text(path))


def replace(temporary, path):
    """os.replace that waits while a reader has the file open."""
    try:
        _again(lambda: os.replace(temporary, path))
    except OSError:
        Path(temporary).unlink(missing_ok=True)
        raise


def temporary_for(path):
    path = Path(path)
    return path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")


def write_text(path, text):
    """A temporary file first, so a crash never leaves half a file, then swapped in."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_for(path)
    temporary.write_text(text, encoding="utf-8")
    replace(temporary, path)


def write_json(path, data, indent=2):
    write_text(path, json.dumps(data, indent=indent, ensure_ascii=False))

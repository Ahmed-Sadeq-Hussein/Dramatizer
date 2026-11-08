"""keys.py | Who may use the API: one key per person.

    python main_backend.py api-key add Ana       # makes Ana's key and shows it ONCE
    python main_backend.py api-key list          # who has a key
    python main_backend.py api-key remove Ana    # Ana's key stops working at once

Only a SHA-256 hash of each key is stored (history/api/keys.json, never
committed), so the file does not give the keys away. A lost key cannot be shown
again: remove it and add a new one. Give each key only to that person, through a
private channel, and only to members of the group (LIMITATIONS.md, Ethics).
"""
import argparse
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
from datetime import datetime
from pathlib import Path

from backend import model

KEYS_PATH = model.PROJECT_DIR / "history" / "api" / "keys.json"
NAME = re.compile(r"^[A-Za-z0-9 ._-]{1,40}$")
PREFIX = "drm_"
_lock = threading.Lock()


def _hash(key):
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def load(path=None):
    path = Path(path or KEYS_PATH)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"keys": {}}


def _save(data, path=None):
    path = Path(path or KEYS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def add(name, path=None):
    """Make a key for `name` and return it. It is not stored, only its hash: show it once."""
    name = " ".join(str(name).split())
    if not NAME.match(name):
        raise ValueError("A name must be 1 to 40 letters, digits, spaces or ._-")
    with _lock:
        data = load(path)
        if name.lower() in (n.lower() for n in data["keys"]):
            raise ValueError(f"{name} already has a key: remove it first to make a new one.")
        key = PREFIX + secrets.token_urlsafe(32)
        data["keys"][name] = {"sha256": _hash(key), "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        _save(data, path)
    return key


def remove(name, path=None):
    with _lock:
        data = load(path)
        match = next((n for n in data["keys"] if n.lower() == str(name).strip().lower()), None)
        if match is None:
            raise ValueError(f"Nobody called {name!r} has a key.")
        del data["keys"][match]
        _save(data, path)
    return match


def who(key, path=None):
    """The name whose key this is, or None. Compares every hash in constant time."""
    if not key or not key.startswith(PREFIX) or len(key) > 200:
        return None
    digest = _hash(key)
    found = None
    for name, entry in load(path)["keys"].items():
        if hmac.compare_digest(digest, entry["sha256"]):
            found = name
    return found


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage who may use the Dramatizer API.")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("add", help="make a key for a person").add_argument("name")
    sub.add_parser("remove", help="stop a person's key").add_argument("name")
    sub.add_parser("list", help="who has a key")
    args = parser.parse_args()

    try:
        if args.action == "add":
            key = add(args.name)
            print(f"Key for {args.name} (shown only now; send it to them privately):\n\n    {key}\n")
            print("They start the client with:  python main_frontend.py connect https://<playit address>:<port>")
        elif args.action == "remove":
            print(f"{remove(args.name)}'s key no longer works.")
        else:
            people = load()["keys"]
            if not people:
                print("Nobody has a key yet: python main_backend.py api-key add NAME")
            for name, entry in people.items():
                print(f"  {name:20s} since {entry['created']}")
    except ValueError as error:
        raise SystemExit(str(error))

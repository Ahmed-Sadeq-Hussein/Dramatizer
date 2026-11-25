"""move_state.py | Copy the site's state (history/) into a Docker named volume, once (Milestone 4, step 2).

    python deploy/move_state.py dramatizer-state --image dramatizer:v4

Docker Desktop shares Windows folders with Linux containers through a layer whose file locks SQLite cannot rely on,
so the database lives in a named volume (inside WSL's own disk) instead. This copies, into an EMPTY volume:

  - a consistent copy of the database (storage.backup, safe while the API runs), as dramatizer.db
  - everything else the site reads from history/: accounts' keys, the TLS certificate (the same one, so clients'
    pinned fingerprints still match), ratings, evidence, the records index, logs

and leaves out what the running site never reads and should not sit next to it: the sealed Report evaluation
(history/eval, api/eval_key.txt), passwords waiting to be handed out, old backups, migrated files and staging.

Then it reads the copy back inside the container (storage check) and compares every table's row count with the
host's. From then on the volume is the truth: do not start the API on the host's history/ again, or the two drift.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import storage  # noqa: E402

HISTORY = ROOT / "history"
LEFT_OUT = {"eval", "backups", "staging", "live", "releases", "front_door.json", "dramatizer.db", "dramatizer.db-wal",
            "dramatizer.db-shm", "api/eval_key.txt", "api/passwords_to_hand_out.txt", "api/test_users.txt"}


def left_out(relative):
    text = relative.as_posix()
    return text in LEFT_OUT or relative.parts[0] in LEFT_OUT or text.endswith(".migrated")


def docker(*args):
    done = subprocess.run(["docker", *args], capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"docker {' '.join(args[:2])}: {done.stderr.strip()}")
    return done.stdout


def gather(target):
    """The files to copy, in `target`; returns [relative paths] (the database copy included)."""
    copied = []
    for path in sorted(HISTORY.rglob("*")):
        relative = path.relative_to(HISTORY)
        if path.is_dir() or left_out(relative):
            continue
        (target / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target / relative)
        copied.append(relative.as_posix())
    shutil.copy2(storage.backup(), target / "dramatizer.db")
    copied.append("dramatizer.db")
    return copied


def counts_in(volume, image):
    out = docker("run", "--rm", "-v", f"{volume}:/app/history", "--entrypoint", "python", image,
                 "main_backend.py", "storage", "check")
    return json.loads(out[out.index("{"):])["tables"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Copy history/ into an empty Docker named volume.")
    parser.add_argument("volume")
    parser.add_argument("--image", required=True, help="an app image, to copy with and to check the copy")
    args = parser.parse_args(argv)

    existing = docker("volume", "ls", "-q", "--filter", f"name=^{args.volume}$").split()
    if existing:
        listed = docker("run", "--rm", "-v", f"{args.volume}:/v", "--entrypoint", "sh", args.image, "-c",
                        "ls -A /v | grep -v '^api$' ; ls -A /v/api 2>/dev/null").split()
        if listed:
            raise SystemExit(f"The volume {args.volume} is not empty ({len(listed)} entries): nothing copied.")
    else:
        docker("volume", "create", args.volume)

    host = storage.counts()["tables"]
    with tempfile.TemporaryDirectory() as folder:
        copied = gather(Path(folder))
        docker("run", "--rm", "--user", "root", "-v", f"{args.volume}:/dst", "-v", f"{folder}:/src:ro",
               "--entrypoint", "sh", args.image, "-c", "cp -a /src/. /dst/ && chown -R 10001:10001 /dst")
    copy = counts_in(args.volume, args.image)
    print(f"Copied {len(copied)} files into the volume {args.volume} (left out: {', '.join(sorted(LEFT_OUT))}).")
    print(json.dumps({"host": host, "volume": copy}, indent=1))
    if host != copy:
        raise SystemExit("The row counts differ: check before starting the site on this volume.")
    print("Same rows in every table. From now on the volume is the site's state; do not start the API on history/.")


if __name__ == "__main__":
    main()

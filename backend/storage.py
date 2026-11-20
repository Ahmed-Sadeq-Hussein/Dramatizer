"""storage.py | The project's database: accounts, login sessions and listener ratings, in one SQLite file.

    python main_backend.py storage migrate     # once: copy the old JSON files in (they are backed up first)
    python main_backend.py storage check       # the schema version and how many rows each table has
    python main_backend.py storage backup      # a consistent copy in history/backups/ (the last 14 are kept)

Why a database (Best practices.txt, part 2A). These used to be JSON files rewritten whole on every change
(history/api/accounts.json, sessions.json) and a log file (history/ratings/mos.jsonl) beside a ratings.json
per run. A lock inside one process kept two requests apart, but not the API and a command run at the same
moment, which could each overwrite the other's change. SQLite gives, with nothing to install:

    - transactions between processes: a change is all made or not at all, and writers take turns
      (BEGIN IMMEDIATE, WAL journal, a busy timeout instead of an error)
    - constraints: a score is 0 to 5, a session belongs to a real account, an email is used once
    - one place to back up (sqlite's own backup, safe while the API runs)

Where it is: history/dramatizer.db (DATABASE_PATH in .env changes it; git-ignored: it holds password hashes,
emails and ratings, which are personal data). Audio, embeddings and charts stay files.

Using it:
    with storage.connect() as db:              # read
        db.execute("SELECT ...")
    with storage.connect(write=True) as db:    # change: committed at the end, rolled back on an error
        db.execute("UPDATE ...")
A call inside another shares its transaction, so accounts.load() inside a change sees the change.

The schema is changed only by adding a step to MIGRATIONS (never by editing one that has run): each
database records how many steps it has had (PRAGMA user_version) and runs the rest when it is opened.
"""
import argparse
import json
import shutil
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from backend import model, safe_files

DB_PATH = model.PROJECT_DIR / model.setting("DATABASE_PATH", "history/dramatizer.db")
BACKUP_DIR = model.PROJECT_DIR / "history" / "backups"
BACKUPS_KEPT = 14
BUSY_SECONDS = 30            # how long a writer waits for another to finish before giving up

# The files the data came from (read only by `migrate`)
LEGACY = {
    "accounts": model.PROJECT_DIR / "history" / "api" / "accounts.json",
    "sessions": model.PROJECT_DIR / "history" / "api" / "sessions.json",
    "rating_log": model.PROJECT_DIR / "history" / "ratings" / "mos.jsonl",
}
MIGRATED = ".migrated"       # an imported file is renamed to this, so no one edits a copy that is not used

MIGRATIONS = [
    # 1: accounts, sessions, ratings
    """
    CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);

    CREATE TABLE accounts (
        username         TEXT PRIMARY KEY,
        display_name     TEXT NOT NULL,
        role             TEXT NOT NULL,
        password         TEXT NOT NULL,                 -- pbkdf2_sha256$rounds$salt$hash, never the password
        disabled         INTEGER NOT NULL DEFAULT 0 CHECK (disabled IN (0, 1)),
        pending          INTEGER NOT NULL DEFAULT 0 CHECK (pending IN (0, 1)),
        email            TEXT UNIQUE,                   -- lower case; NULL on accounts made before emails
        created          TEXT NOT NULL,
        password_changed TEXT NOT NULL,
        last_login       TEXT,
        failed_logins    INTEGER NOT NULL DEFAULT 0,
        locked_until     TEXT,
        schema_version   INTEGER NOT NULL
    );
    CREATE UNIQUE INDEX accounts_display_name ON accounts (lower(display_name));

    CREATE TABLE account_piles (                        -- the listening piles an account is to rate
        username TEXT NOT NULL REFERENCES accounts (username) ON DELETE CASCADE,
        pile     TEXT NOT NULL,                         -- "batch|pile"
        position INTEGER NOT NULL,
        PRIMARY KEY (username, pile)
    );

    CREATE TABLE sessions (
        token_hash TEXT PRIMARY KEY,                    -- SHA-256 of the token; the token itself is never kept
        username   TEXT NOT NULL REFERENCES accounts (username) ON DELETE CASCADE,
        created    TEXT NOT NULL,
        expires    TEXT NOT NULL
    );
    CREATE INDEX sessions_username ON sessions (username);

    CREATE TABLE rating_runs (                          -- one per rated genome search (its folder name)
        run     TEXT PRIMARY KEY,
        folder  TEXT,
        raters  TEXT NOT NULL DEFAULT '[]',             -- JSON list, in the order they first rated
        picked  TEXT NOT NULL DEFAULT '[]',             -- JSON list of {genome, name, time}
        updated TEXT
    );

    CREATE TABLE ratings (                              -- each listener's current score of each clip
        run          TEXT NOT NULL REFERENCES rating_runs (run) ON DELETE CASCADE,
        rater        TEXT NOT NULL,
        clip         TEXT NOT NULL,                     -- baseline | genome_NN
        score        REAL NOT NULL CHECK (score BETWEEN 0 AND 5),
        artifact     INTEGER CHECK (artifact IN (0, 1)),  -- heard an artifact; NULL: not asked
        first_given  TEXT,
        time         TEXT,
        changes      INTEGER NOT NULL DEFAULT 0,
        account      TEXT,
        submitted_by TEXT,
        PRIMARY KEY (run, rater, clip)
    );
    CREATE INDEX ratings_rater ON ratings (rater);

    CREATE TABLE rating_log (                           -- every score ever given or changed; rows are only added
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        time    TEXT NOT NULL,
        run     TEXT NOT NULL,
        rater   TEXT NOT NULL,
        account TEXT,
        clip    TEXT NOT NULL,
        score   REAL,
        entry   TEXT NOT NULL                           -- the whole entry as JSON (voice, line, genes, markup...)
    );
    CREATE INDEX rating_log_run ON rating_log (run);
    CREATE TRIGGER rating_log_is_append_only_u BEFORE UPDATE ON rating_log
        BEGIN SELECT RAISE(ABORT, 'the rating log is append-only'); END;
    CREATE TRIGGER rating_log_is_append_only_d BEFORE DELETE ON rating_log
        BEGIN SELECT RAISE(ABORT, 'the rating log is append-only'); END;
    """,
    # 2: whether the listener heard a different person than the voice's original recording (NULL: not asked)
    """
    ALTER TABLE ratings ADD COLUMN different INTEGER CHECK (different IN (0, 1));
    """,
]
SCHEMA_VERSION = len(MIGRATIONS)

_local = threading.local()
_ready = set()               # databases already brought up to SCHEMA_VERSION in this process
_ready_lock = threading.Lock()


def _open(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=BUSY_SECONDS, isolation_level=None, check_same_thread=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys = ON")
    db.execute(f"PRAGMA busy_timeout = {BUSY_SECONDS * 1000}")
    return db


def _migrate(db, path):
    key = str(Path(path).resolve())
    with _ready_lock:
        if key in _ready:
            return
        db.execute("PRAGMA journal_mode = WAL")             # readers do not wait for a writer
        db.execute("BEGIN IMMEDIATE")
        try:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(f"{path} has schema {version}, newer than this code ({SCHEMA_VERSION}): update the code.")
            for step in range(version, SCHEMA_VERSION):
                for statement in _statements(MIGRATIONS[step]):
                    db.execute(statement)
                db.execute(f"PRAGMA user_version = {step + 1}")
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        _ready.add(key)


def _statements(script):
    """The statements of a migration, one by one (executescript would commit on its own)."""
    out, current = [], []
    for line in script.splitlines():
        current.append(line)
        if sqlite3.complete_statement("\n".join(current)):
            text = "\n".join(current).strip()
            if text:
                out.append(text)
            current = []
    return out


@contextmanager
def connect(path=None, write=False):
    """A connection for one unit of work: read, or (write=True) change with a transaction.

    Inside another connect() of the same thread, the same connection and transaction are used.
    """
    outer = getattr(_local, "db", None)
    if outer is not None:
        if write and not _local.write:
            raise RuntimeError("A change was started inside a read: open the outer connect() with write=True.")
        yield outer
        return
    path = Path(path or DB_PATH)
    db = _open(path)
    try:
        _migrate(db, path)
        db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        _local.db, _local.write = db, write
        try:
            yield db
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        finally:
            _local.db = None
    finally:
        db.close()


def counts(path=None):
    with connect(path) as db:
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type = 'table' "
                                           "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        return {"schema_version": db.execute("PRAGMA user_version").fetchone()[0],
                "tables": {t: db.execute(f'SELECT count(*) FROM "{t}"').fetchone()[0] for t in tables}}


def get_meta(db, key, default=None):
    row = db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else default


def set_meta(db, key, value):
    db.execute("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
               (key, str(value)))


# --------------------------------------------------------------------------- backups
def backup(path=None, folder=None, keep=BACKUPS_KEPT):
    """A consistent copy of the database (safe while the API runs); the oldest beyond `keep` are deleted."""
    path, folder = Path(path or DB_PATH), Path(folder or BACKUP_DIR)
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{path.stem}-{datetime.now():%Y%m%d-%H%M%S}.db"
    source, copy = _open(path), sqlite3.connect(target)
    try:
        source.backup(copy)
    finally:
        copy.close()
        source.close()
    for old in sorted(folder.glob(f"{path.stem}-*.db"))[:-keep]:
        old.unlink()
    return target


# --------------------------------------------------------------------------- the one-time import
def migrate(path=None, rating_folders=None, legacy=None, rename=True, backup_folder=None):
    """Copy the old files into the database, once. Returns what was imported.

    Every file read is first copied to history/backups/pre-database-<time>/. Accounts and the rating log are
    imported only if their tables are empty; each run's ratings.json only if the run is not in the database.
    Then each imported run is read back and compared with its file: any difference stops the import
    (nothing is committed). Last, the imported account, session and log files are renamed *.migrated.
    The ratings.json files stay: they are kept up to date as a copy for reading, the database is the truth.
    """
    from backend.api import accounts
    from backend.genome import ratings

    legacy = {**LEGACY, **(legacy or {})}
    folders = list(rating_folders) if rating_folders is not None else _rating_folders()
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    saved = Path(backup_folder or BACKUP_DIR) / f"pre-database-{stamp}"
    files = [p for p in legacy.values() if Path(p).is_file()] + [f / ratings.RATINGS_FILE for f in folders]
    for i, f in enumerate(files):
        target = saved / f"{i:03d}_{Path(f).parent.name}_{Path(f).name}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)

    done = {"backup": str(saved) if files else None, "accounts": 0, "sessions": 0, "rating_log": 0, "runs": 0}
    with connect(path, write=True) as db:
        if Path(legacy["accounts"]).is_file() and not db.execute("SELECT 1 FROM accounts LIMIT 1").fetchone():
            records = safe_files.read_json(legacy["accounts"]).get("accounts", {})
            imported = {name: accounts._upgrade(record) for name, record in records.items()}
            accounts._save(imported)
            done["accounts"] = len(imported)
        if Path(legacy["sessions"]).is_file() and not db.execute("SELECT 1 FROM sessions LIMIT 1").fetchone():
            now = accounts._now()
            known = {r[0] for r in db.execute("SELECT username FROM accounts")}
            for token_hash, s in safe_files.read_json(legacy["sessions"], default={}).items():
                if s["username"] in known and accounts._parse(s["expires"]) > now:
                    db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?)",
                               (token_hash, s["username"], s["created"], s["expires"]))
                    done["sessions"] += 1
        if Path(legacy["rating_log"]).is_file() and not db.execute("SELECT 1 FROM rating_log LIMIT 1").fetchone():
            lines = Path(legacy["rating_log"]).read_text(encoding="utf-8").splitlines()
            entries = [json.loads(line) for line in lines if line.strip()]
            ratings._log(db, entries)
            done["rating_log"] = len(entries)
        for folder in folders:
            if db.execute("SELECT 1 FROM rating_runs WHERE run = ?", (folder.name,)).fetchone():
                continue
            from_file = safe_files.read_json(folder / ratings.RATINGS_FILE)
            ratings._store(db, folder, from_file)
            again = ratings._from_db(db, folder.name)
            if _comparable(again) != _comparable(from_file):
                raise RuntimeError(f"{folder}: the ratings read back from the database differ from the file. "
                                   "Nothing was imported.")
            done["runs"] += 1
        set_meta(db, "migrated_from_files", stamp)

    if rename:
        for key in ("accounts", "sessions", "rating_log"):
            p = Path(legacy[key])
            if p.is_file():
                p.replace(p.with_name(p.name + MIGRATED))
    return done


def _comparable(doc):
    """A ratings document without the fields that only differ in form (int/float, a missing empty dict)."""
    given = {r: {c: {k: (float(v) if k == "score" else v) for k, v in d.items()
                     if k not in ("artifact", "different") or v is not None}
                 for c, d in clips.items()} for r, clips in doc.get("given", {}).items()}
    return {"raters": doc["raters"], "picked": doc["picked"], "updated": doc["updated"],
            "scores": {r: {c: float(s) for c, s in clips.items()} for r, clips in doc["scores"].items()},
            "artifacts": {r: dict(c) for r, c in doc.get("artifacts", {}).items() if c},
            "different": {r: dict(c) for r, c in doc.get("different", {}).items() if c}, "given": given}


def _rating_folders():
    """Every run folder with a ratings.json: the genome searches, in OUTPUT_DIR and in the old output/."""
    roots = {model.OUTPUT_DIR, model.PROJECT_DIR / "output"}
    found = {}
    for root in roots:
        for f in Path(root).glob("**/ratings.json"):
            if (f.parent / "run.json").is_file():
                found.setdefault(f.parent.name, f.parent)
    return sorted(found.values())


def legacy_left(path=None):
    """The old files that are still in use (not yet imported), for a clear message instead of an empty login."""
    return [str(p) for p in LEGACY.values() if Path(p).is_file()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The project's database (accounts, sessions, ratings).")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("migrate", help="copy the old JSON files in, once (backed up first)")
    sub.add_parser("check", help="the schema version and the rows in each table")
    sub.add_parser("backup", help=f"a consistent copy in {BACKUP_DIR} (the last {BACKUPS_KEPT} are kept)")
    args = parser.parse_args()
    # Run as a command, this file is "__main__", a second copy of the module beside the backend.storage that
    # accounts.py and ratings.py import. Use that one, so there is one connection per thread, not two that lock.
    from backend import storage as shared
    if args.action == "migrate":
        result = shared.migrate()
        print(f"Imported into {shared.DB_PATH}:")
        for key, value in result.items():
            print(f"  {key:11s} {value}")
    elif args.action == "check":
        print(json.dumps(shared.counts(), indent=2))
    else:
        print(f"Backed up to {shared.backup()}")

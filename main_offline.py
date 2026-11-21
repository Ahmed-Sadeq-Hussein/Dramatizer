"""main_offline.py | A local Dramatizer for debugging: the whole site in one go, on a copy of the data.

    python main_offline.py                  # a menu (or double-click "Dramatizer offline.bat")
    python main_offline.py site             # the website on the sandbox, logged in as debug-admin
    python main_offline.py site --as listener      # ... as debug-listener (or researcher, key, login)
    python main_offline.py site --live      # the website on the REAL data: you log in with your own account
    python main_offline.py test             # every test that needs no GPU (main_backend.py test)
    python main_offline.py test lab         # only the test files whose path has "lab" in it
    python main_offline.py reset            # throw the sandbox away; the next start copies the data again
    python main_offline.py run NAME ...     # any main_backend.py command (on the real data, as there)

What `site` does, in one process tree you stop with Ctrl+C:

    - the API on https://127.0.0.1:5105 (OFFLINE_API_PORT) and the page on http://127.0.0.1:5111
      (OFFLINE_PAGE_PORT): other ports than the real API (5005) and client (5010), so both can run at once.
      Only this PC can reach them: nothing is tunnelled
    - the browser opens already logged in: debug-admin, debug-researcher or debug-listener (accounts that
      exist only in the sandbox, with the agreement accepted), an API key (--as key), or the login page
      (--as login: any account in the copy, your own included, with its real password)
    - debug-admin can do everything: every permission, the private Policy lab, every voice people uploaded,
      and every listening pile, the Report evaluation's too (still blind: see open_everything).
      debug-researcher and debug-listener have their roles' limits, to test what those people see
    - it restarts by itself when a .py file of the project changes, so a fix is live a few seconds after
      saving. The page's own files (frontend/client/static) need no restart: reload the browser
    - the chatbot's model loads on the first question, not at start (--chatbot loads it at start)

The sandbox (OUTPUT_DIR/offline, on the renders drive) is a copy made on the first start: the database
(accounts, sessions, ratings), the genome searches, the renders and their notes, the named genomes. Everything
the site writes goes there: ratings, renders, searches, uploads, deleted renders, lab results, logs. The real
data is never changed. `reset` (or --fresh) copies it again from the real data as it is now.

--live is the exception: the real database and folders, as `python main_backend.py api` uses them.
"""
import argparse
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

API_PORT = int(os.environ.get("OFFLINE_API_PORT", "5105"))
PAGE_PORT = int(os.environ.get("OFFLINE_PAGE_PORT", "5111"))
ROLES = {"admin": "debug-admin", "researcher": "debug-researcher", "listener": "debug-listener"}
WATCHED = ("backend", "frontend/client", "API calls", "private", "main_offline.py")


# --------------------------------------------------------------------------- #
# The sandbox
# --------------------------------------------------------------------------- #

def sandbox_dir():
    from backend import model
    return Path(model.OUTPUT_DIR) / "offline"


def real_paths():
    """Where the real data is, before anything is pointed at the sandbox."""
    from backend import storage
    from backend.api import server
    from backend.genome import ratings, search
    return {"db": Path(storage.DB_PATH), "runs": Path(search.RUNS_DIR), "renders": Path(server.GENOME_RENDERS),
            "speak": Path(server.SPEAK_DIR), "named": Path(ratings.NAMED_PATH)}


def make_sandbox(fresh=False):
    """Copy the real data into the sandbox (once, or again with fresh=True). Returns the sandbox folder."""
    box = sandbox_dir()
    if (box / "made.json").is_file() and not fresh:
        return box
    if box.exists():                                 # --fresh, or a copy that was stopped half-way
        print(f"Removing the old sandbox ({box})...", flush=True)
        shutil.rmtree(box)
    real = real_paths()
    print(f"Making the sandbox in {box} (a copy of the real data; it is never written back)...", flush=True)
    box.mkdir(parents=True, exist_ok=True)

    # The database through SQLite's own backup, so a copy taken while the real API is running is consistent
    source, target = sqlite3.connect(real["db"]), sqlite3.connect(box / "dramatizer.db")
    with target:
        source.backup(target)
    source.close()
    target.close()
    who = real["db"].parent / "ratings" / "who_rated_what.csv"
    if who.is_file():
        (box / "ratings").mkdir(exist_ok=True)
        shutil.copy2(who, box / "ratings" / who.name)

    print("  copying the genome searches...", flush=True)
    if real["runs"].is_dir():
        shutil.copytree(real["runs"], box / "genome_search")
    print("  copying the renders...", flush=True)
    if real["renders"].is_dir():
        shutil.copytree(real["renders"], box / "api" / "genome_renders")
    for name in ("render_ratings.json", "private_genomes.json", "lab_results.json"):
        if (real["speak"] / name).is_file():
            (box / "api").mkdir(exist_ok=True)
            shutil.copy2(real["speak"] / name, box / "api" / name)
    if real["named"].is_file():
        shutil.copy2(real["named"], box / "named.json")
    from backend.voice_data import uploaded_voices
    real_uploads = uploaded_voices.root()
    if real_uploads and Path(real_uploads).is_dir():
        shutil.copytree(real_uploads, box / "uploaded_voices")
    (box / "uploaded_voices").mkdir(exist_ok=True)

    point_at_sandbox(box)
    from backend.api import accounts, agreement
    for role, username in ROLES.items():
        if not accounts.get(username):
            accounts.new_account(username, f"Debug {role.title()}", role=role)
        accounts.accept_agreement(username, agreement.VERSION)
    (box / "made.json").write_text(json.dumps({"made": time.strftime("%Y-%m-%d %H:%M:%S"),
                                               "from": {k: str(v) for k, v in real.items()}}, indent=2), encoding="utf-8")
    print("  done.", flush=True)
    return box


def open_everything(box):
    """
    Make the sandbox show the debug accounts everything (run on every start; doing it twice changes nothing):

    - the private extensions (Policy lab): its runs are copied, so retraining writes the sandbox's copy
    - the voices people uploaded: copied, so admins see and use them as on the host
    - the Report evaluation piles, normally seen only by their two listeners. They stay blind: the key of
      which version is which is not in the runs and is not copied, and the scores already given to these piles
      are taken out of the sandbox's copy, so nobody's answers are shown before every score is in
    """
    private_runs = PROJECT_DIR / "private" / "genome_policy" / "runs"
    if private_runs.is_dir() and not (box / "policy_runs").exists():
        shutil.copytree(private_runs, box / "policy_runs")

    runs = box / "genome_search"
    sealed = []
    for note in runs.glob("*/run.json") if runs.is_dir() else []:
        try:
            if json.loads(note.read_text(encoding="utf-8")).get("sealed"):
                sealed.append(note.parent.name)
                (note.parent / "ratings.json").unlink(missing_ok=True)
        except (OSError, ValueError):
            pass
    if sealed:
        db = sqlite3.connect(box / "dramatizer.db")
        marks = ",".join("?" * len(sealed))
        with db:
            triggers = db.execute("SELECT sql FROM sqlite_master WHERE type = 'trigger' AND tbl_name = 'rating_log'").fetchall()
            for (sql,) in triggers:              # the copy's log is append-only like the real one: lifted for this
                db.execute(f"DROP TRIGGER {sql.split()[2]}")
            db.execute(f"DELETE FROM rating_log WHERE run IN ({marks})", sealed)
            db.execute(f"DELETE FROM ratings WHERE run IN ({marks})", sealed)
            for (sql,) in triggers:
                db.execute(sql)
        db.close()
        csv = box / "ratings" / "who_rated_what.csv"
        if csv.is_file():
            lines = csv.read_text(encoding="utf-8-sig").splitlines(keepends=True)
            kept = [line for line in lines if not any(f",{run}," in line for run in sealed)]
            if len(kept) != len(lines):
                csv.write_text("﻿" + "".join(kept), encoding="utf-8")


def point_at_sandbox(box):
    """Point every place the site reads and writes at the sandbox. Must run before the site starts."""
    os.environ["UPLOADED_VOICES_DIR"] = str(box / "uploaded_voices")
    os.environ["DATABASE_PATH"] = str(box / "dramatizer.db")
    from backend import storage
    from backend.api import accounts, assistant, keys, server
    from backend.genome import ratings, search

    storage.DB_PATH = box / "dramatizer.db"
    search.RUNS_DIR = box / "genome_search"
    ratings.NAMED_PATH = box / "named.json"
    ratings.LOG_PATH = box / "rating_log.jsonl"
    accounts.ACCOUNTS_PATH, accounts.SESSIONS_PATH = box / "accounts.json", box / "sessions.json"
    keys.KEYS_PATH = box / "keys.json"
    assistant.LOG_PATH = box / "logs" / "assistant.log"
    speak = box / "api"
    server.SPEAK_DIR = speak
    server.GENOME_RENDERS = speak / "genome_renders"
    server.RENDER_RATINGS = speak / "render_ratings.json"
    server.PRIVATE_GENOMES = speak / "private_genomes.json"
    server.LAB_RESULTS = speak / "lab_results.json"
    server.ACCESS_LOG, server.AUDIT_LOG = box / "logs" / "access.log", box / "logs" / "audit.jsonl"

    # The debug accounts may use everything: the private extensions, and the piles kept for named listeners
    debug = set(ROLES.values())
    server.PRIVATE_ACCOUNTS = set(server.PRIVATE_ACCOUNTS) | debug
    try:
        from private.genome_policy import lab as policy
        policy.RUNS = box / "policy_runs"
    except ImportError:
        pass
    only_for_listeners = server.may_see

    def may_see(run):
        account = getattr(server.g, "account", None)
        if run.get("only") and account is not None and account.username in debug:
            return True
        return only_for_listeners(run)
    server.may_see = may_see
    (box / "logs").mkdir(parents=True, exist_ok=True)
    speak.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# The site: the API and the page in one process (the child the watcher restarts)
# --------------------------------------------------------------------------- #

def serve(who, live, chatbot):
    if not chatbot:
        os.environ["CHATBOT_PRELOAD"] = "0"
    from backend.api import accounts, keys, server

    if not live:
        point_at_sandbox(sandbox_dir())
    token, name = None, None
    if who == "key":
        token, name = keys.add(f"offline-{int(time.time())}"), "offline key"
    elif who in ROLES:
        account = accounts.get(ROLES[who])
        if account is None:
            raise SystemExit(f"{ROLES[who]} is not in the sandbox: run  python main_offline.py reset")
        token, name = accounts.open_session(account), account.display_name

    threading.Thread(target=lambda: server.main(port=API_PORT), name="api", daemon=True).start()
    url = f"https://127.0.0.1:{API_PORT}"
    for _ in range(100):                                        # wait for the API to listen
        try:
            urllib.request.urlopen(f"{url}/api/v1/health", context=_unchecked(), timeout=1)
            break
        except OSError:
            time.sleep(0.2)

    from frontend.client import client
    expected = "".join(c for c in server.fingerprint() if c.isalnum()).lower()
    session = client.session_for(url, token, expected)
    print(f"\n  Page: http://127.0.0.1:{PAGE_PORT}   ({'REAL DATA' if live else 'sandbox'}, "
          f"{name or 'log in on the page'})\n", flush=True)
    client.make_app(url, session, name).run(host="127.0.0.1", port=PAGE_PORT, threaded=True)


def _unchecked():
    import ssl
    context = ssl.create_default_context()
    context.check_hostname, context.verify_mode = False, ssl.CERT_NONE      # only to see that our own API is up
    return context


# --------------------------------------------------------------------------- #
# The watcher: restart the site when a .py file changes
# --------------------------------------------------------------------------- #

def python_files():
    for place in WATCHED:
        path = PROJECT_DIR / place
        files = [path] if path.is_file() else path.rglob("*.py") if path.is_dir() else []
        for file in files:
            if "__pycache__" not in file.parts:
                yield file


def snapshot():
    out = {}
    for file in python_files():
        try:
            out[file] = file.stat().st_mtime
        except OSError:
            pass
    return out


def stop(process):
    """Stop the site and everything it started (the chatbot's model runs in a process of its own)."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    process.wait()


def page_up():
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PAGE_PORT}/", timeout=1)
        return True
    except OSError:
        return False


def site(args):
    if not args.live:
        open_everything(make_sandbox(fresh=args.fresh))
    else:
        print("\n  --live: the REAL database and folders. Ratings, renders and deletions here are real.\n", flush=True)
    command = [sys.executable, str(Path(__file__).resolve()), "_serve", "--as", args.who]
    if args.live:
        command.append("--live")
    if args.chatbot:
        command.append("--chatbot")
    extra = {"start_new_session": True} if os.name != "nt" else {}
    opened = args.no_browser
    while True:
        seen = snapshot()
        process = subprocess.Popen(command, cwd=PROJECT_DIR, **extra)
        changed = None
        try:
            while changed is None and process.poll() is None:
                time.sleep(1)
                if not opened and page_up():
                    webbrowser.open(f"http://127.0.0.1:{PAGE_PORT}/")
                    opened = True
                now = snapshot()
                changed = next((f for f in now.keys() | seen.keys() if now.get(f) != seen.get(f)), None)
        except KeyboardInterrupt:
            print("\nStopping...", flush=True)
            stop(process)
            return 0
        if changed is None:                        # it stopped by itself: wait for a fix, then start again
            print(f"\n  The site stopped (exit code {process.returncode}). Save a fix to start it again, "
                  "or press Ctrl+C.\n", flush=True)
            try:
                while snapshot() == seen:
                    time.sleep(1)
            except KeyboardInterrupt:
                return 0
            continue
        print(f"\n  {changed.relative_to(PROJECT_DIR)} changed: restarting...\n", flush=True)
        stop(process)


# --------------------------------------------------------------------------- #
# Tests, other commands, the menu
# --------------------------------------------------------------------------- #

def tests(words):
    import main_backend
    if not words:
        return main_backend.run_tests()
    chosen = [(t, n) for t, n in main_backend.TESTS if all(w.lower() in t.lower() for w in words)]
    if not chosen:
        print(f"No test file matches {' '.join(words)!r}. They are:\n  " + "\n  ".join(t for t, _ in main_backend.TESTS))
        return 2
    main_backend.TESTS[:] = chosen
    return main_backend.run_tests()


def reset():
    box = sandbox_dir()
    if box.exists():
        shutil.rmtree(box)
        print(f"Removed the sandbox ({box}). The next start copies the real data again.")
    else:
        print("There is no sandbox yet.")
    return 0


def backend_command(argv):
    return subprocess.run([sys.executable, str(PROJECT_DIR / "main_backend.py"), *argv], cwd=PROJECT_DIR).returncode


MENU = [
    ("Website on the sandbox, as debug-admin", ["site"]),
    ("Website on the sandbox, as debug-researcher", ["site", "--as", "researcher"]),
    ("Website on the sandbox, as debug-listener", ["site", "--as", "listener"]),
    ("Website on the sandbox, with the login page (any account in the copy)", ["site", "--as", "login"]),
    ("Website on the sandbox, as debug-admin, with the chatbot's model loaded at start", ["site", "--chatbot"]),
    ("Website on the REAL data (log in with your own account)", ["site", "--live", "--as", "login"]),
    ("Run every test", ["test"]),
    ("Run some tests (type words of their file names)", None),
    ("A main_backend.py command (type it)", None),
    ("Copy the real data into the sandbox again", ["reset"]),
]


def menu():
    print("\nDramatizer offline: a local instance for debugging\n")
    for number, (what, _) in enumerate(MENU, 1):
        print(f"  {number:2d}  {what}")
    print("   q  Quit\n")
    choice = input("Choose: ").strip().lower()
    if choice in ("q", ""):
        return 0
    if not choice.isdigit() or not 1 <= int(choice) <= len(MENU):
        print("Not one of the choices.")
        return 2
    what, argv = MENU[int(choice) - 1]
    if argv is None and what.startswith("Run some tests"):
        argv = ["test", *input("Words of the test files' names (e.g. lab api): ").split()]
    elif argv is None:
        import shlex
        argv = ["run", *shlex.split(input("python main_backend.py ").strip(), posix=False)]
    return main(argv)


def main(argv):
    if not argv:
        return menu()
    parser = argparse.ArgumentParser(prog="main_offline.py", description="A local Dramatizer for debugging.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("site", "_serve"):
        p = sub.add_parser(name, help="the website (API and page), restarted when a .py file changes" if name == "site"
                           else "(used by site: the process it restarts)")
        p.add_argument("--as", dest="who", default="admin", choices=[*ROLES, "key", "login"],
                       help="who the page is logged in as (default admin; login: the login page)")
        p.add_argument("--live", action="store_true", help="the REAL data instead of the sandbox")
        p.add_argument("--chatbot", action="store_true", help="load the chatbot's model at start")
        if name == "site":
            p.add_argument("--fresh", action="store_true", help="copy the real data into the sandbox again first")
            p.add_argument("--no-browser", action="store_true", help="do not open the browser")
    t = sub.add_parser("test", help="the tests; words narrow them to the files whose path has them")
    t.add_argument("words", nargs="*")
    sub.add_parser("reset", help="throw the sandbox away")
    r = sub.add_parser("run", help="a main_backend.py command")
    r.add_argument("rest", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.command == "site":
        return site(args)
    if args.command == "_serve":
        serve(args.who, args.live, args.chatbot)
        return 0
    if args.command == "test":
        return tests(args.words)
    if args.command == "reset":
        return reset()
    return backend_command(args.rest)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

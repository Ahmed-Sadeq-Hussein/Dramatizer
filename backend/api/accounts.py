"""accounts.py | Who may log in: accounts with a username and password, and the sessions they open.

    python main_backend.py account add julius --name Julius               # a listener account
    python main_backend.py account add ana --name Ana --role admin        # an account that may do everything
    python main_backend.py account list                                   # who has an account
    python main_backend.py account reset julius                           # a new password for julius
    python main_backend.py account disable julius | enable julius         # stop or allow logging in
    python main_backend.py account remove julius                          # delete the account
    python main_backend.py account assign julius mos-diverse-v1 2         # julius rates listening pile 2
    python main_backend.py account unassign julius mos-diverse-v1 2

`add` and `reset` make a random password and show it once. With
--password-file FILE it is written to that file instead of the screen, so it
can be passed on privately. Give each password only to its owner.

How passwords are kept (history/api/accounts.json, never committed):

    - never as text: only a salted PBKDF2-SHA256 hash, 600,000 rounds (the OWASP
      2023 advice for PBKDF2-SHA256), stored as "pbkdf2_sha256$rounds$salt$hash",
      so the method can be made stronger later and old hashes still checked
    - checked in constant time, and a wrong username costs as long as a wrong
      password, so a guesser cannot tell which usernames exist
    - after MAX_FAILURES wrong passwords in a row the account is locked for
      LOCK_MINUTES; logging in successfully resets the count

A login opens a session: a random token that the client sends as
"Authorization: Bearer <token>". Only its SHA-256 hash is stored
(history/api/sessions.json), it ends after SESSION_HOURS or at logout, and
resetting, disabling or removing an account ends its sessions at once.

The account record is the model the login system grows from. Its fields:

    username          the login name: lower case, 2-32 letters, digits, . _ -
    display_name      the name shown in the page and saved with ratings
    role              what the account may do (ROLES below)
    disabled          true: cannot log in
    created, password_changed, last_login      times, local
    failed_logins, locked_until                the lockout state
    password          the hash described above (never sent anywhere)
    assignments       the listening piles this account is to rate, as "batch|pile"
                      (explore.py makes the piles). A listener with assignments
                      sees and rates only those; with none, every pile.

Adding a field: give it a default in `new_account`, and `public` decides
whether the page may see it. Records older than SCHEMA_VERSION are filled in by
`_upgrade` when read.
"""
import argparse
import hashlib
import hmac
import re
import secrets
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from backend import model, safe_files

SCHEMA_VERSION = 1
ACCOUNTS_PATH = model.PROJECT_DIR / "history" / "api" / "accounts.json"
SESSIONS_PATH = model.PROJECT_DIR / "history" / "api" / "sessions.json"

# What each role may do. The API checks permissions, never role names, so a new
# role is one line here.
ROLES = {
    "listener": {"rate", "speak"},                          # rate genomes as themself; speak and render lines
    "researcher": {"rate", "speak", "search", "name"},      # also make speech, start searches, name genomes
    "admin": {"rate", "speak", "search", "name", "rate_for_others"},
}
PERMISSIONS = sorted(set().union(*ROLES.values()))

USERNAME = re.compile(r"^[a-z0-9._-]{2,32}$")
DISPLAY_NAME = re.compile(r"^[\w .'-]{1,40}$")      # the same rule as a listener's name in ratings.py
MIN_PASSWORD = 12
MAX_PASSWORD = 200
HASH_NAME = "pbkdf2_sha256"
ROUNDS = 600_000
MAX_FAILURES = 5
LOCK_MINUTES = 15
SESSION_HOURS = 12
TOKEN_PREFIX = "drs_"            # sessions; API keys start with drm_ (keys.py)

_lock = threading.Lock()


def _now():
    return datetime.now().replace(microsecond=0)


def _stamp(time=None):
    return (time or _now()).strftime("%Y-%m-%d %H:%M:%S")


def _parse(stamp):
    return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S") if stamp else None


def _write(path, data):
    safe_files.write_json(path, data)       # waits while another request reads the file (Windows)


# --------------------------------------------------------------------------- #
# Passwords
# --------------------------------------------------------------------------- #

def hash_password(password, rounds=None, salt=None):
    rounds = rounds or ROUNDS
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), rounds).hex()
    return f"{HASH_NAME}${rounds}${salt}${digest}"


def check_password(password, stored):
    """True if `password` matches the stored hash. Constant time; never raises on a bad hash."""
    try:
        method, rounds, salt, digest = stored.split("$")
        if method != HASH_NAME:
            return False
        again = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), int(rounds)).hex()
    except (AttributeError, ValueError):
        return False
    return hmac.compare_digest(again, digest)


# Checked when the username does not exist, so that costs as long as a wrong password
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16), salt="00" * 16)


def new_password():
    """A random password: 16 characters from letters and digits that are easy to read (no 0/O, 1/l/I)."""
    alphabet = "abcdefghijkmnpqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    while True:
        word = "".join(secrets.choice(alphabet) for _ in range(16))
        if any(c.isdigit() for c in word) and any(c.islower() for c in word) and any(c.isupper() for c in word):
            return word


def password_problem(password):
    """Why a chosen password is refused, or None."""
    if not isinstance(password, str) or not MIN_PASSWORD <= len(password) <= MAX_PASSWORD:
        return f"A password must be {MIN_PASSWORD} to {MAX_PASSWORD} characters."
    if len(set(password)) < 6:
        return "That password repeats too few different characters."
    return None


# --------------------------------------------------------------------------- #
# The account model
# --------------------------------------------------------------------------- #

@dataclass
class Account:
    username: str
    display_name: str
    role: str = "listener"
    password: str = ""
    disabled: bool = False
    created: str = field(default_factory=_stamp)
    password_changed: str = field(default_factory=_stamp)
    last_login: str | None = None
    failed_logins: int = 0
    locked_until: str | None = None
    assignments: list = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    @property
    def permissions(self):
        return set(ROLES.get(self.role, ()))

    def can(self, permission):
        return permission in self.permissions

    def locked(self, now=None):
        until = _parse(self.locked_until)
        return until is not None and (now or _now()) < until

    def may_rate_pile(self, batch, pile):
        """A listening pile this account may rate: its own, or any if it has none or may rate for others."""
        if not self.assignments or self.can("rate_for_others"):
            return True
        return pile_key(batch, pile) in self.assignments

    def public(self):
        """What the page may see: never the password hash or the lockout details."""
        return {"username": self.username, "display_name": self.display_name, "role": self.role,
                "permissions": sorted(self.permissions), "last_login": self.last_login,
                "assignments": list(self.assignments)}


def pile_key(batch, pile):
    return f"{batch}|{int(pile)}"


def clean_username(username):
    username = str(username or "").strip().lower()
    if not USERNAME.match(username):
        raise ValueError("A username must be 2 to 32 lower-case letters, digits, or . _ -")
    return username


def clean_display_name(name):
    name = " ".join(str(name or "").split())
    if not DISPLAY_NAME.match(name):
        raise ValueError("A display name must be 1 to 40 letters, digits, spaces or .'-")
    return name


def _upgrade(record):
    """An account record from any earlier schema, as the current Account."""
    known = Account.__dataclass_fields__
    return Account(**{k: v for k, v in record.items() if k in known})


def load(path=None):
    """{username: Account}."""
    path = Path(path or ACCOUNTS_PATH)
    if not path.is_file():
        return {}
    data = safe_files.read_json(path)
    return {name: _upgrade(record) for name, record in data.get("accounts", {}).items()}


def _save(accounts, path=None):
    _write(path or ACCOUNTS_PATH, {"schema_version": SCHEMA_VERSION,
                                   "accounts": {name: asdict(a) for name, a in sorted(accounts.items())}})


def get(username, path=None):
    try:
        return load(path).get(clean_username(username))
    except ValueError:
        return None


def new_account(username, display_name, role="listener", password=None, path=None):
    """Make an account and return (account, password). A random password is made when none is given."""
    username, display_name = clean_username(username), clean_display_name(display_name)
    if role not in ROLES:
        raise ValueError(f"The role must be one of: {', '.join(ROLES)}.")
    password = password or new_password()
    problem = password_problem(password)
    if problem:
        raise ValueError(problem)
    with _lock:
        accounts = load(path)
        if username in accounts:
            raise ValueError(f"There is already an account called {username}.")
        if any(a.display_name.lower() == display_name.lower() for a in accounts.values()):
            raise ValueError(f"Another account is already shown as {display_name}; ratings would mix.")
        account = Account(username=username, display_name=display_name, role=role, password=hash_password(password))
        accounts[username] = account
        _save(accounts, path)
    return account, password


def _change(username, change, path=None, sessions_path=None, end_sessions=False):
    username = clean_username(username)
    with _lock:
        accounts = load(path)
        if username not in accounts:
            raise ValueError(f"There is no account called {username}.")
        result = change(accounts, accounts[username])
        _save(accounts, path)
    if end_sessions:
        end_sessions_of(username, sessions_path)
    return result


def set_password(username, password=None, path=None, sessions_path=None):
    """A new password (random when none is given), returned once. Ends the account's sessions."""
    password = password or new_password()
    problem = password_problem(password)
    if problem:
        raise ValueError(problem)

    def change(accounts, account):
        account.password = hash_password(password)
        account.password_changed = _stamp()
        account.failed_logins, account.locked_until = 0, None
    _change(username, change, path, sessions_path, end_sessions=True)
    return password


def set_disabled(username, disabled, path=None, sessions_path=None):
    def change(accounts, account):
        account.disabled = bool(disabled)
    _change(username, change, path, sessions_path, end_sessions=disabled)


def set_role(username, role, path=None, sessions_path=None):
    if role not in ROLES:
        raise ValueError(f"The role must be one of: {', '.join(ROLES)}.")

    def change(accounts, account):
        account.role = role
    _change(username, change, path, sessions_path, end_sessions=True)


def assign(username, batch, pile, path=None):
    """Give an account a listening pile to rate."""
    key = pile_key(batch, pile)

    def change(accounts, account):
        if key not in account.assignments:
            account.assignments.append(key)
    _change(username, change, path)


def unassign(username, batch, pile, path=None):
    key = pile_key(batch, pile)

    def change(accounts, account):
        if key not in account.assignments:
            raise ValueError(f"{account.username} has no pile {key}.")
        account.assignments.remove(key)
    _change(username, change, path)


def remove(username, path=None, sessions_path=None):
    def change(accounts, account):
        del accounts[account.username]
    _change(username, change, path, sessions_path, end_sessions=True)


# --------------------------------------------------------------------------- #
# Logging in, and sessions
# --------------------------------------------------------------------------- #

class LoginRefused(Exception):
    """Wrong username or password, a locked or a disabled account. The message is safe to show."""


def authenticate(username, password, path=None, now=None):
    """
    The Account if the password is right; otherwise LoginRefused. Every refusal
    of a username or password says the same thing, so it does not tell a
    guesser whether the username exists.
    """
    wrong = LoginRefused("Wrong username or password.")
    now = now or _now()
    try:
        username = clean_username(username)
    except ValueError:
        check_password(str(password), _DUMMY_HASH)
        raise wrong
    if not isinstance(password, str) or len(password) > MAX_PASSWORD:
        raise wrong

    with _lock:
        accounts = load(path)
        account = accounts.get(username)
        if account is None:
            check_password(password, _DUMMY_HASH)
            raise wrong
        if account.locked(now):
            raise LoginRefused(f"Too many wrong passwords: this account is locked until {account.locked_until[11:16]}.")
        if not check_password(password, account.password):
            account.failed_logins += 1
            if account.failed_logins >= MAX_FAILURES:
                account.locked_until = _stamp(now + timedelta(minutes=LOCK_MINUTES))
                account.failed_logins = 0
            _save(accounts, path)
            raise wrong
        if account.disabled:
            raise LoginRefused("This account is switched off. Ask the host.")
        account.failed_logins, account.locked_until, account.last_login = 0, None, _stamp(now)
        _save(accounts, path)
    return account


def _token_hash(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _load_sessions(path=None):
    path = Path(path or SESSIONS_PATH)
    return safe_files.read_json(path, default={})


def open_session(account, path=None, now=None):
    """A new session token for this account. Only its hash is kept; expired sessions are cleared out."""
    now = now or _now()
    token = TOKEN_PREFIX + secrets.token_urlsafe(32)
    with _lock:
        sessions = {h: s for h, s in _load_sessions(path).items() if _parse(s["expires"]) > now}
        sessions[_token_hash(token)] = {"username": account.username, "created": _stamp(now),
                                        "expires": _stamp(now + timedelta(hours=SESSION_HOURS))}
        _write(path or SESSIONS_PATH, sessions)
    return token


def session_account(token, path=None, accounts_path=None, now=None):
    """The Account a live session token belongs to, or None (unknown, expired, or the account is off)."""
    if not token or not token.startswith(TOKEN_PREFIX) or len(token) > 200:
        return None
    entry = _load_sessions(path).get(_token_hash(token))
    if entry is None or _parse(entry["expires"]) <= (now or _now()):
        return None
    account = load(accounts_path).get(entry["username"])
    return account if account and not account.disabled else None


def close_session(token, path=None):
    with _lock:
        sessions = _load_sessions(path)
        if sessions.pop(_token_hash(token or ""), None) is not None:
            _write(path or SESSIONS_PATH, sessions)


def end_sessions_of(username, path=None):
    with _lock:
        sessions = _load_sessions(path)
        kept = {h: s for h, s in sessions.items() if s["username"] != username}
        if len(kept) != len(sessions):
            _write(path or SESSIONS_PATH, kept)


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #

def _hand_over(password, username, password_file):
    if password_file:
        with open(password_file, "a", encoding="utf-8") as f:
            f.write(f"{username}\t{password}\n")
        print(f"The password of {username} is in {password_file}: pass it on privately, then delete the file.")
    else:
        print(f"Password for {username} (shown only now; send it to them privately):\n\n    {password}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage who may log in to the Dramatizer API.")
    sub = parser.add_subparsers(dest="action", required=True)
    add = sub.add_parser("add", help="make an account with a random password")
    add.add_argument("username")
    add.add_argument("--name", help="the name shown in the page and with ratings (default: the username)")
    add.add_argument("--role", default="listener", choices=list(ROLES))
    add.add_argument("--password-file", help="append the password to this file instead of printing it")
    reset = sub.add_parser("reset", help="a new random password; ends the account's sessions")
    reset.add_argument("username")
    reset.add_argument("--password-file", help="append the password to this file instead of printing it")
    role = sub.add_parser("role", help="change an account's role")
    role.add_argument("username")
    role.add_argument("role", choices=list(ROLES))
    for name in ("disable", "enable", "remove"):
        sub.add_parser(name).add_argument("username")
    for name, text in (("assign", "give an account a listening pile to rate"), ("unassign", "take a pile away")):
        one = sub.add_parser(name, help=text)
        one.add_argument("username")
        one.add_argument("batch", help="e.g. mos-diverse-v1")
        one.add_argument("pile", type=int)
    sub.add_parser("list", help="who has an account")
    args = parser.parse_args()

    try:
        if args.action == "add":
            account, password = new_account(args.username, args.name or args.username, args.role)
            print(f"Made {account.username} ({account.display_name}, {account.role}).")
            _hand_over(password, account.username, args.password_file)
        elif args.action == "reset":
            _hand_over(set_password(args.username), clean_username(args.username), args.password_file)
        elif args.action == "role":
            set_role(args.username, args.role)
            print(f"{args.username} is now {args.role}; they log in again.")
        elif args.action in ("disable", "enable"):
            set_disabled(args.username, args.action == "disable")
            print(f"{args.username} is {args.action}d.")
        elif args.action == "remove":
            remove(args.username)
            print(f"{args.username} is removed.")
        elif args.action in ("assign", "unassign"):
            (assign if args.action == "assign" else unassign)(args.username, args.batch, args.pile)
            print(f"{args.username}: {', '.join(get(args.username).assignments) or 'no piles'}")
        else:
            accounts = load()
            if not accounts:
                print("No accounts yet: python main_backend.py account add USERNAME --name NAME")
            for a in accounts.values():
                state = "disabled" if a.disabled else ("locked" if a.locked() else "active")
                print(f"  {a.username:16s} {a.display_name:16s} {a.role:11s} {state:9s} last login {a.last_login or 'never'}"
                      f"  piles: {', '.join(a.assignments) or 'any'}")
    except ValueError as error:
        raise SystemExit(str(error))

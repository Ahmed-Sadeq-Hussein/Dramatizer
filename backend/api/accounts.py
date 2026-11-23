"""accounts.py | Who may log in: accounts with a username and password, and the sessions they open.

    python main_backend.py account add julius --name Julius --email j@x.org   # a listener account
    python main_backend.py account add ana --name Ana --role admin        # an account that may do everything
    python main_backend.py account list                                   # who has an account
    python main_backend.py account approve julius                         # let a registered account log in
    python main_backend.py account email julius j@x.org                   # set or change an email
    python main_backend.py account reset julius                           # a new password for julius
    python main_backend.py account disable julius | enable julius         # stop or allow logging in
    python main_backend.py account remove julius                          # delete the account
    python main_backend.py account assign julius mos-diverse-v1 2         # julius rates listening pile 2
    python main_backend.py account unassign julius mos-diverse-v1 2

`add` and `reset` make a random password and show it once. With
--password-file FILE it is written to that file instead of the screen, so it
can be passed on privately. Give each password only to its owner.

Where it is kept: the project's database (backend/storage.py, history/dramatizer.db, never
committed): the accounts, account_piles and sessions tables. Every change is one transaction,
so the API and a command run at the same time cannot overwrite each other. (Until 16 Nov 2026
these were history/api/accounts.json and sessions.json; `python main_backend.py storage migrate`
imports them once.) The `path` argument of the functions below is the database file, for tests.

How passwords are kept:

    - never as text: only a salted PBKDF2-SHA256 hash, 600,000 rounds (the OWASP
      2023 advice for PBKDF2-SHA256), stored as "pbkdf2_sha256$rounds$salt$hash",
      so the method can be made stronger later and old hashes still checked
    - checked in constant time, and a wrong username costs as long as a wrong
      password, so a guesser cannot tell which usernames exist
    - after MAX_FAILURES wrong passwords in a row the account is locked for
      LOCK_MINUTES; logging in successfully resets the count

Registering (POST /register on the website): anyone who reaches the site may make
a member account with a username, display name, email and their own password, and
log in at once. A member may speak and render lines, measure them in the Lab, upload
their own voice and ask the chatbot, within the per-minute limits in the user
agreement (server.py, per_minute). A member does not rate: ratings are the study's
data, so only listeners the host chose rate (`account role NAME listener`), and a
stranger cannot add scores to the results. REGISTRATION=off in .env turns
registering off. The email is not checked by sending a message yet (there is no mail
service); it is for the host to know who registered, and to reach them.

Accounts registered before 23 Nov 2026 were made pending and still need
`account approve` (or `account role`) once.

A login opens a session: a random token that the client sends as
"Authorization: Bearer <token>". Only its SHA-256 hash is stored
(the sessions table), it ends after SESSION_HOURS or at logout, and
resetting, disabling or removing an account ends its sessions at once.

The account record is the model the login system grows from. Its fields:

    username          the login name: lower case, 2-32 letters, digits, . _ -
    display_name      the name shown in the page and saved with ratings
    email             the owner's email, lower case (personal data: shown only to the
                      owner and the host, never logged); None on accounts made before it
    pending           true: registered on the website and not yet approved by the host
    role              what the account may do (ROLES below)
    disabled          true: cannot log in
    created, password_changed, last_login      times, local
    failed_logins, locked_until                the lockout state
    password          the hash described above (never sent anywhere)
    assignments       the listening piles this account is to rate, as "batch|pile"
                      (explore.py makes the piles). A listener with assignments
                      sees and rates only those; with none, every pile.

Adding a field: a new step in storage.MIGRATIONS (a column with a default),
the field here, its column in _to_row/_from_row, and `public` decides whether
the page may see it.
"""
import argparse
import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from backend import storage

SCHEMA_VERSION = 2          # 2: email and pending
ACCOUNTS_PATH = storage.LEGACY["accounts"]    # the old files, read only by `storage migrate`
SESSIONS_PATH = storage.LEGACY["sessions"]

# What each role may do. The API checks permissions, never role names, so a new
# role is one line here.
ROLES = {
    "member": {"speak"},                                    # registered on the website: speak, render, upload, ask
    "listener": {"rate", "speak"},                          # rate genomes as themself; speak and render lines
    "researcher": {"rate", "speak", "search", "name"},      # also make speech, start searches, name genomes
    "admin": {"rate", "speak", "search", "name", "rate_for_others", "remove_renders"},   # also delete renders
}
PERMISSIONS = sorted(set().union(*ROLES.values()))

USERNAME = re.compile(r"^[a-z0-9._-]{2,32}$")
# A practical email check (one @, a dot in the domain, no spaces), not the full RFC 5322 grammar
EMAIL = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
                   r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$")
MAX_EMAIL = 254
DISPLAY_NAME = re.compile(r"^[\w .'-]{1,40}$")      # the same rule as a listener's name in ratings.py
MIN_PASSWORD = 12
MAX_PASSWORD = 200
HASH_NAME = "pbkdf2_sha256"
ROUNDS = 600_000
MAX_FAILURES = 5
LOCK_MINUTES = 15
SESSION_HOURS = 12
TOKEN_PREFIX = "drs_"            # sessions; API keys start with drm_ (keys.py)



def _now():
    return datetime.now().replace(microsecond=0)


def _stamp(time=None):
    return (time or _now()).strftime("%Y-%m-%d %H:%M:%S")


def _parse(stamp):
    return datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S") if stamp else None


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
    email: str | None = None
    pending: bool = False
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
                "assignments": list(self.assignments), "email": self.email}


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


def clean_email(email):
    email = str(email or "").strip()
    if len(email) > MAX_EMAIL or not EMAIL.match(email):
        raise ValueError("Give a valid email address, e.g. name@example.org.")
    return email.lower()


def _upgrade(record):
    """An account record from any earlier schema, as the current Account (v1 had no email or pending)."""
    known = Account.__dataclass_fields__
    account = Account(**{k: v for k, v in record.items() if k in known})
    account.schema_version = SCHEMA_VERSION
    return account


COLUMNS = ("username", "display_name", "role", "password", "disabled", "pending", "email", "created",
           "password_changed", "last_login", "failed_logins", "locked_until", "schema_version")


def _from_row(row, piles):
    record = dict(row)
    record["disabled"], record["pending"] = bool(record["disabled"]), bool(record["pending"])
    return Account(**record, assignments=list(piles))


def _to_row(account):
    return tuple(int(v) if isinstance(v, bool) else v for v in (getattr(account, c) for c in COLUMNS))


def _not_migrated():
    left = [p for p in (ACCOUNTS_PATH, SESSIONS_PATH) if Path(p).is_file()]
    if left:
        raise RuntimeError(f"The accounts are still in {left[0]}: run `python main_backend.py storage migrate` once.")


def load(path=None):
    """{username: Account}."""
    with storage.connect(path) as db:
        piles = {}
        for row in db.execute("SELECT username, pile FROM account_piles ORDER BY username, position"):
            piles.setdefault(row["username"], []).append(row["pile"])
        found = {row["username"]: _from_row(row, piles.get(row["username"], ()))
                 for row in db.execute(f"SELECT {', '.join(COLUMNS)} FROM accounts ORDER BY username")}
    if not found and path is None:
        _not_migrated()
    return found


def _save(accounts, path=None):
    """Make the accounts table hold exactly `accounts` (removed ones are deleted, with their sessions)."""
    with storage.connect(path, write=True) as db:
        keep = list(accounts)
        db.execute(f"DELETE FROM accounts WHERE username NOT IN ({', '.join('?' * len(keep))})", keep)
        for account in accounts.values():
            db.execute(f"INSERT INTO accounts ({', '.join(COLUMNS)}) VALUES ({', '.join('?' * len(COLUMNS))}) "
                       f"ON CONFLICT (username) DO UPDATE SET "
                       + ", ".join(f"{c} = excluded.{c}" for c in COLUMNS[1:]), _to_row(account))
            db.execute("DELETE FROM account_piles WHERE username = ?", (account.username,))
            db.executemany("INSERT INTO account_piles VALUES (?, ?, ?)",
                           [(account.username, pile, i) for i, pile in enumerate(account.assignments)])


def get(username, path=None):
    try:
        return load(path).get(clean_username(username))
    except ValueError:
        return None


def new_account(username, display_name, role="listener", password=None, path=None, email=None, pending=False):
    """Make an account and return (account, password). A random password is made when none is given."""
    username, display_name = clean_username(username), clean_display_name(display_name)
    email = clean_email(email) if email else None
    if role not in ROLES:
        raise ValueError(f"The role must be one of: {', '.join(ROLES)}.")
    password = password or new_password()
    problem = password_problem(password)
    if problem:
        raise ValueError(problem)
    with storage.connect(path, write=True):
        accounts = load(path)
        if username in accounts:
            raise ValueError(f"There is already an account called {username}.")
        if any(a.display_name.lower() == display_name.lower() for a in accounts.values()):
            raise ValueError(f"Another account is already shown as {display_name}; ratings would mix.")
        if email and any(a.email == email for a in accounts.values()):
            raise ValueError("Another account already has that email.")
        account = Account(username=username, display_name=display_name, role=role, password=hash_password(password),
                          email=email, pending=bool(pending))
        accounts[username] = account
        _save(accounts, path)
    return account, password


class RegistrationRefused(Exception):
    """A registration that cannot be made. The message is safe to show."""


def register(username, display_name, email, password, path=None, agreement=None):
    """A member account made on the website: the person's own password, an email required.

    Returns the Account, which may log in at once. A username, display name or email already in use gets one
    and the same answer, so registering does not tell whose email is here.
    """
    if not email:
        raise RegistrationRefused("An email is required.")
    try:
        username, display_name, email = clean_username(username), clean_display_name(display_name), clean_email(email)
    except ValueError as error:
        raise RegistrationRefused(str(error))
    problem = password_problem(password)
    if problem:
        raise RegistrationRefused(problem)
    try:
        with storage.connect(path, write=True):          # the account and its accepted agreement, or neither
            account, _ = new_account(username, display_name, "member", password, path, email=email)
            if agreement:
                accept_agreement(account.username, agreement, how="register", path=path)
    except ValueError:
        raise RegistrationRefused("That username, name or email is already in use. Choose another, or ask the host.")
    return account


# --------------------------------------------------------------------------- #
# The user agreement (agreement.py): which version each account accepted
# --------------------------------------------------------------------------- #

def accept_agreement(username, version, how="login", path=None):
    """Record that `username` accepted agreement `version` (once per version; accepting again changes nothing)."""
    with storage.connect(path, write=True) as db:
        db.execute("INSERT OR IGNORE INTO agreements (username, version, time, how) VALUES (?, ?, ?, ?)",
                   (clean_username(username), str(version), _stamp(), how))


def has_agreed(username, version, path=None):
    """Whether `username` has accepted agreement `version`."""
    with storage.connect(path) as db:
        return db.execute("SELECT 1 FROM agreements WHERE username = ? AND version = ?",
                          (username, str(version))).fetchone() is not None


def agreements_of(username, path=None):
    """Every agreement version `username` accepted: [{"version", "time", "how"}], oldest first."""
    with storage.connect(path) as db:
        return [dict(r) for r in db.execute("SELECT version, time, how FROM agreements WHERE username = ? "
                                            "ORDER BY time", (username,))]


def approve(username, path=None):
    """Let a registered (pending) account log in."""
    def change(accounts, account):
        if not account.pending:
            raise ValueError(f"{account.username} is not waiting for approval.")
        account.pending = False
    _change(username, change, path)


def set_email(username, email, path=None):
    email = clean_email(email)

    def change(accounts, account):
        if any(a.email == email and a.username != account.username for a in accounts.values()):
            raise ValueError("Another account already has that email.")
        account.email = email
    _change(username, change, path)


def _change(username, change, path=None, sessions_path=None, end_sessions=False):
    """Change one account in one transaction. sessions_path: unused (the sessions are in the same database)."""
    username = clean_username(username)
    with storage.connect(path, write=True):
        accounts = load(path)
        if username not in accounts:
            raise ValueError(f"There is no account called {username}.")
        result = change(accounts, accounts[username])
        _save(accounts, path)
        if end_sessions:
            end_sessions_of(username, path)
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

    refused = None
    with storage.connect(path, write=True):
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
            refused = wrong                # raised after the count is committed, not rolled back with it
        elif account.pending:
            raise LoginRefused("This account is waiting for the host's approval.")
        elif account.disabled:
            raise LoginRefused("This account is switched off. Ask the host.")
        else:
            account.failed_logins, account.locked_until, account.last_login = 0, None, _stamp(now)
            _save(accounts, path)
    if refused:
        raise refused
    return account


def _token_hash(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def open_session(account, path=None, now=None):
    """A new session token for this account. Only its hash is kept; expired sessions are cleared out."""
    now = now or _now()
    token = TOKEN_PREFIX + secrets.token_urlsafe(32)
    with storage.connect(path, write=True) as db:
        db.execute("DELETE FROM sessions WHERE expires <= ?", (_stamp(now),))
        db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?)",
                   (_token_hash(token), account.username, _stamp(now), _stamp(now + timedelta(hours=SESSION_HOURS))))
    return token


def session_account(token, path=None, accounts_path=None, now=None):
    """The Account a live session token belongs to, or None (unknown, expired, or the account is off).

    accounts_path: unused (accounts and sessions are in the same database).
    """
    if not token or not token.startswith(TOKEN_PREFIX) or len(token) > 200:
        return None
    with storage.connect(path) as db:
        row = db.execute(f"SELECT {', '.join('a.' + c for c in COLUMNS)} FROM sessions s "
                         "JOIN accounts a ON a.username = s.username WHERE s.token_hash = ? AND s.expires > ?",
                         (_token_hash(token), _stamp(now or _now()))).fetchone()
        if row is None:
            return None
        piles = [r[0] for r in db.execute("SELECT pile FROM account_piles WHERE username = ? ORDER BY position",
                                          (row["username"],))]
    account = _from_row(row, piles)
    return None if account.disabled or account.pending else account


def close_session(token, path=None):
    with storage.connect(path, write=True) as db:
        db.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token or ""),))


def end_sessions_of(username, path=None):
    with storage.connect(path, write=True) as db:
        db.execute("DELETE FROM sessions WHERE username = ?", (username,))


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
    add.add_argument("--email", help="the owner's email")
    add.add_argument("--password-file", help="append the password to this file instead of printing it")
    reset = sub.add_parser("reset", help="a new random password; ends the account's sessions")
    reset.add_argument("username")
    reset.add_argument("--password-file", help="append the password to this file instead of printing it")
    role = sub.add_parser("role", help="change an account's role")
    role.add_argument("username")
    role.add_argument("role", choices=list(ROLES))
    sub.add_parser("approve", help="let a registered account log in").add_argument("username")
    mail = sub.add_parser("email", help="set or change an account's email")
    mail.add_argument("username")
    mail.add_argument("email")
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
            account, password = new_account(args.username, args.name or args.username, args.role, email=args.email)
            print(f"Made {account.username} ({account.display_name}, {account.role}).")
            _hand_over(password, account.username, args.password_file)
        elif args.action == "reset":
            _hand_over(set_password(args.username), clean_username(args.username), args.password_file)
        elif args.action == "approve":
            approve(args.username)
            print(f"{args.username} may log in now.")
        elif args.action == "email":
            set_email(args.username, args.email)
            print(f"{args.username}'s email is changed.")
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
            from backend.api import agreement
            for a in accounts.values():
                agreed = "agreed" if has_agreed(a.username, agreement.VERSION) else "not agreed"
                state ="pending" if a.pending else "disabled" if a.disabled else ("locked" if a.locked() else "active")
                print(f"  {a.username:16s} {a.display_name:16s} {a.role:11s} {state:9s} {a.email or '(no email)':28s} "
                      f"last login {a.last_login or 'never'}  {agreed}"
                      f"  piles: {', '.join(a.assignments) or 'any'}")
    except ValueError as error:
        raise SystemExit(str(error))

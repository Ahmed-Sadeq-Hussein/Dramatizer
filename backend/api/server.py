"""server.py | The back end as a web API: others use it through the client app, the work runs on this PC.

    python main_backend.py api-key add Ana     # one key per person (backend/api/keys.py)
    python main_backend.py api                 # start the API on https://127.0.0.1:5005

    In playit.gg: a TCP tunnel to local address 127.0.0.1, port 5005. Give each
    person the playit address, their key and the fingerprint this prints.

Safe to put behind a tunnel because:

    - every call needs a key or a login session (Authorization: Bearer <key or token>); keys are per
      person and can be removed; passwords are kept only as salted PBKDF2 hashes (accounts.py)
    - it speaks HTTPS with its own certificate (history/api/); the client pins its fingerprint,
      so a tunnel or network in between cannot read or change the traffic
    - it can only do what is listed below: no commands, no file paths from outside, voices by id
    - the GPU work waits in one queue, one job at a time, at most 3 waiting per person
    - only validation voices are offered; final test speakers and custom voices never are
    - every call is logged with who made it (history/api/access.log); logins, refused logins,
      logouts, password changes and saved scores also go to history/api/audit.jsonl
    - an account locks after 5 wrong passwords; past 30 refused logins in 10 minutes (someone
      trying many names) every login waits until it is quiet again; each refusal takes a second
    - replies carry strict browser headers (no framing, no sniffing, no referrer, no caching),
      bodies are capped at 64 KB, and the server does not say what software it runs

The website is served here too, for phones and browsers without the client app
(python main_backend.py tunnel gives it a public https:// address through Cloudflare):

    GET  /, /static/<file>                        the same page as the client app (frontend/client/static)
    GET  /connection                              who is logged in by this browser's cookie, if anyone

A login from that page (its Origin is this server) gets an HttpOnly, Secure, SameSite=Strict
cookie instead of a token, and a change sent with the cookie must come from the page itself.

The API, all under /api/v1 (JSON in and out, audio as WAV):

    GET  /health                                  no key needed: is it up?
    POST /login   {username, password}            no key needed: an account logs in -> {token, account}
    POST /register {username, display_name, email, password}   no key needed: ask for a listener account;
                                                  it waits for the host's approval (account approve)
    POST /logout                                  end this session
    POST /me/password {current, new}              change your own password (ends your sessions)
    GET  /me                                      who this key or session belongs to, and the queue
    GET  /datasets                                the datasets and how many validation voices each has
    GET  /voices?dataset=&q=&page=                validation clips, 50 a page; q matches speaker or words
    GET  /voices/audio?dataset=&clip_id=          a clip's recording
    POST /speak   {dataset, clip_id, text, seed?} queue a line spoken in that voice -> {job}
    GET  /jobs/<id>                               a job's state, place in the queue and result
    GET  /jobs/<id>/audio                         the spoken line, when a speak job is done
    GET  /genome/searches                         every genome search, newest first
    POST /genome/searches {dataset, clip_id, text, genes?, seed?}   queue a random genome search -> {job, run}
    GET  /genome/searches/<run>                   a search: its genomes, verdicts, ratings and ranking
    POST /genome/searches/<run>/continue          queue a stopped search again
    GET  /genome/searches/<run>/audio/<clip>      baseline.wav or genome_NN.wav
    POST /genome/searches/<run>/ratings {scores: {listener: {clip: 0-5}},
                                         artifacts: {listener: {clip: true/false}}}
                                                  save MOS scores, and whether an artifact was heard
    POST /genome/searches/<run>/pick    {genome, name}                    pick and name a genome
    POST /genome/searches/<run>/unpick  {genome}                          take the name off
    GET  /genome/named                            the named genomes
    GET  /genome/library                          every genome to pick from (named, then every search's),
                                                  and what the editor offers (features, limits)
    POST /genome/preview {text, genes}            the genes on a line, in words and as markup (no GPU)
    POST /genome/edited  {name, genes, text?, based_on?}   save a genome edited by hand, under a new name
    POST /genome/edited/remove {name}             delete a genome made by hand
    POST /genome/render  {dataset, clip_id, text, genes, seed?, name?, private_name?}
                                                  (private_name: also keep the genes as the caller's own private genome)
                                                  queue a line spoken with a genome (no genes: the plain
                                                  line) -> {job}; the audio as for /speak
    GET  /genome/listening?batch=&pile=           a listening pile: every clip, your score and the MOS
    GET  /results                                 the baselines' and Milestone 2's summaries
    GET  /evidence/scores?listener=&voice=&dataset=&gene_type=&genes=&genome=&verdict=&pile=&clip=&from=&to=
                         &group_by=&sort=&order=&limit=    the listeners' scores, filtered, sorted or grouped
                                                  (only your own unless you may see others')
    GET  /evidence/questions/<1-6>?from=&to=     one of the six genome questions, answered from its chart
    GET  /evidence/questions/<1-6>/chart          that chart (JPEG)
    GET  /evidence/search?q=&kind=&k=             the evidence records best matching a question (BM25)
    GET  /evidence/answer?q=                      an answer citing its record IDs, or "no evidence" (rag/answer.py)
    GET  /about                                   what the Dramatizer is, how to use it, its contributors
    GET  /assistant                               whether the chatbot's language model is ready
    POST /assistant/ask  {question, history?}     "Ask the Dramatizer": the chatbot (Gemma 4, with your own
                                                  token) or, until it is ready, the records answer
    GET  /renders?q=&mine=&page=                  every line spoken or rendered through the API, newest first:
                                                  voice, line, genome, who made it and when
    GET  /renders/<id>/audio                      one of them
    POST /renders/<id>/rating  {score: 1-5|null}  score your own render (kept apart from the MOS)

Who may do what: an API key may do everything. An account (accounts.py) may do
what its role allows: listeners rate (only as themselves, only their own pile),
see rankings, and speak or render lines; starting searches and naming genomes
need a researcher or admin.
"""
import argparse
import atexit
import hashlib
import json
import queue
import re
import ssl
import threading
import time
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlencode, urlsplit

from flask import Flask, abort, g, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import HTTPException
from werkzeug.serving import WSGIRequestHandler

from backend import model, safe_files
from backend.api import accounts, keys

API = "/api/v1"
VERSION = 1
PORT = int(model.setting("API_PORT", "5005"))
API_DIR = model.PROJECT_DIR / "history" / "api"
CERT_PATH, KEY_PATH = API_DIR / "api_cert.pem", API_DIR / "api_key.pem"
ACCESS_LOG = API_DIR / "access.log"
AUDIT_LOG = API_DIR / "audit.jsonl"
LOGIN_WINDOW_MINUTES, LOGIN_FAILURES_ALLOWED = 10, 30
REGISTER_WINDOW_MINUTES, REGISTRATIONS_ALLOWED = 60, 10      # from anyone: an open form is a target for scripts
SPEAK_DIR = model.OUTPUT_DIR / "api"
DATASETS = ("VCTK", "CREMA-D", "LibriTTS-R")
PAGE_SIZE = 50
MAX_TEXT = 300
MAX_WAITING = 3          # jobs per person, queued or running
IDLE_SECONDS = 30
CLIP_FILE = re.compile(r"^(baseline|genome_\d{2})\.wav$")
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
PAGE_DIR = model.PROJECT_DIR / "frontend" / "client" / "static"     # the website, served here too (phones)
COOKIE = "dramatizer_session"
PAGE_CSP = ("default-src 'self'; img-src 'self' data:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; "
            "frame-ancestors 'none'; form-action 'self'; base-uri 'none'")

app = Flask(__name__, static_folder=None)      # /static is the website (PAGE_DIR), below
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024
app.json.sort_keys = False


# --------------------------------------------------------------------------- #
# Errors as JSON, keys, the access log
# --------------------------------------------------------------------------- #

class Refused(Exception):
    """A request that cannot be done, with the reason shown to the person."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


@app.errorhandler(Refused)
def refused(error):
    return jsonify(error=str(error)), error.status


@app.errorhandler(HTTPException)
def http_error(error):
    return jsonify(error=error.description or error.name), error.code


@app.errorhandler(Exception)
def crashed(error):
    traceback.print_exc()
    return jsonify(error="Something went wrong on the server; it is in the server's log."), 500


_audit_lock = threading.Lock()
_failed_logins = []          # times of recent refused logins, from anyone
_registrations = []          # times of recent registrations, from anyone


def audit(event, **details):
    """One line in the audit log: what happened, when, by whom, from where. Never a password or token."""
    entry = {"time": f"{datetime.now():%Y-%m-%d %H:%M:%S}", "event": event,
             "who": getattr(g, "user", None), "account": getattr(getattr(g, "account", None), "username", None),
             "address": request.remote_addr, **details}
    try:
        with _audit_lock:
            API_DIR.mkdir(parents=True, exist_ok=True)
            with AUDIT_LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _too_many_failed_logins(add=False):
    """Many refused logins from anyone in a short time: someone is guessing names. Wait until it is quiet."""
    now = time.monotonic()
    with _audit_lock:
        _failed_logins[:] = [t for t in _failed_logins if now - t < LOGIN_WINDOW_MINUTES * 60]
        if add:
            _failed_logins.append(now)
        return len(_failed_logins) > LOGIN_FAILURES_ALLOWED


def registration_open():
    """REGISTRATION=off (or 0, no, false) in .env turns registering on the website off."""
    return model.setting("REGISTRATION", "on").strip().lower() not in ("0", "no", "false", "off")


def _too_many_registrations(add=False):
    now = time.monotonic()
    with _audit_lock:
        _registrations[:] = [t for t in _registrations if now - t < REGISTER_WINDOW_MINUTES * 60]
        if add:
            _registrations.append(now)
        return len(_registrations) >= REGISTRATIONS_ALLOWED


def _bearer():
    header = request.headers.get("Authorization", "")
    return header[7:].strip() if header.startswith("Bearer ") else ""


def _token():
    """The caller's key or session: an Authorization header (scripts, the client app) or the page's cookie."""
    return _bearer() or request.cookies.get(COOKIE, "")


def from_page():
    """A request sent by this server's own page: browsers name the page's origin, and it is this server."""
    origin = request.headers.get("Origin")
    return bool(origin) and urlsplit(origin).netloc == request.host


def _same_page_only():
    """
    The cookie goes with every request the browser makes here, so a change sent with it must come from
    this server's own page, never from another website (on top of SameSite=Strict and JSON-only bodies).
    """
    if request.method != "GET" and not _bearer() and request.cookies.get(COOKIE):
        site = request.headers.get("Sec-Fetch-Site")
        if not from_page() or (site and site != "same-origin"):
            raise Refused("Only this site's own page may do that.", 403)


def needs_key(view):
    """
    Who is calling: an API key (keys.py; the host's people, allowed everything) or
    a login session (accounts.py; allowed what the account's role allows).
    Sets g.user (the name kept with ratings), g.account and g.permissions.
    """
    @wraps(view)
    def checked(*args, **kwargs):
        _same_page_only()
        token = _token()
        account = accounts.session_account(token)
        name = account.display_name if account else keys.who(token)
        if name is None:
            time.sleep(1)       # slows down guessing; keys and tokens are far too long to guess anyway
            raise Refused("Log in, or use a valid API key (Authorization: Bearer <key>).", 401)
        g.user, g.account = name, account
        g.permissions = account.permissions if account else set(accounts.PERMISSIONS)
        return view(*args, **kwargs)
    return checked


def needs(permission):
    """A route that only accounts allowed to `permission` may use (after needs_key)."""
    def wrap(view):
        @wraps(view)
        def checked(*args, **kwargs):
            if permission not in g.permissions:
                raise Refused("Your account may not do this. Ask the host.", 403)
            return view(*args, **kwargs)
        return checked
    return wrap


@app.after_request
def log_and_protect(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    page = not request.path.startswith(API + "/") and request.path != "/connection"
    response.headers["Content-Security-Policy"] = PAGE_CSP if page else "default-src 'none'; frame-ancestors 'none'"
    response.headers["Strict-Transport-Security"] = "max-age=31536000"
    response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    try:
        API_DIR.mkdir(parents=True, exist_ok=True)
        with ACCESS_LOG.open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}\t{getattr(g, 'user', '-')}\t{request.method}\t"
                    f"{request.path}\t{response.status_code}\n")
    except OSError:
        pass
    return response


def body():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise Refused("Send a JSON object.")
    return data


def text_of(data, field="text"):
    text = " ".join(str(data.get(field) or "").split())
    if not text:
        raise Refused("Type the line to speak.")
    if len(text) > MAX_TEXT:
        raise Refused(f"The line is longer than {MAX_TEXT} characters.")
    return text


def seed_of(data):
    seed = data.get("seed")
    if seed in (None, ""):
        return None
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        raise Refused("The seed must be a whole number.")
    if not 0 <= seed < 2**31:
        raise Refused("The seed must be from 0 to 2147483647.")
    return seed


# --------------------------------------------------------------------------- #
# The voices: validation speakers of the three datasets only
# --------------------------------------------------------------------------- #

_library = {"clips": None}
_library_lock = threading.Lock()


def library():
    """
    {dataset: {"clips": [...], "by_id": {...}}} of the validation voices, read once (a few seconds).
    Each voice offers one clip: the one the voice manifest chose for it, its longest neutral clip.
    """
    with _library_lock:
        if _library["clips"] is None:
            from backend.voice_data import split_voice_data

            chosen = set(split_voice_data.voice_clips("validation").items())
            out = {name: {"clips": [], "by_id": {}} for name in DATASETS}
            for clip in split_voice_data.read_library():
                if clip.dataset in out and (f"{clip.dataset}/{clip.speaker}", clip.clip_id) in chosen:
                    out[clip.dataset]["clips"].append(clip)
                    out[clip.dataset]["by_id"][clip.clip_id] = clip
            _library["clips"] = out
        return _library["clips"]


def clip_of(dataset, clip_id):
    found = library().get(str(dataset), {}).get("by_id", {}).get(str(clip_id))
    if found is None:
        raise Refused("No such voice. Only validation voices can be used.", 404)
    return found


def clip_json(clip):
    return {"dataset": clip.dataset, "clip_id": clip.clip_id, "speaker": clip.speaker, "text": clip.text,
            "details": {k: str(v) for k, v in clip.details.items()}}


# --------------------------------------------------------------------------- #
# The job queue: one GPU job at a time
# --------------------------------------------------------------------------- #

jobs = {}
waiting = queue.Queue()
jobs_lock = threading.RLock()     # re-entrant: /me holds it while job_json asks place_in_queue, which takes it too


def queue_job(kind, owner, work, about):
    with jobs_lock:
        mine = [j for j in jobs.values() if j["owner"] == owner and j["status"] in ("queued", "running")]
        if len(mine) >= MAX_WAITING:
            raise Refused(f"You already have {MAX_WAITING} jobs waiting: let one finish first.", 429)
        job = {"id": uuid.uuid4().hex, "kind": kind, "owner": owner, "status": "queued", "about": about,
               "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "started": None, "finished": None,
               "result": None, "error": None}
        jobs[job["id"]] = job
    waiting.put((job["id"], work))
    return job


def place_in_queue(job_id):
    with jobs_lock:
        queued = sorted((j for j in jobs.values() if j["status"] == "queued"), key=lambda j: j["created"])
        ids = [j["id"] for j in queued]
    return ids.index(job_id) + 1 if job_id in ids else 0


def worker():
    while True:
        job_id, work = waiting.get()
        job = jobs[job_id]
        job.update(status="running", started=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        try:
            job["result"] = work()
            job["status"] = "done"
        except Exception as error:
            traceback.print_exc()
            job.update(status="failed", error=f"{type(error).__name__}: {error}")
        job["finished"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def job_json(job):
    shown = {k: v for k, v in job.items() if k != "result"}
    shown["place_in_queue"] = place_in_queue(job["id"]) if job["status"] == "queued" else 0
    result = job.get("result") or {}
    if job["kind"] == "speak" and job["status"] == "done":
        shown["audio"] = f"{API}/jobs/{job['id']}/audio"
        shown["seconds"] = result.get("seconds")
    if job["kind"] == "genome":
        shown["run"] = (job.get("about") or {}).get("run")
    return shown


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get(f"{API}/health")
def health():
    return jsonify(ok=True, version=VERSION)


@app.post(f"{API}/login")
def login():
    data = body()
    tried = str(data.get("username") or "")[:40]
    if _too_many_failed_logins():
        audit("login held back", username=tried)
        time.sleep(1)
        raise Refused("Too many wrong logins on this server just now. Wait a few minutes and try again.", 429)
    try:
        account = accounts.authenticate(data.get("username"), data.get("password"))
    except accounts.LoginRefused as error:
        _too_many_failed_logins(add=True)
        audit("login refused", username=tried, reason=str(error))
        time.sleep(1)           # slows down guessing; the account also locks after a few wrong passwords
        raise Refused(str(error), 401)
    g.user, g.account = account.display_name, account
    token = accounts.open_session(account)
    if from_page():
        # The website: the session goes in a cookie its scripts cannot read, sent only over HTTPS and only here
        audit("login", by="page")
        reply = jsonify(account=account.public(), expires_in_hours=accounts.SESSION_HOURS)
        reply.set_cookie(COOKIE, token, max_age=accounts.SESSION_HOURS * 3600, secure=True, httponly=True,
                         samesite="Strict", path="/")
        return reply
    audit("login")
    return jsonify(token=token, account=account.public(), expires_in_hours=accounts.SESSION_HOURS)


@app.get(f"{API}/register")
def registration_state():
    """Whether the website may offer registering."""
    return jsonify(open=registration_open(), needs_approval=True, email_required=True)


@app.post(f"{API}/register")
def register():
    """Ask for a listener account: {username, display_name, email, password}. It waits for the host's approval."""
    if not registration_open():
        raise Refused("Registering is switched off on this server. Ask the host for an account.", 403)
    _same_page_only()
    data = body()
    tried = str(data.get("username") or "")[:40]
    if _too_many_registrations():
        audit("registration held back", username=tried)
        raise Refused("Too many accounts were asked for just now. Try again later.", 429)
    try:
        account = accounts.register(data.get("username"), data.get("display_name"), data.get("email"),
                                    data.get("password"))
    except accounts.RegistrationRefused as error:
        audit("registration refused", username=tried, reason=str(error))      # never the email or password
        time.sleep(0.5)
        raise Refused(str(error), 400)
    _too_many_registrations(add=True)
    g.user = account.display_name
    audit("registered", username=account.username)
    return jsonify(ok=True, username=account.username, pending=True,
                   message="Thank you. The host will approve your account; then you can log in."), 201


@app.post(f"{API}/logout")
@needs_key
def logout():
    accounts.close_session(_token())
    audit("logout")
    reply = jsonify(ok=True)
    reply.delete_cookie(COOKIE, path="/", secure=True, httponly=True, samesite="Strict")
    return reply


@app.post(f"{API}/me/password")
@needs_key
def change_password():
    if g.account is None:
        raise Refused("API keys have no password.", 400)
    data = body()
    try:
        accounts.authenticate(g.account.username, data.get("current"))
    except accounts.LoginRefused:
        audit("password change refused")
        time.sleep(1)
        raise Refused("The current password is wrong.", 403)
    try:
        accounts.set_password(g.account.username, str(data.get("new") or ""))
    except ValueError as error:
        raise Refused(str(error))
    audit("password changed")
    return jsonify(ok=True, message="Password changed: log in again.")


@app.get(f"{API}/me")
@needs_key
def me():
    with jobs_lock:
        mine = [job_json(j) for j in jobs.values() if j["owner"] == g.user]
        busy = sum(j["status"] in ("queued", "running") for j in jobs.values())
    account = g.account.public() if g.account else {"username": None, "display_name": g.user, "role": "api key",
                                                    "permissions": sorted(g.permissions), "last_login": None}
    extras = EXTRAS if private_allowed() else []
    return jsonify(name=g.user, account=account, version=VERSION, jobs_waiting=busy, my_jobs=mine[-20:], extras=extras)


@app.get(f"{API}/datasets")
@needs_key
def datasets():
    """The datasets' validation voices, and the voice split: how many voices the project uses in all."""
    from backend.voice_data.split_voice_data import load_splits
    lib = library()
    split = {name: len(voices) for name, voices in load_splits().items()}
    return jsonify(datasets=[{"name": name, "clips": len(lib[name]["clips"]),
                              "speakers": len({c.speaker for c in lib[name]["clips"]})} for name in DATASETS],
                   voice_split={**split, "total": sum(split.values()),
                                "about": "one clip per voice; validation voices are used for all the work, "
                                         "test voices stay locked until the final evaluation"})


@app.get(f"{API}/voices")
@needs_key
def voices():
    dataset = request.args.get("dataset", DATASETS[0])
    if dataset not in DATASETS:
        raise Refused("Unknown dataset.", 404)
    words = " ".join(request.args.get("q", "").lower().split())[:100]
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1
    clips = library()[dataset]["clips"]
    if words:
        clips = [c for c in clips if words == c.speaker.lower() or words in c.text.lower()]
    start = (page - 1) * PAGE_SIZE
    return jsonify(dataset=dataset, page=page, pages=max(1, -(-len(clips) // PAGE_SIZE)), total=len(clips),
                   clips=[clip_json(c) for c in clips[start:start + PAGE_SIZE]])


@app.get(f"{API}/voices/audio")
@needs_key
def voice_audio():
    from backend.voice_data.Voice_Importer import get_clip

    audio, _ = get_clip(clip_of(request.args.get("dataset"), request.args.get("clip_id")))
    return send_file(audio, mimetype="audio/wav")


@app.post(f"{API}/speak")
@needs_key
@needs("speak")
def speak():
    from backend.voice_data.Voice_Importer import get_clip

    data = body()
    clip = clip_of(data.get("dataset"), data.get("clip_id"))
    text, seed, owner = text_of(data), seed_of(data), g.user

    def work():
        from backend.speech.Voice_model import generate_speech

        reference_audio, reference_text = get_clip(clip)
        started = time.time()
        spoken = generate_speech(reference_audio, reference_text, text,
                                 voice_name=f"api_{clip.dataset}_{clip.speaker}", seed=seed)
        seconds = round(time.time() - started, 1)
        path = keep_render(spoken, clip, text, owner, seconds, seed=seed)
        return {"path": str(path), "seconds": seconds}

    job = queue_job("speak", owner, work, {"voice": clip_json(clip), "text": text, "seed": seed})
    return jsonify(job=job_json(job)), 202


@app.get(f"{API}/jobs/<job_id>")
@needs_key
def job_state(job_id):
    job = jobs.get(job_id) if JOB_ID.match(job_id) else None
    if job is None:
        raise Refused("No such job (jobs are forgotten when the server restarts).", 404)
    return jsonify(job=job_json(job))


@app.get(f"{API}/jobs/<job_id>/audio")
@needs_key
def job_audio(job_id):
    job = jobs.get(job_id) if JOB_ID.match(job_id) else None
    if job is None or job["kind"] != "speak" or job["status"] != "done":
        raise Refused("No audio for that job.", 404)
    return send_file(job["result"]["path"], mimetype="audio/wav", download_name=f"spoken_{job_id[:8]}.wav")


# --- Genome searches ---------------------------------------------------------

def run_dir_of(run_id):
    from backend.genome import search

    try:
        folder = search.run_folder(run_id)
    except ValueError:
        raise Refused("No such search.", 404)
    if not (folder / "run.json").is_file():
        raise Refused("No such search.", 404)
    return folder


def searching(run_id):
    with jobs_lock:
        return any(j["kind"] == "genome" and j["status"] in ("queued", "running")
                   and (j.get("about") or {}).get("run") == run_id for j in jobs.values())


def may_see(run):
    """
    A listener (an account that may not search) sees only the runs of the listening
    piles it may rate: no other searches, no other listener's pile. Researchers,
    admins and API keys see every run.
    """
    if g.account is None or g.account.can("search"):
        return True
    return bool(run.get("batch") and run.get("pile")) and g.account.may_rate_pile(run["batch"], run["pile"])


def sees_others():
    """Whether this caller may see other listeners' scores (and so the MOS). Listeners rate blind."""
    return "rate_for_others" in g.permissions


def own_ratings(rating):
    """A run's ratings as a listener may see them: only their own scores."""
    if sees_others():
        return rating
    mine = rating["scores"].get(g.user)
    return {**rating, "raters": [g.user] if mine else [], "scores": {g.user: mine} if mine else {},
            "given": {g.user: rating.get("given", {}).get(g.user, {})},
            "artifacts": {g.user: rating.get("artifacts", {}).get(g.user, {})}}


def public_run(run):
    """A run without this PC's file paths."""
    shown = {k: v for k, v in run.items() if k not in ("reference_audio",)}
    shown["baseline"] = {k: v for k, v in run.get("baseline", {}).items() if k != "copied_from"}
    return shown


def queue_search(run_id, owner):
    from backend.genome import search

    folder = search.run_folder(run_id)
    return queue_job("genome", owner, lambda: (search.run_search(folder), {"run": run_id})[1], {"run": run_id})


@app.get(f"{API}/genome/searches")
@needs_key
def searches():
    from backend.genome import ratings, search

    # ?pile=N (and ?batch=NAME): only one listener's pile of a listening batch (backend/genome/explore.py)
    pile, batch = request.args.get("pile"), request.args.get("batch")
    runs = [(f, r) for f, r in search.list_runs() if may_see(r)]
    if pile or batch:
        runs = [(f, r) for f, r in runs if r.get("batch") and (not batch or r["batch"] == batch)
                and (not pile or str(r.get("pile")) == pile)]
        runs.sort(key=lambda item: item[1]["voice"])
    out, piles = [], {}
    for folder, run in runs:
        if run.get("batch"):
            piles.setdefault(run["batch"], set()).add(run.get("pile"))
    for folder, run in runs[:100] if not (pile or batch) else runs:
        out.append({"id": run["id"], "voice": run["voice"], "text": run["text"], "created": run["created"],
                    "status": "queued or running" if searching(run["id"]) else run["status"],
                    "summary": search.summary(run), "requested_by": run.get("requested_by"),
                    "rated": ratings.is_rated(folder), "batch": run.get("batch"), "pile": run.get("pile")})
    return jsonify(searches=out, piles={b: sorted(p for p in ps if p) for b, ps in piles.items()})


@app.get(f"{API}/genome/listening")
@needs_key
def listening():
    """
    One listening pile for the logged-in listener: every clip of its runs with
    their own score and the MOS, for the Rate tab and its ranking.
    ?batch=NAME&pile=N; without them, only the piles that exist.
    """
    from backend.genome import ratings, search

    every = [(f, r) for f, r in search.list_runs() if r.get("batch") and r.get("pile")
             and (g.account is None or g.account.may_rate_pile(r["batch"], r["pile"]))]
    piles = {}
    for _, run in every:
        piles.setdefault(run["batch"], set()).add(run["pile"])
    piles = {b: sorted(ps) for b, ps in sorted(piles.items())}
    batch, pile = request.args.get("batch"), request.args.get("pile")
    assigned = g.account.assignments if g.account else []
    if not (batch and pile):
        return jsonify(piles=piles, rater=g.user, assigned=assigned)
    runs = sorted(((f, r) for f, r in every if r["batch"] == batch and str(r["pile"]) == pile
                   and r["status"] == "finished" and not searching(r["id"])), key=lambda item: item[1]["voice"])
    if not runs:
        raise Refused("No such pile.", 404)
    return jsonify(piles=piles, rater=g.user, assigned=assigned, batch=batch, pile=int(pile), max_raters=ratings.MAX_RATERS,
                   sees_others=sees_others(), **ratings.collection(runs, g.user, others=sees_others()))


@app.post(f"{API}/genome/searches")
@needs_key
@needs("search")
def new_search():
    from backend.genome import search
    from backend.voice_data import split_voice_data
    from backend.voice_data.Voice_Importer import get_clip

    data = body()
    clip = clip_of(data.get("dataset"), data.get("clip_id"))
    text = text_of(data)
    try:
        genes = int(data.get("genes") or search.GENES)
    except (TypeError, ValueError):
        raise Refused("Genes per genome must be a whole number.")
    if not 1 <= genes <= search.MAX_OPERATIONS:
        raise Refused(f"Genes per genome must be from 1 to {search.MAX_OPERATIONS}.")
    reference_audio, reference_text = get_clip(clip)
    try:
        folder = search.new_run(reference_audio, reference_text, text=text, voice=clip.speaker, genes=genes,
                                seed=seed_of(data), origin=split_voice_data.voice_origin(clip),
                                method=data.get("method") or "random", starting_genome=data.get("starting_genome"))
    except ValueError as error:
        raise Refused(str(error))
    run = search.load_run(folder)
    run["requested_by"] = g.user
    search.save_run(folder, run)
    job = queue_search(folder.name, g.user)
    return jsonify(job=job_json(job), run=folder.name), 202


@app.get(f"{API}/genome/searches/<run_id>")
@needs_key
def one_search(run_id):
    from backend.genome import ratings, search

    folder = run_dir_of(run_id)
    run = search.load_run(folder)
    if not may_see(run):
        raise Refused("No such search.", 404)
    busy = searching(run_id)
    state = run["status"]
    if state == "running" and not busy:
        state = "stopped"
    rating = own_ratings(ratings.load(folder))
    return jsonify(run=public_run(run), state="queued or running" if busy else state,
                   summary=search.summary(run), ratings={**{k: rating[k] for k in ("raters", "scores", "picked")},
                                                    "artifacts": rating.get("artifacts", {})},
                   ranking=ratings.ranked(folder, run, rating), max_raters=ratings.MAX_RATERS)


@app.post(f"{API}/genome/searches/<run_id>/continue")
@needs_key
@needs("search")
def continue_search(run_id):
    from backend.genome import search

    folder = run_dir_of(run_id)
    if searching(run_id):
        raise Refused("This search is already queued or running.", 409)
    if search.load_run(folder)["status"] == "finished":
        raise Refused("This search has finished.", 409)
    return jsonify(job=job_json(queue_search(run_id, g.user))), 202


@app.get(f"{API}/genome/searches/<run_id>/audio/<name>")
@needs_key
def search_audio(run_id, name):
    if not CLIP_FILE.match(name):
        raise Refused("No such clip.", 404)
    from backend.genome import search

    folder = run_dir_of(run_id)
    if not may_see(search.load_run(folder)):
        raise Refused("No such clip.", 404)
    path = folder / name
    if not path.is_file():
        raise Refused("That clip is not made yet.", 404)
    return send_file(path, mimetype="audio/wav")


def not_while_searching(run_id):
    if searching(run_id):
        raise Refused("This search is still running: rate it when it has finished.", 409)


@app.post(f"{API}/genome/searches/<run_id>/ratings")
@needs_key
@needs("rate")
def rate(run_id):
    from backend.genome import ratings, search

    folder = run_dir_of(run_id)
    not_while_searching(run_id)
    data = body()
    scores, artifacts = data.get("scores") or {}, data.get("artifacts") or {}
    if not isinstance(scores, dict) or not isinstance(artifacts, dict) or not (scores or artifacts):
        raise Refused('Send {"scores": {"listener name": {"genome_01": 4, ...}}, '
                      '"artifacts": {"listener name": {"genome_01": true, ...}}}.')
    if not all(isinstance(clips, dict) for clips in list(scores.values()) + list(artifacts.values())):
        raise Refused("Each listener's scores and artifact answers must be an object of clip: value.")
    if "rate_for_others" not in g.permissions and not set(scores) | set(artifacts) <= {g.user}:
        raise Refused(f"You can only save your own scores, as {g.user}.", 403)
    run = search.load_run(folder)
    if not may_see(run):
        raise Refused("No such search.", 404)
    if g.account and run.get("batch") and not g.account.may_rate_pile(run["batch"], run.get("pile") or 0):
        raise Refused("This run is in another listener's pile.", 403)
    try:
        ratings.save_scores(folder, scores, submitted_by=g.user,
                            account=g.account.username if g.account else None, artifacts=artifacts)
    except (TypeError, ValueError) as error:
        raise Refused(str(error))
    stored = ratings.load(folder)        # read back from the disk, so the page shows only what was really kept
    sent = set(scores) | set(artifacts)
    audit("rated", run=run_id, clips=sum(len(c) for c in scores.values()),
          artifacts=sum(len(c) for c in artifacts.values()))
    shown = own_ratings(stored)
    return jsonify(saved=sum(len(c) for c in scores.values()), raters=shown["raters"],
                   stored={rater: stored["scores"].get(rater, {}) for rater in sent},
                   stored_artifacts={rater: stored.get("artifacts", {}).get(rater, {}) for rater in sent},
                   rated_on={rater: {clip: d.get("time") for clip, d in stored.get("given", {}).get(rater, {}).items()}
                             for rater in sent},
                   ranking=ratings.ranked(folder, ratings=shown))


@app.post(f"{API}/genome/searches/<run_id>/pick")
@needs_key
@needs("name")
def pick(run_id):
    from backend.genome import ratings

    folder = run_dir_of(run_id)
    not_while_searching(run_id)
    data = body()
    try:
        entry = ratings.pick(folder, int(data.get("genome")), data.get("name"), named_by=g.user)
    except (TypeError, ValueError) as error:
        raise Refused(str(error))
    return jsonify(named=entry)


@app.post(f"{API}/genome/searches/<run_id>/unpick")
@needs_key
@needs("name")
def unpick(run_id):
    from backend.genome import ratings

    folder = run_dir_of(run_id)
    try:
        ratings.unpick(folder, int(body().get("genome")))
    except (TypeError, ValueError) as error:
        raise Refused(str(error))
    return jsonify(ok=True)


@app.get(f"{API}/genome/named")
@needs_key
def named():
    from backend.genome import ratings

    return jsonify(named=[{k: v for k, v in e.items() if k != "origin"} for e in ratings.load_named()["genomes"]])


# --- The genome studio: pick, edit, save, render -----------------------------

GENOME_RENDERS = SPEAK_DIR / "genome_renders"      # every render made through the API, each with a .json note
RENDER_ID = re.compile(r"^[A-Za-z0-9_-]{1,160}$")
RENDERS_PAGE = 30
RENDER_RATINGS = SPEAK_DIR / "render_ratings.json"    # people's scores of their own renders, apart from the MOS
PRIVATE_GENOMES = SPEAK_DIR / "private_genomes.json"  # {user: [genome]}: named blank genomes, each seen by its maker only
_own_lock = threading.Lock()


def read_json(path, empty):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return empty


def keep_private(user, name, genes, text, clip):
    """Save (or replace) one of `user`'s private genomes, under `name`."""
    with _own_lock:
        everyone = read_json(PRIVATE_GENOMES, {})
        mine = [e for e in everyone.get(user, []) if e["name"] != name]
        mine.append({"name": name, "genes": list(genes), "text": text, "voice": f"{clip.dataset}/{clip.speaker}",
                     "made": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        everyone[user] = mine
        PRIVATE_GENOMES.parent.mkdir(parents=True, exist_ok=True)
        safe_files.write_json(PRIVATE_GENOMES, everyone)


def keep_render(path, clip, text, owner, seconds, seed=None, genome=None, genes=(), shown=None):
    """
    Keep a finished render in GENOME_RENDERS with a note: which voice, the line, the genome, who made it
    and when, so everyone can listen to it later (GET /renders). A plain /speak line is moved there too.
    """
    path = Path(path)
    GENOME_RENDERS.mkdir(parents=True, exist_ok=True)
    if path.parent != GENOME_RENDERS:
        target = GENOME_RENDERS / f"{clip.dataset}_{clip.speaker}_spoken_{datetime.now():%Y%m%d_%H%M%S_%f}.wav"
        path.replace(target)
        path = target
    shown = shown or {"markup": text, "speed": 1.0}
    safe_files.write_json(path.with_suffix(".json"), {
        "voice": f"{clip.dataset}/{clip.speaker}", "clip_id": clip.clip_id, "text": text, "genome": genome,
        "genes": list(genes), "markup": shown["markup"], "speed": shown["speed"], "seed": seed, "by": owner,
        "made": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "seconds": seconds})
    return path


def genes_of(data, allow_empty=False, usable=True):
    """The request's genes, cleaned. usable=False (a preview) lets an old genome's disabled genes through."""
    from backend.genome import editor

    try:
        return editor.clean_genes(data.get("genes"), allow_empty=allow_empty, usable=usable)
    except ValueError as error:
        raise Refused(str(error))


@app.get(f"{API}/genome/library")
@needs_key
def genome_library():
    from backend.genome import editor, search

    runs = [(f, r) for f, r in search.list_runs() if may_see(r)]
    genomes = editor.library(runs, with_mos=True)       # the MOS orders the list; its number is shown only to those who may
    for e in genomes:
        e["group"] = e.get("voice") or "Made by hand"
    # Grouped by the voice each was found for, the best rated first (unrated last)
    genomes.sort(key=lambda e: (e["group"] == "Made by hand", str(e["group"]),
                                e["mos"] is None, -(e["mos"] or 0)))
    if not sees_others():
        for e in genomes:
            e["mos"] = None
    mine = [{"key": "private:" + e["name"], "source": "private", "name": e["name"], "group": "My genomes",
             "voice": e.get("voice"), "text": e.get("text"), "genes": e["genes"], "described": None, "markup": None,
             "verdict": None, "mos": None, "made": e.get("made"), "unusable": editor.unusable(e["genes"])}
            for e in sorted(read_json(PRIVATE_GENOMES, {}).get(g.user, []), key=lambda e: e.get("made") or "", reverse=True)]
    return jsonify(options=editor.options(), genomes=mine + genomes)


@app.post(f"{API}/genome/preview")
@needs_key
def genome_preview():
    from backend.genome import editor

    data = body()
    try:
        return jsonify(preview=editor.preview(text_of(data), genes_of(data, allow_empty=True, usable=False)))
    except ValueError as error:
        raise Refused(str(error))


@app.post(f"{API}/genome/edited")
@needs_key
@needs("name")
def save_edited():
    from backend.genome import editor

    data = body()
    based_on = data.get("based_on")
    based_on = str(based_on)[:200] if based_on else None
    text = text_of(data) if data.get("text") else None
    try:
        entry = editor.save(data.get("name"), genes_of(data), named_by=g.user, text=text, based_on=based_on)
    except ValueError as error:
        raise Refused(str(error))
    audit("genome saved", name=entry["name"], genes=len(entry["genes"]))
    return jsonify(named=entry)


@app.post(f"{API}/genome/edited/remove")
@needs_key
@needs("name")
def remove_edited():
    from backend.genome import editor

    try:
        editor.remove(body().get("name"))
    except ValueError as error:
        raise Refused(str(error))
    return jsonify(ok=True)


@app.post(f"{API}/genome/render")
@needs_key
@needs("speak")
def render_genome():
    """A line spoken with a genome (or with none: the plain line) in a validation voice -> a speak job."""
    from backend.genome import editor, search
    from backend.voice_data.Voice_Importer import get_clip

    data = body()
    clip = clip_of(data.get("dataset"), data.get("clip_id"))
    text, genes, owner = text_of(data), genes_of(data, allow_empty=True), g.user
    seed = seed_of(data)
    seed = search.render_seed_default() if seed is None else seed
    name = str(data.get("name") or "")[:60] or None
    private = " ".join(str(data.get("private_name") or "").split())[:60] or None
    try:
        shown = editor.preview(text, genes)
    except ValueError as error:
        raise Refused(str(error))
    if private:
        if not genes:
            raise Refused("Add at least one gene to keep the genome.")
        keep_private(owner, private, genes, text, clip)
        name = private

    def work():
        reference_audio, reference_text = get_clip(clip)
        GENOME_RENDERS.mkdir(parents=True, exist_ok=True)
        stem = f"{clip.dataset}_{clip.speaker}_{'genome' if genes else 'plain'}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        path = GENOME_RENDERS / f"{stem}.wav"
        started = time.time()
        editor.render(reference_audio, reference_text, text, genes, path, seed)
        seconds = round(time.time() - started, 1)
        keep_render(path, clip, text, owner, seconds, seed=seed, genome=name, genes=genes, shown=shown)
        return {"path": str(path), "seconds": seconds}

    job = queue_job("speak", owner, work, {"voice": clip_json(clip), "text": text, "seed": seed, "genome": name,
                                           "genes": genes, "markup": shown["markup"]})
    return jsonify(job=job_json(job), preview=shown), 202


# --- Renders: everyone's spoken and rendered lines, to listen to ------------------

def render_notes():
    """Every kept render, newest first: its note plus its id (the file name without .wav)."""
    out = []
    for note in GENOME_RENDERS.glob("*.json") if GENOME_RENDERS.is_dir() else []:
        if not note.with_suffix(".wav").is_file():
            continue
        try:
            data = json.loads(note.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out.append({"id": note.stem, "voice": data.get("voice"), "text": data.get("text"),
                    "genome": data.get("genome"), "genes": len(data.get("genes") or []), "markup": data.get("markup"),
                    "speed": data.get("speed"), "seed": data.get("seed"), "by": data.get("by"),
                    "made": data.get("made"), "seconds": data.get("seconds")})
    return sorted(out, key=lambda r: r["made"] or "", reverse=True)


@app.get(f"{API}/renders")
@needs_key
def renders():
    words = " ".join(request.args.get("q", "").lower().split())[:100]
    mine = request.args.get("mine") in ("1", "true")
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1
    found = render_notes()
    scores = read_json(RENDER_RATINGS, {})
    for r in found:
        r["mine"] = r["by"] == g.user
        r["my_score"] = scores.get(r["id"], {}).get(g.user)
    if mine:
        found = [r for r in found if r["by"] == g.user]
    if words:
        found = [r for r in found if words in " ".join(str(r[k] or "") for k in ("voice", "text", "genome", "by")).lower()]
    start = (page - 1) * RENDERS_PAGE
    return jsonify(renders=found[start:start + RENDERS_PAGE], total=len(found), page=page,
                   pages=max(1, -(-len(found) // RENDERS_PAGE)))


@app.post(f"{API}/renders/<render_id>/rating")
@needs_key
def rate_render(render_id):
    """Score one of your own renders, 1 to 5 (or null to clear). Kept apart from the listening piles' MOS."""
    if not RENDER_ID.match(render_id) or not (GENOME_RENDERS / f"{render_id}.wav").is_file():
        raise Refused("No such render.", 404)
    note = read_json(GENOME_RENDERS / f"{render_id}.json", {})
    if note.get("by") != g.user:
        raise Refused("You can rate only your own renders.", 403)
    score = body().get("score")
    if score is not None and (not isinstance(score, int) or isinstance(score, bool) or not 1 <= score <= 5):
        raise Refused("A score is a whole number from 1 to 5.")
    with _own_lock:
        scores = read_json(RENDER_RATINGS, {})
        if score is None:
            scores.get(render_id, {}).pop(g.user, None)
        else:
            scores.setdefault(render_id, {})[g.user] = score
        RENDER_RATINGS.parent.mkdir(parents=True, exist_ok=True)
        safe_files.write_json(RENDER_RATINGS, scores)
    return jsonify(ok=True, score=score)


@app.get(f"{API}/renders/<render_id>/audio")
@needs_key
def render_audio(render_id):
    path = GENOME_RENDERS / f"{render_id}.wav"
    if not RENDER_ID.match(render_id) or not path.is_file():
        raise Refused("No such render.", 404)
    return send_file(path, mimetype="audio/wav", conditional=True)


# --- Private extensions -----------------------------------------------------------
# Work kept out of git in private/ (see .gitignore). If the folder is there, it adds its own routes, shown
# only to the accounts named in PRIVATE_ACCOUNTS (.env, default "ash"); everyone else gets 404, and /me
# lists the extension (so the page loads it) only for them.
PRIVATE_ACCOUNTS = {name.strip().lower() for name in model.setting("PRIVATE_ACCOUNTS", "ash").split(",") if name.strip()}
EXTRAS = []


def private_allowed():
    return bool(getattr(g, "account", None)) and g.account.username.lower() in PRIVATE_ACCOUNTS


try:
    from private.genome_policy import web as _lab
except ImportError:
    _lab = None
if _lab is not None:
    _lab.register(app, API, needs_key, Refused, private_allowed, clip_of=clip_of)
    EXTRAS.append("lab")


# --- The website ---------------------------------------------------------------
# The same page as the client app (frontend/client), so a phone or any browser can use the host directly,
# through a tunnel with a trusted certificate (python main_backend.py tunnel). Logins keep a cookie.

@app.get("/")
def page():
    return send_from_directory(PAGE_DIR, "index.html")


@app.get("/static/<path:name>")
def page_file(name):
    return send_from_directory(PAGE_DIR, name)       # refuses anything outside PAGE_DIR


@app.get("/connection")
def connection():
    """What the page asks first: who is logged in here (by the cookie), if anyone."""
    account = accounts.session_account(request.cookies.get(COOKIE, ""))
    return jsonify(host=request.host, name=account.display_name if account else None, mode="login")


# --- Evidence: the listeners' scores, the six questions, and the evidence search (for the chatbot) ---------
# Built from the evidence corpus (backend/rag/corpus.py), which reads who_rated_what.csv: rebuilt after every
# saved score, so a cache keyed on that file's time stays current. A caller that may not see other
# listeners' scores (sees_others) gets only their own, and no MOS, no rankings and no charts.

_evidence = {"key": None, "records": None, "index": None, "genomes": None}
_evidence_lock = threading.Lock()
GROUPS = ("listener", "voice", "dataset", "gene_type", "genes", "verdict", "genome", "pile")
SORTS = ("score", "above_plain", "rated_on", "mean", "mean_above_plain", "count", "beat_plain")


def evidence():
    """(corpus records, BM25 index, {run: {genome: (described, markup)}}), rebuilt when the scores change."""
    from backend.genome import ratings, search
    from backend.rag import baseline, corpus

    who = ratings.who_path()
    key = who.stat().st_mtime if who.is_file() else None
    with _evidence_lock:
        if _evidence["records"] is None or _evidence["key"] != key:
            records = corpus.build()
            genomes = {run["id"]: {gm["number"]: (gm.get("described") or [], gm.get("markup")) for gm in run["genomes"]}
                       for _, run in search.list_runs()}
            _evidence.update(key=key, records=records, index=baseline.BM25(records), genomes=genomes)
        return _evidence["records"], _evidence["index"], _evidence["genomes"]


def score_rows():
    """Every score this caller may see, one row per listener per clip."""
    records, _, genomes = evidence()
    rows = []
    for r in records:
        if r["kind"] != "clip":
            continue
        f = r["fields"]
        if not sees_others() and f["listener"] != g.user:
            continue
        described, markup = genomes.get(f["run"], {}).get(f["genome"], ([], None)) if f["genome"] else ([], None)
        rows.append({"listener": f["listener"], "voice": f["voice"], "dataset": f["dataset"], "pile": f["pile"],
                     "search": f["run"], "clip": f["clip"], "genome": f["genome"], "score": f["score"],
                     "above_plain": f["above_plain"], "genes": f["features"], "gene_types": f["gene_types"],
                     "genes_described": described, "spoken_as": markup, "verdict": f["verdict"],
                     "artifact_heard": f["artifact"], "rated_on": r["revision"]})
    return rows


def _matches(row, args):
    def has(name, value):
        return value.lower() in str(row[name]).lower()
    kind = args.get("clip", "genome")
    if kind == "genome" and row["genome"] is None or kind == "plain" and row["genome"] is not None:
        return False
    for name in ("listener", "voice", "dataset", "verdict"):
        if args.get(name) and not has(name, args[name]):
            return False
    if args.get("gene_type") and not any(args["gene_type"].lower() == t.lower() for t in row["gene_types"]):
        return False
    if args.get("genes") not in (None, "") and str(row["genes"]) != str(args["genes"]):
        return False
    if args.get("genome") not in (None, "") and str(row["genome"]) != str(args["genome"]):
        return False
    if args.get("pile") not in (None, "") and str(row["pile"]) != str(args["pile"]):
        return False
    if args.get("search") and args["search"] != row["search"]:
        return False
    if args.get("from") and row["rated_on"] < args["from"] or args.get("to") and row["rated_on"] > args["to"] + "~":
        return False
    return True


def _group(rows, by):
    groups = {}
    for row in rows:
        keys = row["gene_types"] if by == "gene_type" else [f"{row['voice']} genome {row['genome']}" if by == "genome"
                                                           else row[by]]
        for k in keys or ["(none)"]:
            groups.setdefault(str(k), []).append(row)
    out = []
    for k, members in groups.items():
        scores = [m["score"] for m in members]
        above = [m["above_plain"] for m in members if m["above_plain"] is not None]
        out.append({by: k, "count": len(members), "mean": round(sum(scores) / len(scores), 3),
                    "low": min(scores), "high": max(scores),
                    "mean_above_plain": round(sum(above) / len(above), 3) if above else None,
                    "beat_plain": round(sum(a > 0 for a in above) / len(above), 3) if above else None,
                    "listeners": sorted({m["listener"] for m in members})})
    return out


@app.get(f"{API}/evidence/scores")
@needs_key
def evidence_scores():
    """
    The listeners' scores, filtered, sorted and optionally grouped: every small question about who rated
    what. Filters: listener, voice, dataset, gene_type, genes (how many), genome (its number), verdict, pile,
    search, from, to,
    clip=genome|plain|all. group_by: listener|voice|dataset|gene_type|genes|verdict|genome|pile.
    sort: score|above_plain|rated_on (rows), mean|mean_above_plain|count|beat_plain (groups). order, limit.
    """
    args = {k: v.strip()[:80] for k, v in request.args.items()}
    by = args.get("group_by") or None
    if by and by not in GROUPS:
        raise Refused(f"group_by is one of: {', '.join(GROUPS)}.")
    sort = args.get("sort") or ("mean" if by else "score")
    if sort not in SORTS:
        raise Refused(f"sort is one of: {', '.join(SORTS)}.")
    try:
        limit = max(1, min(200, int(args.get("limit") or 20)))
    except ValueError:
        raise Refused("limit is a whole number.")
    rows = [r for r in score_rows() if _matches(r, args)]
    items = _group(rows, by) if by else rows
    if by and sort in ("score", "above_plain", "rated_on"):
        sort = "mean" if sort == "score" else "mean_above_plain" if sort == "above_plain" else "count"
    if not by and sort not in ("score", "above_plain", "rated_on"):
        sort = "score"
    present = [i for i in items if i.get(sort) is not None]
    present.sort(key=lambda i: i[sort], reverse=args.get("order", "desc") != "asc")
    shown = present[:limit] + ([] if len(present) >= limit else [i for i in items if i.get(sort) is None][:limit - len(present)])
    filters = {k: v for k, v in args.items() if k not in ("limit", "order", "sort", "group_by") and v}
    note = None if sees_others() else "Only your own scores: this account rates blind, so it does not see other listeners'."
    return jsonify(filters=filters, group_by=by, sort=sort, total=len(items), rows_matched=len(rows),
                   listeners=sorted({r["listener"] for r in rows}), **{"groups" if by else "rows": shown}, note=note)


_charts = {"key": None, "summary": None}


@app.get(f"{API}/evidence/questions/<int:number>")
@needs_key
def evidence_question(number):
    """One of the six genome questions: the chart's claim and note, made from everyone's scores (from, to: dates)."""
    from backend.genome import evidence_charts

    if not sees_others():
        raise Refused("The six questions are answered from every listener's scores, which this account may not see.", 403)
    if not 1 <= number <= len(evidence_charts.QUESTIONS):
        raise Refused(f"There are {len(evidence_charts.QUESTIONS)} questions, 1 to {len(evidence_charts.QUESTIONS)}.", 404)
    start, end = request.args.get("from") or None, request.args.get("to") or None
    records, _, _ = evidence()
    key = (_evidence["key"], start, end)
    with _evidence_lock:
        if _charts["key"] != key:
            try:
                _charts.update(key=key, summary=evidence_charts.make_charts(start, end, records=records))
            except ValueError as error:
                raise Refused(str(error))
        summary = _charts["summary"]
    chart = next(c for c in summary["charts"] if c["number"] == str(number))
    out = {"number": number, "question": chart["question"], "range": summary["range"],
           "genome_scores": summary["genome_scores"], "sanity_checks": summary["sanity_checks"],
           "sanity_source": summary.get("sanity_source")}
    if "skipped" in chart:
        return jsonify(**out, skipped=chart["skipped"])
    query = "?" + urlencode({k: v for k, v in (("from", start), ("to", end)) if v}) if start or end else ""
    return jsonify(**out, answer=chart["claim"], keep_in_mind=chart["note"],
                   chart=f"{API}/evidence/questions/{number}/chart{query}")


@app.get(f"{API}/evidence/questions/<int:number>/chart")
@needs_key
def evidence_chart(number):
    """The JPEG chart of one question (made by GET /evidence/questions/<n> with the same range)."""
    if not sees_others():
        raise Refused("The six questions are answered from every listener's scores, which this account may not see.", 403)
    evidence_question(number)                   # makes the charts for this range if needed
    chart = next(c for c in _charts["summary"]["charts"] if c["number"] == str(number))
    if "file" not in chart:
        raise Refused("No chart for this range: " + chart.get("skipped", "no data") + ".", 404)
    return send_file(chart["file"], mimetype="image/jpeg")


@app.get(f"{API}/evidence/search")
@needs_key
def evidence_search():
    """The evidence records best matching a question (BM25): scores, searches' rankings, sanity checks."""
    q = " ".join(request.args.get("q", "").split())[:300]
    if not q:
        raise Refused("Ask something: ?q=words")
    kind = request.args.get("kind") or None
    if kind not in (None, "clip", "run", "sanity"):
        raise Refused("kind is clip, run or sanity.")
    try:
        k = max(1, min(20, int(request.args.get("k") or 6)))
    except ValueError:
        raise Refused("k is a whole number.")
    from backend.rag import baseline

    records, index, _ = evidence()
    query = list(dict.fromkeys(baseline.words(q)))
    scored = []
    for i, r in enumerate(records):                    # by position: a clip's ID is shared by its listeners
        if kind not in (None, r["kind"]):
            continue
        if not sees_others() and (r["kind"] == "run" or r["kind"] == "clip" and r["fields"]["listener"] != g.user):
            continue                                   # rankings and others' scores need sees_others
        score = index.score(query, i)
        if score > 0:
            scored.append((score, i))
    scored.sort(key=lambda x: (-x[0], records[x[1]]["id"], x[1]))
    return jsonify(question=q, hits=[{"rank": n, "id": records[i]["id"], "kind": records[i]["kind"],
                                      "title": records[i]["title"], "text": records[i]["text"], "score": round(sc, 3)}
                                     for n, (sc, i) in enumerate(scored[:k], start=1)])


@app.get(f"{API}/evidence/answer")
@needs_key
def evidence_answer():
    """An answer from the evidence that cites its records, or says that nothing answers it (backend/rag/answer.py).
    Made only from the records this caller may see."""
    q = " ".join(request.args.get("q", "").split())[:300]
    if not q:
        raise Refused("Ask something: ?q=words")
    from backend.rag import answer

    records, index, _ = evidence()
    a = answer.answer(q, records, index, listener=None if sees_others() else g.user)
    a.pop("cited_ids", None)
    return jsonify(a)


# --- Ask the Dramatizer: About, and questions answered by the chatbot or the records ---------------------

ABOUT_QUESTION = re.compile(r"contribut|who (made|built|created|wrote|works? on|is behind)|what is (the )?dramatizer"
                            r"|how (do|can|should) (i|we|you)\b|how to use|\bhelp\b|what can (i|you) do",
                            re.IGNORECASE)
ASK_CHARS = 500
DECLINED = "I only answer questions about the Dramatizer"     # how the chatbot starts its off-topic reply (chatbot.OFF_TOPIC)
FACTS_QUESTION = re.compile(r"\b(how many|number of|which|what|list)\b.*\b(voices?|speakers?|datasets?|clips?|listeners?|"
                            r"raters?|data|split)\b", re.IGNORECASE)


def _facts_answer(question):
    """Questions about the data itself (voices, datasets, clips, listeners), counted from the site's own data."""
    from backend.genome import search
    from backend.rag import answer
    from backend.voice_data.split_voice_data import load_splits

    if not FACTS_QUESTION.search(question):
        return None
    splits = load_splits()
    lib = library()
    sets = [(name, len({c.speaker for c in lib[name]["clips"]})) for name in DATASETS]
    records, _, _ = evidence()
    everyone = sees_others()
    clips = [r for r in answer.visible(records, None if everyone else g.user) if r["kind"] == "clip"]
    rated_voices = {r["fields"]["voice"] for r in clips}
    listeners = {r["fields"].get("listener") for r in clips} - {None}
    searched = {run["voice"] for _, run in search.list_runs()}
    text = (f"The project uses {len(splits['validation']) + len(splits['test'])} voices, one clip each, from VCTK, "
            f"CREMA-D and LibriTTS-R: {len(splits['validation'])} validation voices for all the work, and "
            f"{len(splits['test'])} test voices kept locked until the final evaluation. "
            "Validation voices per dataset: " + ", ".join(f"{name} {n}" for name, n in sets) + ". "
            f"Genome searches have been run on {len(searched)} voices. "
            f"Listener scores ({'everyone' if everyone else 'yours'}): {len(clips)} clip scores on "
            f"{len(rated_voices)} voices" + (f", from {len(listeners)} listeners" if everyone else "") + ".")
    return {"answer": text, "source": "records", "route": "facts", "citations": [], "cited": [], "absent": False}


@app.get(f"{API}/about")
@needs_key
def about():
    """What the Dramatizer is, how to use it, and its contributors (backend/api/about.py)."""
    from backend.api.about import ABOUT
    return jsonify(ABOUT)


@app.get(f"{API}/assistant")
@needs_key
def assistant_status():
    """Whether the language model behind "Ask the Dramatizer" is off, idle, starting, ready or failed."""
    from backend.api.assistant import ASSISTANT
    return jsonify(ASSISTANT.status())


def _answer_from_records(question):
    """The answer without the model: About for questions about the project, else the records (answer.py)."""
    from backend.api import about as about_text
    from backend.rag import answer

    if ABOUT_QUESTION.search(question):
        return {"answer": about_text.as_text(), "source": "about", "citations": [], "absent": False}
    records, index, _ = evidence()
    a = answer.answer(question, records, index, listener=None if sees_others() else g.user)
    if a["absent"]:
        facts = _facts_answer(question)        # a question about the data itself, not about the scores
        if facts:
            return facts
    return {"answer": a["answer"], "source": "records", "citations": a["citations"], "cited": a["cited"],
            "absent": a["absent"], "route": a["route"]}


@app.post(f"{API}/assistant/ask")
@needs_key
def assistant_ask():
    """
    Answer a question about the Dramatizer: {question, history?: [{role, content}]}. The chatbot (Gemma 4)
    answers when it is ready, calling this API with the asker's own token; otherwise, or if it fails, the
    records answer (with citations). "source" says which: model, records or about. "note" says why.
    """
    from backend.api.assistant import ASSISTANT

    data = request.get_json(silent=True) or {}
    question = " ".join(str(data.get("question") or "").split())[:ASK_CHARS]
    if not question:
        raise Refused("Ask something.")
    history = [{"role": m.get("role"), "content": str(m.get("content", ""))[:2000]}
               for m in (data.get("history") or [])[-6:] if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
    status = ASSISTANT.status()
    note = None
    if status["state"] == "ready":
        try:
            reply = ASSISTANT.ask(question, _token(), history)
            if str(reply.get("answer", "")).startswith(DECLINED):
                out = _answer_from_records(question)        # declined, but About or the records may still answer it
                if not out.get("absent"):
                    out["note"] = None
                    return jsonify(out)
            return jsonify(answer=reply["answer"], source="model", model=status["model"], tools=reply.get("tools", []),
                           seconds=reply.get("seconds"), citations=[])
        except RuntimeError as error:
            note = f"The language model could not answer ({error}), so this answer comes from the records."
    elif status["state"] in ("idle", "failed"):
        ASSISTANT.start()
        note = ("The language model is starting (about a minute); until then answers come straight from the records."
                if ASSISTANT.status()["state"] == "starting" else None)
    elif status["state"] == "starting":
        note = "The language model is still loading; this answer comes straight from the records."
    out = _answer_from_records(question)
    out["note"] = note
    return jsonify(out)


# --- Results -----------------------------------------------------------------

def summaries(kind):
    """Every OUTPUT/kind/NAME/summary.json, from the project's output/ and the render folder (the first one wins)."""
    found = {}
    for base in (model.PROJECT_DIR / "output", model.OUTPUT_DIR):
        for path in sorted((base / kind).glob("*/summary.json")):
            found.setdefault(path.parent.name, path)
    return [found[name] for name in sorted(found)]


@app.get(f"{API}/results")
@needs_key
def results():
    out = {"baselines": [], "milestone2": []}
    for folder in summaries("baseline"):
        record_path = folder.parent / "record.json"
        record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
        checks_path = folder.parent / "checks.json"
        checks = json.loads(checks_path.read_text(encoding="utf-8")).get("checks", []) if checks_path.is_file() else []
        out["baselines"].append({"name": folder.parent.name, "made": record.get("made"),
                                 "lines": record.get("lines"), "voices": len(record.get("voices") or []),
                                 "git_commit": record.get("git_commit"), "git_dirty": record.get("git_dirty"),
                                 "checks": [{"name": c["name"], "passed": c["passed"]} for c in checks],
                                 "summary": json.loads(folder.read_text(encoding="utf-8"))})
    for folder in summaries("milestone2"):
        out["milestone2"].append({"name": folder.parent.name, "summary": json.loads(folder.read_text(encoding="utf-8"))})
    return jsonify(out)


# --------------------------------------------------------------------------- #
# HTTPS and starting
# --------------------------------------------------------------------------- #

def make_certificate():
    """This API's own self-signed certificate, valid two years. Its fingerprint is what clients pin."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Dramatizer API")])
    now = datetime.now(timezone.utc)
    certificate = (x509.CertificateBuilder()
                   .subject_name(name).issuer_name(name).public_key(key.public_key())
                   .serial_number(x509.random_serial_number())
                   .not_valid_before(now - timedelta(minutes=5)).not_valid_after(now + timedelta(days=730))
                   .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost"),
                                                               x509.IPAddress(ip_address("127.0.0.1"))]), critical=False)
                   .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
                   .sign(key, hashes.SHA256()))
    API_DIR.mkdir(parents=True, exist_ok=True)
    KEY_PATH.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    CERT_PATH.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))


def fingerprint():
    """SHA-256 of the certificate, as the client shows it: AB:CD:..."""
    der = ssl.PEM_cert_to_DER_cert(CERT_PATH.read_text(encoding="ascii"))
    digest = hashlib.sha256(der).hexdigest().upper()
    return ":".join(digest[i:i + 2] for i in range(0, len(digest), 2))


class HandshakeOnItsOwnThread(ssl.SSLContext):
    """The TLS handshake runs on the request's thread, so one silent visitor cannot stop the others."""

    def wrap_socket(self, sock, *args, **kwargs):
        kwargs["do_handshake_on_connect"] = False
        return super().wrap_socket(sock, *args, **kwargs)


class ClosesIdleConnections(WSGIRequestHandler):
    timeout = IDLE_SECONDS
    server_version, sys_version = "Dramatizer", ""      # do not tell the world which Werkzeug and Python run here


def tls_context():
    if not CERT_PATH.is_file() or not KEY_PATH.is_file():
        make_certificate()
    context = HandshakeOnItsOwnThread(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(CERT_PATH, KEY_PATH)
    return context


def main(port=PORT):
    people = list(keys.load()["keys"]) + [f"{a.display_name} (login)" for a in accounts.load().values() if not a.disabled]
    context = tls_context()
    threading.Thread(target=worker, name="gpu jobs", daemon=True).start()
    threading.Thread(target=library, name="read voices", daemon=True).start()
    print(f"""
Dramatizer API on https://127.0.0.1:{port}{API}
  Certificate fingerprint (give it to the people you share with; their client checks it):
    {fingerprint()}
  Who may use it: {len(people)} ({', '.join(people) or 'nobody yet: python main_backend.py api-key add NAME, or account add USERNAME'})
  playit.gg: a TCP tunnel to local address 127.0.0.1, port {port}. Only tunnel this port.
  Stop with Ctrl+C.
""", flush=True)
    from backend.api.assistant import ASSISTANT
    ASSISTANT.configure(f"https://127.0.0.1:{port}", fingerprint())     # where "Ask the Dramatizer"'s model calls back
    atexit.register(ASSISTANT.stop)
    if ASSISTANT._optional("CHATBOT_PRELOAD").strip().lower() not in ("0", "no", "false", "off"):
        ASSISTANT.start()          # load the model now, so the first question does not wait for it (it calls back later)
    app.run(host="127.0.0.1", port=port, threaded=True, ssl_context=context, request_handler=ClosesIdleConnections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The Dramatizer back end as a web API.")
    parser.add_argument("--port", type=int, default=PORT, help=f"default {PORT} (API_PORT in .env)")
    parser.add_argument("--fingerprint", action="store_true", help="print the certificate fingerprint and stop")
    args = parser.parse_args()
    if args.fingerprint:
        tls_context()
        print(fingerprint())
    else:
        main(args.port)

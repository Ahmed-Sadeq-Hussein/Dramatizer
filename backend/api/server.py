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
    - the GPU work waits in one queue, one job at a time, at most 3 waiting per person and MAX_QUEUE in all
      (then 503 + Retry-After); a job not started within MAX_WAIT, or a render whose page was closed, is dropped
    - only validation voices are offered to everyone; final test speakers never are. A custom voice
      (CUSTOM_VOICES, someone's own recorded voice) is offered only to its owner's account (the
      account whose username is the voice's name) and to admins; its renders are listed only to them
      and to whoever made them, and it is never used for genome searches. A voice an account uploads
      (uploaded_voices.py) is a custom voice of that account in the same way
    - an account must accept the user agreement (agreement.py) when registering, and an account that
      has not accepted its current version can do nothing but read and accept it, see itself and log out
    - every call is logged with who made it (history/api/access.log); logins, refused logins,
      logouts, password changes and saved scores also go to history/api/audit.jsonl
    - an account locks after 5 wrong passwords; past 30 refused logins in 10 minutes (someone
      trying many names) every login waits until it is quiet again; each refusal takes a second
    - replies carry strict browser headers (no framing, no sniffing, no referrer, no caching),
      bodies are capped at 64 KB (an uploaded voice: 12 MB), and the server does not say what software it runs

The website is served here too, for phones and browsers without the client app
(python main_backend.py tunnel puts it on https://app.dramatizer.dev through Cloudflare):

    GET  /, /static/<file>                        the same page as the client app (frontend/client/static)
    GET  /connection                              who is logged in by this browser's cookie, if anyone

A login from that page (its Origin is this server) gets an HttpOnly, Secure, SameSite=Strict
cookie instead of a token, and a change sent with the cookie must come from the page itself.

The API, all under /api/v1 (JSON in and out, audio as WAV):

    GET  /health                                  no key needed: is it up?
    POST /login   {username, password}            no key needed: an account logs in -> {token, account}
    POST /register {username, display_name, email, password}   no key needed: make a member account,
                                                  which may log in at once (members do not rate)
    GET  /agreement                               no key needed: the user agreement {version, title, sections}
    POST /register also needs {agreement: <its version>}: the person ticked that they accept it
    POST /logout                                  end this session
    POST /me/agreement {version}                  accept the current user agreement (asked at login until done)
    GET  /me/voices                               the voices this account uploaded, and the upload limits
    POST /me/voices {name, script, filename, audio: base64, consent: true}   upload a voice (wav, mp3, flac or ogg,
                                                  3-30 s, at most 3 per account): only this account (and admins) may use it
    POST /me/voices/remove {folder}               delete one of your uploaded voices from the host
    POST /me/password {current, new}              change your own password (ends your sessions)
    GET  /me                                      who this key or session belongs to, and the queue
    GET  /datasets                                the datasets and how many validation voices each has, and
                                                  "custom": how many custom voices this account may use
    GET  /voices?dataset=&q=&page=                validation clips, 50 a page; q matches speaker or words
                                                  (dataset=Custom voices: the custom voices this account may use)
    GET  /voices/audio?dataset=&clip_id=          a clip's recording
    POST /speak   {dataset, clip_id, text, seed?} queue a line spoken in that voice -> {job}
    GET  /jobs/<id>                               a job's state, place in the queue, expected wait and result
    DELETE /jobs/<id>                             cancel your own job while it is still queued
    GET  /jobs/<id>/audio                         the spoken line, when a speak job is done
    GET  /genome/searches                         every genome search, newest first
    POST /genome/searches {dataset, clip_id, text, genes?, seed?}   queue a random genome search -> {job, run}
    GET  /genome/searches/<run>                   a search: its genomes, verdicts, ratings and ranking
    POST /genome/searches/<run>/continue          queue a stopped search again
    GET  /genome/searches/<run>/audio/<clip>      baseline.wav or genome_NN.wav
    POST /genome/searches/<run>/ratings {scores: {listener: {clip: 0-5}},
                                         artifacts: {listener: {clip: true/false}},
                                         different: {listener: {clip: true/false}}}
                                                  save MOS scores, whether an artifact was heard, and whether
                                                  it sounded like a different person than the original
                                                  (clip: baseline, genome_NN, or the Rate page's blind token)
    GET  /genome/searches/<run>/reference.wav     the voice's original recording, played first on the Rate page
    POST /genome/searches/<run>/pick    {genome, name}                    pick and name a genome
    POST /genome/searches/<run>/unpick  {genome}                          take the name off
    GET  /genome/named                            the named genomes
    GET  /genome/library                          every genome to pick from (named, then every search's),
                                                  and what the editor offers (features, limits)
    POST /genome/preview {text, genes}            the genes on a line, in words and as markup (no GPU)
    POST /genome/edited  {name, genes, text?, based_on?}   save a genome edited by hand, under a new name;
                                                  only the account that made it sees it (and API keys)
    POST /genome/edited/remove {name}             delete a genome you made by hand
    POST /genome/render  {dataset, clip_id, text, genes, seed?, name?, private_name?}
                                                  (private_name: also keep the genes as the caller's own private genome)
                                                  queue a line spoken with a genome (no genes: the plain
                                                  line) -> {job}; the audio as for /speak
    GET  /genome/listening?batch=&pile=           a listening pile, blind: every clip (the plain line mixed in) by
                                                  token, in your own shuffled order, with your own answers
    GET  /results                                 the baselines' and Milestone 2's summaries
    GET  /evidence/scores?listener=&voice=&dataset=&gene_type=&genes=&genome=&verdict=&pile=&clip=&from=&to=
                         &group_by=&sort=&order=&limit=    the listeners' scores, filtered, sorted or grouped
                                                  (only your own unless you may see others')
    GET  /evidence/snapshot?voice=               the analysis the assistant explains: provenance (snapshot_id,
                                                  ratings up to, commit, rule), coverage, per voice a winner
                                                  only with enough evidence, artifact checks, caveats
    GET  /evidence/questions/<1-6>?from=&to=     one of the six genome questions, answered from its chart
    GET  /evidence/questions/<1-6>/chart          that chart (JPEG)
    GET  /evidence/charts/latest                  the six charts as they were last made (the Graphs tab)
    GET  /evidence/charts/latest/<file>.jpg       one of them
    GET  /evidence/mos-study                      the listening studies from assigned piles only: the Report
                                                  evaluation's methods compared (totals only) and the pilot
    GET  /evidence/mos-study/<file>.jpg           one of its charts
    GET  /evidence/search?q=&kind=&k=             the evidence records best matching a question (BM25)
    GET  /evidence/answer?q=                      an answer citing its record IDs, or "no evidence" (rag/answer.py)
                                                  both with provenance {snapshot_id, ratings_up_to, data_revision,
                                                  scope}: the data state the reply was made from (blind accounts:
                                                  their own); search hits carry each record's revision
    GET  /about                                   what the Dramatizer is, how to use it, its contributors
    GET  /glossary?q=                             what a term means (WER, STOI, MOS, genome, plain line...): the best
                                                  matches with how to read the number; without q, every term
    GET  /assistant                               whether the chatbot's language model is ready
    POST /assistant/ask  {question, history?}     "Ask the Dramatizer": the chatbot (Gemma 4, with your own
                                                  token) or, until it is ready, the records answer
    GET  /renders?q=&mine=&page=                  every line spoken or rendered through the API, newest first:
                                                  voice, line, genome, who made it and when
    GET  /renders/<id>                            one of them by its id or short id (render:<time>): its note and genes
    GET  /renders/<id>/audio                      one of them
    POST /renders/<id>/rating  {score: 1-5|null}  score your own render (kept apart from the MOS)
    POST /renders/<id>/name {name}                name your own render (empty: none); its ID never changes
    POST /me/voices/rename {folder, name}         a new name for your uploaded voice; its ID (speaker) never changes
    POST /lab/compare {recordings: [{kind: render, id} | {kind: voice, dataset, clip_id} | {kind: search, run, clip}],
                       reference?: index, measure?: [wer, similarity, quality], text?}
                                                  the Lab: queue measuring up to 8 recordings side by side -> {job};
                                                  the finished job has "lab": WER (and what was heard, word by word),
                                                  speaker similarity to the reference and between every pair, quality
    GET  /lab/results                             your last 20 comparisons
    POST /renders/remove {ids: [...]}             admins (remove_renders): delete renders. They are moved to
                                                  genome_renders/removed/ with who removed them and when, not erased

Who may do what: an API key may do everything. An account (accounts.py) may do
what its role allows: members (registered on the website) speak and render lines; listeners also rate (only as themselves, only their own pile),
see rankings, and speak or render lines; starting searches and naming genomes
need a researcher or admin.
"""
import argparse
import atexit
import hashlib
import json
import math
import os
import queue
import re
import ssl
import threading
import time
import traceback
import uuid
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from functools import wraps
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, g, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import HTTPException
from werkzeug.serving import WSGIRequestHandler

from backend import model, release_info, safe_files
from backend import evidence_views as views
from backend.api import accounts, agreement, keys
from backend.voice_data import uploaded_voices

API = "/api/v1"
VERSION = 1
PORT = int(model.setting("API_PORT", "5005"))
# 127.0.0.1: only this computer (and the tunnel, which connects from here). Inside a container it must be 0.0.0.0
# (deploy/container.env); `docker run -p 127.0.0.1:5005:5005` then keeps it to this computer just the same.
HOST = model.setting("API_HOST", "127.0.0.1")
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
# Limits for everyone together (Milestone 4, step 6). One GPU does one job at a time, so a long queue only means
# long waits: past MAX_QUEUE a new job is refused at once (503 + Retry-After) instead of waiting for ages.
# 4 from the load runs (step 7): about a minute's wait at 13 s a render (12, the first guess, let one wait 4 min).
MAX_QUEUE = int(model.setting("MAX_QUEUE", "4"))                  # jobs queued or running, all people together
MAX_WAIT = int(model.setting("MAX_WAIT_SECONDS", "600"))          # not started by then: dropped, the slot freed
ABANDONED = 120          # a render or Lab job whose page stopped asking about it this long ago is dropped
FOLLOWED = ("speak", "lab")   # their pages ask every 1-2 s; genome searches are started and left to run
TYPICAL_SECONDS = {"speak": 30, "lab": 20, "genome": 900}         # until the server has timed some of its own
# Every account but the admins' (user agreement, section 4): one GPU is shared by everyone
RENDERS_PER_MINUTE, QUESTIONS_PER_MINUTE = 5, 10
UPLOADS_PER_MINUTE = 3          # each one is checked by Whisper on the CPU (uploaded_voices.check_script)
PER_MINUTE = {"render": RENDERS_PER_MINUTE, "question": QUESTIONS_PER_MINUTE, "upload": UPLOADS_PER_MINUTE}
IDLE_SECONDS = 30
CLIP_FILE = re.compile(r"^(baseline|genome_\d{2}|[0-9a-f]{12})\.wav$")     # a blind token on the Rate page
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

from backend.evidence_views import Refused  # noqa: E402  (a request that cannot be done, with the reason)


@app.errorhandler(Refused)
def refused(error):
    reply = jsonify(error=str(error))
    wait = getattr(error, "retry_after", None)
    return (reply, error.status, {"Retry-After": str(wait)}) if wait else (reply, error.status)


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


_per_minute = defaultdict(deque)          # (username, kind) -> when each one was accepted, the last minute


def per_minute(kind, add=False):
    """Refuse (429, with Retry-After) an account that already did PER_MINUTE[kind] of `kind` in the last 60 s.
    Admins and API keys (scripts the host runs) are not limited. add=True counts one, once it was accepted."""
    account = getattr(g, "account", None)
    if account is None or account.role == "admin":
        return
    limit, now = PER_MINUTE[kind], time.monotonic()
    with _audit_lock:
        done = _per_minute[(account.username, kind)]
        while done and now - done[0] >= 60:
            done.popleft()
        if add:
            done.append(now)
            return
        if len(done) >= limit:
            wait = max(1, math.ceil(60 - (now - done[0])))
            what = {"render": f"make {limit} renders", "question": f"ask {limit} questions",
                    "upload": f"upload {limit} voices"}[kind]
            error = Refused(f"You can {what} a minute (see the user agreement). Try again in {wait} s.", 429)
            error.retry_after = wait
            raise error


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
        if account and request.endpoint not in BEFORE_AGREEING and not accounts.has_agreed(account.username,
                                                                                           agreement.VERSION):
            raise Refused("Accept the Dramatizer user agreement first: reload the page to read it.", 403)
        return view(*args, **kwargs)
    return checked


# What an account that has not accepted the current user agreement may still do
BEFORE_AGREEING = {"me", "logout", "accept_agreement", "change_password"}


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
    With VOICE_PACK_DIR set, the clips come from that voice pack (voice_pack.py), not the 24 GB library.
    A server without the voice library (VOICE_LIBRARY_DIR not set or not there: a test machine, a container
    without the voice pack) has no voices, so a request for one is a 404, not a crash. That is not kept, so the
    library is read once it is there.
    """
    with _library_lock:
        if _library["clips"] is None:
            from backend.voice_data import split_voice_data

            out = {name: {"clips": [], "by_id": {}} for name in DATASETS}
            from backend.voice_data import voice_pack

            try:
                chosen = set(split_voice_data.voice_clips("validation").items())
                pack = voice_pack.folder_in_use()
                clips = voice_pack.read(pack) if pack else split_voice_data.read_library()
            except (KeyError, OSError, ValueError, voice_pack.PackError) as error:
                print(f"No voice library: {error}", flush=True)
                return out
            for clip in clips:
                if clip.dataset in out and (f"{clip.dataset}/{clip.speaker}", clip.clip_id) in chosen:
                    out[clip.dataset]["clips"].append(clip)
                    out[clip.dataset]["by_id"][clip.clip_id] = clip
            _library["clips"] = out
        return _library["clips"]


CUSTOM = "Custom voices"


def custom_voices():
    """The custom voices (CUSTOM_VOICES in .env, folders in VOICES_DIR) this account may use: its own, or all for
    an admin. An API key gets none: these are people's own voices, not the project's data."""
    from backend.voice_data.Voice_Importer import read_custom_voices

    account = getattr(g, "account", None)
    if account is None:
        return []
    folder = model.setting("VOICES_DIR", "")
    if not folder or not Path(folder).is_dir():
        return uploaded_voices.clips() if account.role == "admin" else uploaded_voices.clips(account.username)
    every = read_custom_voices(folder)
    if account.role == "admin":
        return every + uploaded_voices.clips()
    return [c for c in every if c.speaker.lower() == account.username.lower()] + uploaded_voices.clips(account.username)


def may_hear_voice(voice):
    """May this account hear renders in this voice ("dataset/speaker")? Every voice but someone else's custom one."""
    if not str(voice or "").startswith(CUSTOM + "/"):
        return True
    return any(f"{c.dataset}/{c.speaker}" == voice for c in custom_voices())


def clip_of(dataset, clip_id):
    if str(dataset) == CUSTOM:
        found = next((c for c in custom_voices() if c.clip_id == str(clip_id)), None)
    else:
        found = library().get(str(dataset), {}).get("by_id", {}).get(str(clip_id))
    if found is None:
        raise Refused("No such voice. Only validation voices, and your own custom voice, can be used.", 404)
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


took = defaultdict(lambda: deque(maxlen=20))     # kind -> seconds the last jobs of that kind took


def seconds_per(kind):
    """How long a job of this kind takes: the median of the last 20, or a first guess."""
    times = sorted(took[kind])
    return times[len(times) // 2] if times else TYPICAL_SECONDS.get(kind, 60)


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def drop(job, status, why):
    job.update(status=status, error=why, finished=now_text())


def expire():
    """Drop the queued jobs that waited too long, or that nobody is waiting for any more. Their slots free at once;
    the worker skips them when it reaches them."""
    now = time.time()
    with jobs_lock:
        for job in jobs.values():
            if job["status"] != "queued":
                continue
            if now - job["_queued"] > MAX_WAIT:
                drop(job, "expired", f"Not started within {MAX_WAIT // 60} minutes: the queue was too long. "
                                     "Try again later.")
            elif job["kind"] in FOLLOWED and now - job["_asked"] > ABANDONED:
                drop(job, "cancelled", "Dropped before it started: its page was closed.")


def wait_seconds(job_id=None):
    """About how long until this queued job starts (no id: until a job sent now would start)."""
    now = time.time()
    with jobs_lock:
        running = [j for j in jobs.values() if j["status"] == "running"]
        queued = sorted((j for j in jobs.values() if j["status"] == "queued"), key=lambda j: j["_queued"])
        ids = [j["id"] for j in queued]
        ahead = queued[:ids.index(job_id)] if job_id in ids else queued
        left = sum(max(5, seconds_per(j["kind"]) - (now - j["_started"])) for j in running)
        return round(left + sum(seconds_per(j["kind"]) for j in ahead))


def until_a_slot():
    """About how long until the full queue has room again: the running job ends, or the oldest queued job is
    dropped for waiting too long, whichever comes first."""
    now = time.time()
    with jobs_lock:
        ends = [max(5, seconds_per(j["kind"]) - (now - j["_started"]))
                for j in jobs.values() if j["status"] == "running"]
        ends += [max(5, MAX_WAIT - (now - j["_queued"])) for j in jobs.values() if j["status"] == "queued"]
    return round(min(ends)) if ends else 5


def in_words(seconds):
    return f"{seconds} s" if seconds < 90 else f"{round(seconds / 60)} min"


def queue_job(kind, owner, work, about):
    expire()
    with jobs_lock:
        active = [j for j in jobs.values() if j["status"] in ("queued", "running")]
        if sum(j["owner"] == owner for j in active) >= MAX_WAITING:
            raise Refused(f"You already have {MAX_WAITING} jobs waiting: let one finish first.", 429)
        if len(active) >= MAX_QUEUE:
            wait = until_a_slot()
            full = Refused(f"The host is busy: {len(active)} jobs are already waiting. Try again in about "
                           f"{in_words(wait)}.", 503)
            full.retry_after = wait
            raise full
        now = time.time()
        job = {"id": uuid.uuid4().hex, "kind": kind, "owner": owner, "status": "queued", "about": about,
               "created": now_text(), "started": None, "finished": None, "result": None, "error": None,
               "_queued": now, "_asked": now, "_started": None}
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
        expire()
        with jobs_lock:
            job = jobs[job_id]
            if job["status"] != "queued":        # cancelled, or dropped for waiting too long
                continue
            job.update(status="running", started=now_text(), _started=time.time())
        try:
            job["result"] = work()
            job["status"] = "done"
            took[job["kind"]].append(time.time() - job["_started"])
        except Exception as error:
            traceback.print_exc()
            job.update(status="failed", error=f"{type(error).__name__}: {error}")
        job["finished"] = now_text()


def job_json(job):
    shown = {k: v for k, v in job.items() if k != "result" and not k.startswith("_")}
    shown["place_in_queue"] = place_in_queue(job["id"]) if job["status"] == "queued" else 0
    shown["expected_start_s"] = wait_seconds(job["id"]) if job["status"] == "queued" else 0
    result = job.get("result") or {}
    if job["kind"] == "speak" and job["status"] == "done":
        shown["audio"] = f"{API}/jobs/{job['id']}/audio"
        shown["seconds"] = result.get("seconds")
    if job["kind"] == "genome":
        shown["run"] = (job.get("about") or {}).get("run")
    if job["kind"] == "lab" and job["status"] == "done":
        shown["lab"] = result.get("lab")
    return shown


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get(f"{API}/health")
def health():
    """Is the process alive? Nothing more (/ready says whether it can do the work)."""
    return jsonify(ok=True, version=VERSION)


STARTED = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
_preload = {"state": "off", "problem": None}     # the voice model loading at start-up (PRELOAD_VOICE_MODEL)


def config_values():
    return {key: os.environ.get(key, "") for key in release_info.CONFIG_KEYS}


@app.get(f"{API}/version")
def version():
    """
    What this process really runs (Milestone 4, step 3), with no key, like /health: its commit, the voice pack and
    model revisions it reads, the models it has loaded so far, and a hash of its settings. Never a path or a secret.
    deploy/smoke.py compares it with the release.json the candidate was released from.
    """
    loaded = [key for key in ("voice", "recogniser", "speaker", "quality", "denoiser") if model.is_loaded(key)]
    return jsonify(api_version=VERSION, started=STARTED, **release_info.describe(config_values(), loaded))


@app.get(f"{API}/ready")
def ready():
    """
    May this process receive traffic? 200 only when the voice model is loaded and the voices are read; otherwise
    503 with what is missing. The chatbot is reported but not required: the records answer until it is ready.
    """
    from backend.api.assistant import ASSISTANT

    voices = _library["clips"]
    count = sum(len(d["clips"]) for d in voices.values()) if voices else 0
    voice_model = ("loaded" if model.is_loaded("voice") else _preload["state"] if _preload["state"] != "off"
                   else "not loaded")
    checks = {"voice_model": voice_model + (f": {_preload['problem']}" if voice_model == "failed" else ""),
              "voices": f"{count} read" if count else "not read",
              "chatbot": ASSISTANT.status()["state"]}
    missing = [name for name, ok in (("voice_model", voice_model == "loaded"), ("voices", count > 0)) if not ok]
    return jsonify(ready=not missing, missing=missing, checks=checks), 503 if missing else 200


def preload_voice_model():
    """Load CosyVoice at start-up, so the first render does not wait for it and /ready can say 200."""
    _preload["state"] = "loading"
    try:
        model.voice_model()
        _preload["state"] = "loaded"
    except Exception as error:          # the server keeps running; /ready says why it is not ready
        traceback.print_exc()
        _preload.update(state="failed", problem=f"{type(error).__name__}: {error}")


def preload_models(assistant=None):
    """The voice model first; then, when given, the chatbot's model (whether or not the voice model loaded)."""
    preload_voice_model()
    if assistant is not None:
        assistant.start()


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
    return jsonify(open=registration_open(), needs_approval=False, email_required=True,
                   limits={"renders_per_minute": RENDERS_PER_MINUTE, "questions_per_minute": QUESTIONS_PER_MINUTE})


@app.post(f"{API}/register")
def register():
    """Make a member account: {username, display_name, email, password}. It may log in at once."""
    if not registration_open():
        raise Refused("Registering is switched off on this server. Ask the host for an account.", 403)
    _same_page_only()
    data = body()
    tried = str(data.get("username") or "")[:40]
    if data.get("agreement") != agreement.VERSION:
        raise Refused("Read the user agreement and tick that you accept it.", 400)
    if _too_many_registrations():
        audit("registration held back", username=tried)
        raise Refused("Too many accounts were asked for just now. Try again later.", 429)
    try:
        account = accounts.register(data.get("username"), data.get("display_name"), data.get("email"),
                                    data.get("password"), agreement=agreement.VERSION)
    except accounts.RegistrationRefused as error:
        audit("registration refused", username=tried, reason=str(error))      # never the email or password
        time.sleep(0.5)
        raise Refused(str(error), 400)
    _too_many_registrations(add=True)
    g.user = account.display_name
    audit("registered", username=account.username, agreement=agreement.VERSION)
    return jsonify(ok=True, username=account.username, pending=False,
                   message="Your account is ready: log in with it."), 201


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
    expire()
    with jobs_lock:
        mine = [job_json(j) for j in jobs.values() if j["owner"] == g.user]
        busy = sum(j["status"] in ("queued", "running") for j in jobs.values())
    account = g.account.public() if g.account else {"username": None, "display_name": g.user, "role": "api key",
                                                    "permissions": sorted(g.permissions), "last_login": None}
    agreed = g.account is None or accounts.has_agreed(g.account.username, agreement.VERSION)
    extras = EXTRAS if private_allowed() and agreed else []
    return jsonify(name=g.user, account=account, version=VERSION, jobs_waiting=busy, my_jobs=mine[-20:], extras=extras,
                   agreement_needed=not agreed, agreement_version=agreement.VERSION)


@app.get(f"{API}/agreement")
def agreement_text():
    """The user agreement, for the register form and for accounts asked to accept it at login."""
    return jsonify(agreement.as_json())


@app.post(f"{API}/me/agreement")
@needs_key
def accept_agreement():
    if g.account is None:
        raise Refused("API keys are given by the host; only accounts accept the user agreement.", 400)
    if body().get("version") != agreement.VERSION:
        raise Refused("That is not the current user agreement: reload the page.", 409)
    accounts.accept_agreement(g.account.username, agreement.VERSION, how="login")
    audit("agreement accepted", version=agreement.VERSION)
    return jsonify(ok=True, version=agreement.VERSION)


# --------------------------------------------------------------------------- #
# Voices an account uploads (uploaded_voices.py): only that account (and admins) may use them
# --------------------------------------------------------------------------- #

UPLOAD_BYTES = 12 * 1024 * 1024        # the JSON body: the audio in base64 is a third larger than the file


@app.before_request
def bigger_body_for_uploads():
    if request.endpoint == "upload_voice":
        request.max_content_length = UPLOAD_BYTES


def _own_account():
    if g.account is None:
        raise Refused("Uploaded voices belong to an account: log in to upload one.", 400)
    return g.account.username


VOICE_FIELDS = ("folder", "name", "speaker", "uploaded", "seconds")


@app.get(f"{API}/me/voices")
@needs_key
def my_voices():
    owner = _own_account()
    mine = uploaded_voices.voices(owner) if uploaded_voices.root() else []
    return jsonify(voices=[{k: v[k] for k in VOICE_FIELDS} for v in mine],
                   limits={"per_account": uploaded_voices.MAX_PER_ACCOUNT, "formats": list(uploaded_voices.FORMATS),
                           "max_mb": uploaded_voices.MAX_BYTES // (1024 * 1024),
                           "seconds": [uploaded_voices.MIN_SECONDS, uploaded_voices.MAX_SECONDS],
                           "script": uploaded_voices.MAX_SCRIPT},
                   available=uploaded_voices.root() is not None)


@app.post(f"{API}/me/voices")
@needs_key
@needs("speak")
def upload_voice():
    import base64
    import binascii

    owner = _own_account()
    per_minute("upload")
    data = body()
    try:
        audio = base64.b64decode(str(data.get("audio") or ""), validate=True)
    except (binascii.Error, ValueError):
        raise Refused("The audio did not arrive whole: upload it again.")
    try:
        about = uploaded_voices.save(owner, data.get("name"), data.get("script"), audio, data.get("filename"),
                                     consent=data.get("consent") is True)
    except uploaded_voices.UploadRefused as error:
        per_minute("upload", add=True)          # a refused upload was still listened to: it counts
        raise Refused(str(error))
    per_minute("upload", add=True)
    audit("voice uploaded", voice=about["speaker"], seconds=about["seconds"], script_wer=about.get("script_wer"))
    return jsonify(ok=True, voice={k: about[k] for k in VOICE_FIELDS}), 201


@app.post(f"{API}/me/voices/remove")
@needs_key
def remove_voice():
    owner = _own_account()
    try:
        about = uploaded_voices.remove(owner, body().get("folder"))
    except uploaded_voices.UploadRefused as error:
        raise Refused(str(error), 404)
    audit("voice deleted", voice=about["speaker"])
    return jsonify(ok=True, removed=about["speaker"])


@app.post(f"{API}/me/voices/rename")
@needs_key
def rename_voice():
    """A new name for one of your uploaded voices: {folder, name}. Its ID (speaker) does not change."""
    owner = _own_account()
    data = body()
    try:
        about = uploaded_voices.rename(owner, data.get("folder"), data.get("name"))
    except uploaded_voices.UploadRefused as error:
        raise Refused(str(error), 404 if "No such" in str(error) else 400)
    audit("voice renamed", voice=about["speaker"], name=about["name"])
    return jsonify(voice={k: about[k] for k in VOICE_FIELDS})


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
                                         "test voices stay locked until the final evaluation"},
                   custom=len(custom_voices()))


@app.get(f"{API}/voices")
@needs_key
def voices():
    dataset = request.args.get("dataset", DATASETS[0])
    if dataset not in DATASETS and dataset != CUSTOM:
        raise Refused("Unknown dataset.", 404)
    words = " ".join(request.args.get("q", "").lower().split())[:100]
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1
    clips = custom_voices() if dataset == CUSTOM else library()[dataset]["clips"]
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

    per_minute("render")
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
    per_minute("render", add=True)
    return jsonify(job=job_json(job)), 202


@app.get(f"{API}/jobs/<job_id>")
@needs_key
def job_state(job_id):
    job = jobs.get(job_id) if JOB_ID.match(job_id) else None
    if job is None:
        raise Refused("No such job (jobs are forgotten when the server restarts).", 404)
    job["_asked"] = time.time()          # someone is still waiting for it (see ABANDONED)
    expire()
    return jsonify(job=job_json(job))


@app.delete(f"{API}/jobs/<job_id>")
@needs_key
def cancel_job(job_id):
    """Take your own job out of the queue. A running job cannot be stopped: the GPU finishes it."""
    with jobs_lock:
        job = jobs.get(job_id) if JOB_ID.match(job_id) else None
        if job is None or job["owner"] != g.user:
            raise Refused("No such job.", 404)
        if job["status"] == "running":
            raise Refused("It is already running and cannot be stopped; it ends on its own.", 409)
        if job["status"] != "queued":
            raise Refused(f"It is already {job['status']}.", 409)
        drop(job, "cancelled", "Cancelled by you.")
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
    admins and API keys see every run. A run with "only" (the Report evaluation, report_eval.py) is seen
    by those accounts alone, admins included, so a pile stays with its two listeners.
    """
    if run.get("only"):
        return g.account is not None and g.account.username in run["only"]
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
            "artifacts": {g.user: rating.get("artifacts", {}).get(g.user, {})},
            "different": {g.user: rating.get("different", {}).get(g.user, {})}}


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
    sums = [search.summary(r) for _, r in runs]
    # The list stops at 100; the totals count every search (the chatbot once added up the 100 and said 485, not 585)
    totals = {"searches": len(runs), "listed": len(out), "voices": len({r["voice"] for _, r in runs}),
              "genomes": sum(s["total"] for s in sums), "passed_checks": sum(s["fine"] for s in sums),
              "terrible": sum(s["terrible"] for s in sums), "failed": sum(s["failed"] for s in sums)}
    return jsonify(searches=out, totals=totals, piles={b: sorted(p for p in ps if p) for b, ps in piles.items()})


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
             and (g.account is None or g.account.may_rate_pile(r["batch"], r["pile"]))
             and (not r.get("only") or may_see(r))]
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
    others = sees_others() and not any(r.get("blind") for _, r in runs)     # a blind pile (second_listen.py): no one
    return jsonify(piles=piles, rater=g.user, assigned=assigned, batch=batch, pile=int(pile),  # sees others' scores
                   max_raters=ratings.MAX_RATERS, sees_others=others, blind=not others and sees_others(),
                   **ratings.collection(runs, g.user, others=others, blind=True))


@app.post(f"{API}/genome/searches")
@needs_key
@needs("search")
def new_search():
    from backend.genome import search
    from backend.voice_data import split_voice_data
    from backend.voice_data.Voice_Importer import get_clip

    data = body()
    if str(data.get("dataset")) == CUSTOM:
        raise Refused("Genome searches use the project's validation voices only, never a custom voice.")
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

    from backend.genome import ratings

    folder = run_dir_of(run_id)
    run = search.load_run(folder)
    if not may_see(run):
        raise Refused("No such clip.", 404)
    try:
        clip = ratings.real_clip(run, name[:-len(".wav")])     # a blind token names the clip without saying which
    except ValueError:
        raise Refused("No such clip.", 404) from None
    path = folder / f"{clip}.wav"
    if not path.is_file():
        raise Refused("That clip is not made yet.", 404)
    return send_file(path, mimetype="audio/wav", download_name=name)


@app.get(f"{API}/genome/searches/<run_id>/reference.wav")
@needs_key
def search_reference(run_id):
    """The voice's original recording the run was cloned from: the Rate page plays it first, so the listener can
    tell whether each clip sounds like the same person."""
    from backend.genome import ratings, search

    run = search.load_run(run_dir_of(run_id))
    if not may_see(run):
        raise Refused("No such recording.", 404)
    path = ratings.reference_path(run)
    if path is None:
        raise Refused("The original recording of this voice is not on the host.", 404)
    return send_file(path, mimetype="audio/wav", download_name="original.wav")


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
    scores, artifacts, different = data.get("scores") or {}, data.get("artifacts") or {}, data.get("different") or {}
    if (not all(isinstance(x, dict) for x in (scores, artifacts, different))
            or not (scores or artifacts or different)):
        raise Refused('Send {"scores": {"listener name": {"genome_01": 4, ...}}, '
                      '"artifacts": {"listener name": {"genome_01": true, ...}}, '
                      '"different": {"listener name": {"genome_01": false, ...}}}.')
    if not all(isinstance(clips, dict) for part in (scores, artifacts, different) for clips in part.values()):
        raise Refused("Each listener's scores and answers must be an object of clip: value.")
    if "rate_for_others" not in g.permissions and not set(scores) | set(artifacts) | set(different) <= {g.user}:
        raise Refused(f"You can only save your own scores, as {g.user}.", 403)
    run = search.load_run(folder)
    if not may_see(run):
        raise Refused("No such search.", 404)
    if g.account and run.get("batch") and not g.account.may_rate_pile(run["batch"], run.get("pile") or 0):
        raise Refused("This run is in another listener's pile.", 403)
    try:
        ratings.save_scores(folder, scores, submitted_by=g.user,
                            account=g.account.username if g.account else None, artifacts=artifacts,
                            different=different)
    except (TypeError, ValueError) as error:
        raise Refused(str(error))
    stored = ratings.load(folder)        # read back from the disk, so the page shows only what was really kept
    sent = set(scores) | set(artifacts) | set(different)
    audit("rated", run=run_id, clips=sum(len(c) for c in scores.values()),
          artifacts=sum(len(c) for c in artifacts.values()), different=sum(len(c) for c in different.values()))
    shown = own_ratings(stored)
    # Sent by blind token (the Rate page): answer in tokens too, and without the ranking, which would name the clips
    blind = any(ratings.TOKEN.match(str(clip)) for part in (scores, artifacts, different)
                for clips in part.values() for clip in clips)
    name = (lambda clip: ratings.clip_token(run_id, clip)) if blind else (lambda clip: clip)
    each = lambda part: {rater: {name(c): v for c, v in stored.get(part, {}).get(rater, {}).items()} for rater in sent}
    out = dict(saved=sum(len(c) for c in scores.values()), raters=shown["raters"],
               stored=each("scores"), stored_artifacts=each("artifacts"), stored_different=each("different"),
               rated_on={rater: {name(clip): d.get("time") for clip, d in stored.get("given", {}).get(rater, {}).items()}
                         for rater in sent})
    if not blind:
        out["ranking"] = ratings.ranked(folder, ratings=shown)
    return jsonify(out)


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

    from backend.genome import editor

    return jsonify(named=[{k: v for k, v in e.items() if k not in ("origin", "account")}
                          for e in ratings.load_named()["genomes"] if editor.owns(e, genome_viewer())])


def genome_viewer():
    """Who is asking, for genomes made by hand (editor.owns): (username, display name), or None for an API key."""
    return (g.account.username, g.account.display_name) if g.account else None


# --- The genome studio: pick, edit, save, render -----------------------------

GENOME_RENDERS = SPEAK_DIR / "genome_renders"      # every render made through the API, each with a .json note
# A render's file name without .wav. Spaces are allowed: a custom voice's renders start with its dataset name,
# "Custom voices_<speaker>_...". Never a dot or a slash, so an id cannot leave GENOME_RENDERS.
RENDER_ID = re.compile(r"^[A-Za-z0-9_-][A-Za-z0-9 _-]{0,159}$")
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
    genomes = editor.library(runs, with_mos=True, viewer=genome_viewer())       # the MOS orders the list; its number is shown only to those who may
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
        entry = editor.save(data.get("name"), genes_of(data), named_by=g.user, text=text, based_on=based_on,
                            account=g.account.username if g.account else None)
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
        editor.remove(body().get("name"), viewer=genome_viewer())
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

    per_minute("render")
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
    per_minute("render", add=True)
    return jsonify(job=job_json(job), preview=shown), 202


# --- Renders: everyone's spoken and rendered lines, to listen to ------------------

SHORT_RENDER_ID = re.compile(r"\d{8}_\d{6}_\d{6}$")      # the time a render was made, to the microsecond


def short_render_id(render_id):
    """What the page's Copy ID copies after "render:": the time in its name (unique, no spaces), else the whole id."""
    found = SHORT_RENDER_ID.search(render_id)
    return found.group(0) if found else render_id


def find_render(given):
    """A render's full id from what someone pasted: "render:<short or full id>", the full id or its short id.
    None when it matches no render, or more than one."""
    given = " ".join(str(given or "").split())
    if given.lower().startswith("render:"):
        given = given[len("render:"):].strip()
    if not given or len(given) > 160:
        return None
    if RENDER_ID.match(given) and (GENOME_RENDERS / f"{given}.wav").is_file():
        return given
    if not GENOME_RENDERS.is_dir():
        return None
    if len(given) < 6:                      # "render:1" names nothing, though some id ends in 1
        return None
    found = [w.stem for w in GENOME_RENDERS.glob("*.wav")         # a whole tail of the id: after a "_"
             if w.stem.endswith("_" + given) and RENDER_ID.match(w.stem)]
    return found[0] if len(found) == 1 else None


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
        out.append({"id": note.stem, "short_id": short_render_id(note.stem), "name": data.get("name"),
                    "voice": data.get("voice"),
                    "clip_id": data.get("clip_id"), "text": data.get("text"), "genome": data.get("genome"),
                    "genes": len(data.get("genes") or []), "gene_list": data.get("genes") or [], "markup": data.get("markup"),
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
    found = [r for r in render_notes() if r["by"] == g.user or may_hear_voice(r["voice"])]
    scores = read_json(RENDER_RATINGS, {})
    for r in found:
        r["mine"] = r["by"] == g.user
        r["renamable"] = r["mine"]
        r["my_score"] = scores.get(r["id"], {}).get(g.user)
    for r in found:
        r["removable"] = "remove_renders" in g.permissions
    if mine:
        found = [r for r in found if r["by"] == g.user]
    if words:
        if words.startswith("render:"):
            words = words[len("render:"):].strip()
        found = [r for r in found if words in " ".join(str(r[k] or "") for k in ("voice", "text", "genome", "by", "id", "name")).lower()]
    start = (page - 1) * RENDERS_PAGE
    return jsonify(renders=found[start:start + RENDERS_PAGE], total=len(found), page=page,
                   pages=max(1, -(-len(found) // RENDERS_PAGE)))


@app.get(f"{API}/renders/<given>")
@needs_key
def one_render(given):
    """One render by its id or short id ("render:20261119_220500_543947"): voice, line, genome and its genes."""
    render_id = find_render(given)
    note = read_json(GENOME_RENDERS / f"{render_id}.json", {}) if render_id else {}
    if not render_id or (note.get("by") != g.user and not may_hear_voice(note.get("voice"))):
        raise Refused("No such render. Copy its ID from Listen > Renders.", 404)
    return jsonify(render={"id": render_id, "short_id": short_render_id(render_id), **note,
                           "audio": f"{API}/renders/{render_id}/audio"})


RENDER_NAME_CHARS = 60


@app.post(f"{API}/renders/<given>/name")
@needs_key
def rename_render(given):
    """Name one of your own renders: {name} (empty: no name). Its ID, the file name, does not change."""
    render_id = find_render(given)
    if not render_id:
        raise Refused("No such render.", 404)
    note_path = GENOME_RENDERS / f"{render_id}.json"
    with _own_lock:
        note = read_json(note_path, {})
        if note.get("by") != g.user:
            raise Refused("You can rename only the renders you made.", 403)
        name = " ".join(str(body().get("name") or "").split())
        if len(name) > RENDER_NAME_CHARS:
            raise Refused(f"A render's name is at most {RENDER_NAME_CHARS} characters.")
        if name:
            note["name"] = name
        else:
            note.pop("name", None)
        safe_files.write_json(note_path, note)
    audit("render renamed", render=render_id, name=name or None)
    return jsonify(id=render_id, short_id=short_render_id(render_id), name=name or None)


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


REMOVED_RENDERS = "removed"          # a sub-folder of GENOME_RENDERS: render_notes() does not look inside it
MAX_REMOVE = 200


@app.post(f"{API}/renders/remove")
@needs_key
@needs("remove_renders")
def remove_renders():
    """Admins delete renders: each .wav and its note are moved to GENOME_RENDERS/removed/, the note recording who
    removed it and when, so a mistake can be undone by moving the two files back."""
    ids = body().get("ids")
    if not isinstance(ids, list) or not ids or len(ids) > MAX_REMOVE:
        raise Refused(f"Send ids: a list of 1 to {MAX_REMOVE} render ids.")
    ids = list(dict.fromkeys(str(i) for i in ids))
    bad = [i for i in ids if not RENDER_ID.match(i) or not (GENOME_RENDERS / f"{i}.wav").is_file()]
    if bad:
        raise Refused(f"No such render: {', '.join(bad[:5])}.", 404)
    bin_dir = GENOME_RENDERS / REMOVED_RENDERS
    bin_dir.mkdir(parents=True, exist_ok=True)
    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    removed = []
    with _own_lock:
        for render_id in ids:
            wav, note = GENOME_RENDERS / f"{render_id}.wav", GENOME_RENDERS / f"{render_id}.json"
            data = read_json(note, {})
            data.update(removed_by=g.user, removed=when)
            safe_files.write_json(bin_dir / f"{render_id}.json", data)
            wav.replace(bin_dir / f"{render_id}.wav")
            note.unlink(missing_ok=True)
            removed.append(render_id)
    audit("renders removed", count=len(removed), ids=removed[:20])
    return jsonify(ok=True, removed=removed)


@app.get(f"{API}/renders/<render_id>/audio")
@needs_key
def render_audio(render_id):
    path = GENOME_RENDERS / f"{render_id}.wav"
    if not RENDER_ID.match(render_id) or not path.is_file():
        raise Refused("No such render.", 404)
    note = read_json(path.with_suffix(".json"), {})
    if note.get("by") != g.user and not may_hear_voice(note.get("voice")):
        raise Refused("No such render.", 404)                 # someone else's custom voice: not even its existence
    return send_file(path, mimetype="audio/wav", conditional=True)


# --- The Lab: measure recordings side by side (backend/metrics/lab.py) -------------------------

LAB_RESULTS = SPEAK_DIR / "lab_results.json"        # {user: [comparison]}: each person's last comparisons
LAB_KEEP = 20
SEARCH_CLIP = re.compile(r"^(baseline|genome_\d{2}|reference)$")


def lab_source(source):
    """One recording the caller may hear, from the request -> {"label", "text", "path" or "clip"}, never a file
    path from outside: a render by id, a voice's original by dataset and clip id, or (accounts that may search)
    a genome search's clip."""
    if not isinstance(source, dict):
        raise Refused("Each recording is an object with a kind: render, voice or search.")
    kind = source.get("kind")
    if kind == "render":
        render_id = find_render(source.get("id")) or str(source.get("id") or "")
        path = GENOME_RENDERS / f"{render_id}.wav"
        if not RENDER_ID.match(render_id) or not path.is_file():
            raise Refused("No such render.", 404)
        note = read_json(path.with_suffix(".json"), {})
        if note.get("by") != g.user and not may_hear_voice(note.get("voice")):
            raise Refused("No such render.", 404)
        what = note.get("name") or note.get("genome") or ("Edited genome" if note.get("genes") else "Plain line")
        return {"label": f"{what} · {note.get('voice') or '?'} · by {note.get('by') or '?'}", "text": note.get("text"),
                "path": path, "source": {"kind": "render", "id": render_id}}
    if kind == "voice":
        clip = clip_of(source.get("dataset"), source.get("clip_id"))
        return {"label": f"Original · {clip.dataset}/{clip.speaker}", "text": clip.text, "clip": clip,
                "source": {"kind": "voice", "dataset": clip.dataset, "clip_id": clip.clip_id}}
    if kind == "search":
        if g.account is not None and not g.account.can("search"):
            raise Refused("Only researchers and admins measure a search's clips.", 403)
        from backend.genome import ratings, search

        run_id, name = str(source.get("run") or ""), str(source.get("clip") or "")
        run = search.load_run(run_dir_of(run_id))
        if not may_see(run) or not SEARCH_CLIP.match(name) or run.get("blind") or run.get("sealed"):
            raise Refused("No such clip.", 404)
        path = ratings.reference_path(run) if name == "reference" else run_dir_of(run_id) / f"{name}.wav"
        if path is None or not path.is_file():
            raise Refused("That clip is not on the host.", 404)
        text = run.get("reference_text") if name == "reference" else run.get("text")
        shown = {"reference": "original", "baseline": "plain line"}.get(name) or f"genome {int(name[len('genome_'):])}"
        return {"label": f"{run['voice']} · {shown} · search {run_id[:20]}", "text": text,
                "path": path, "source": {"kind": "search", "run": run_id, "clip": name}}
    raise Refused("A recording's kind is render, voice or search.")


@app.post(f"{API}/lab/compare")
@needs_key
@needs("speak")
def lab_compare():
    """Queue a comparison: {recordings: [source], reference?: index, measure?: [wer, similarity, quality],
    text?: one script for every recording} -> {job}; the job's result holds the measurements."""
    from backend.metrics import lab

    data = body()
    sources = data.get("recordings")
    if not isinstance(sources, list) or not 1 <= len(sources) <= lab.MAX_RECORDINGS:
        raise Refused(f"Choose 1 to {lab.MAX_RECORDINGS} recordings.")
    picked = [lab_source(s) for s in sources]
    reference = data.get("reference")
    if reference is not None and (not isinstance(reference, int) or isinstance(reference, bool)
                                  or not 0 <= reference < len(picked)):
        raise Refused("The reference must be one of the chosen recordings.")
    measure = data.get("measure") or ["wer", "similarity", "quality"]
    if not isinstance(measure, list) or not measure or any(m not in ("wer", "similarity", "quality") for m in measure):
        raise Refused("measure is a list of: wer, similarity, quality.")
    script = " ".join(str(data.get("text") or "").split())[:500] or None
    owner = g.user

    def work():
        from backend.voice_data.Voice_Importer import get_clip

        recordings = []
        for p in picked:
            path = p["path"] if "path" in p else get_clip(p["clip"])[0]
            recordings.append({"label": p["label"], "path": path, "text": script or p["text"]})
        result = lab.compare(recordings, reference=reference, measure=measure)
        for row, p in zip(result["recordings"], picked):
            row["source"] = p["source"]
        result.update(by=owner, script=script)
        with _own_lock:
            everyone = read_json(LAB_RESULTS, {})
            everyone[owner] = ([result] + everyone.get(owner, []))[:LAB_KEEP]
            LAB_RESULTS.parent.mkdir(parents=True, exist_ok=True)
            safe_files.write_json(LAB_RESULTS, everyone)
        return {"lab": result}

    job = queue_job("lab", owner, work, {"recordings": [p["label"] for p in picked], "reference": reference,
                                         "measure": measure})
    return jsonify(job=job_json(job)), 202


@app.get(f"{API}/lab/results")
@needs_key
def lab_results():
    """Your last comparisons, newest first."""
    return jsonify(results=read_json(LAB_RESULTS, {}).get(g.user, []))


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
GROUPS, SORTS = views.GROUPS, views.SORTS


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


_snapshot = {"key": None, "snapshot": None}


def evidence_snapshot():
    """The evidence snapshot (backend/genome/evidence_snapshot.py), rebuilt when the scores change."""
    from backend.genome import evidence_snapshot as E
    from backend.genome import ratings

    who = ratings.who_path()
    key = who.stat().st_mtime if who.is_file() else None
    with _evidence_lock:
        if _snapshot["snapshot"] is None or _snapshot["key"] != key:
            _snapshot.update(key=key, snapshot=E.build())
        return _snapshot["snapshot"]


def _ratings_stamp():
    from backend.genome import ratings

    who = ratings.who_path()
    return who.stat().st_mtime if who.is_file() else None


def evidence_with_provenance():
    """(records, index, genomes, provenance): the corpus and the snapshot made from one state of the ratings,
    so the provenance names the data an answer was made from. Retried if the scores change in between.

    provenance: snapshot_id and ratings_up_to as GET /evidence/questions gives them, and data_revision: the
    newest revision of the evidence records the caller may see (scores, rankings, sanity checks), which moves
    when any of them changes. A caller who rates blind sees only their own scores, so they get no snapshot_id
    (it is made from everyone's), and ratings_up_to and data_revision come from their own records."""
    for _ in range(3):
        stamp = _ratings_stamp()
        records, index, genomes = evidence()
        snap = evidence_snapshot() if sees_others() else None
        if _ratings_stamp() == stamp:
            break
    return records, index, genomes, views.provenance(records, snap, viewer())


def viewer():
    """None for a caller who may see every listener's scores, else the one listener whose scores they see."""
    return None if sees_others() else g.user


def may_read_record(record):
    """Whether this caller may read an evidence record: rankings and others' scores need sees_others."""
    return views.may_read(record, viewer())


def score_rows():
    """Every score this caller may see, one row per listener per clip."""
    records, _, genomes = evidence()
    return views.score_rows(records, genomes, viewer())


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
    out = views.scores(score_rows(), dict(request.args))
    note = None if sees_others() else "Only your own scores: this account rates blind, so it does not see other listeners'."
    return jsonify(**out, note=note)


_charts = {"key": None, "summary": None}


@app.get(f"{API}/evidence/snapshot")
@needs_key
def evidence_snapshot_route():
    """What the analysis concluded and how sure it is: the assistant retrieves this, it does not decide."""
    from backend.genome import evidence_snapshot as E

    if not sees_others():
        raise Refused("The snapshot is built from every listener's scores, which this account may not see.", 403)
    voice = (request.args.get("voice") or "").strip() or None
    out = E.summary(evidence_snapshot(), voice)
    if voice and not out["voices"]:
        raise Refused(f"No rated voice {voice!r}. Voices are written DATASET/SPEAKER, e.g. VCTK/p250.", 404)
    return jsonify(out)


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
    url = f"{API}/evidence/questions/{number}/chart{views.range_query(start, end)}"
    return jsonify(views.question(number, summary, evidence_snapshot(), chart_url=url))


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


def latest_charts():
    """(folder, summary) of the chart set made most recently (evidence-charts, or a question asked here), or None."""
    from backend.genome import evidence_charts

    best = None
    for path in Path(evidence_charts.CHARTS_DIR).glob("*/summary.json"):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if summary.get("made") and (best is None or summary["made"] > best[1]["made"]):
            best = (path.parent, summary)
    return best


@app.get(f"{API}/evidence/charts/latest")
@needs_key
def evidence_charts_latest():
    """The six questions' charts as they were last made: when, for which dates, and each chart's claim and note."""
    if not sees_others():
        raise Refused("The charts are made from every listener's scores, which this account may not see.", 403)
    found = latest_charts()
    if found is None:
        raise Refused("No charts have been made yet: run `python main_backend.py evidence-charts` on the host.", 404)
    from backend.genome import evidence_charts

    folder, summary = found
    evidence_charts.attach_captions(summary, folder)    # Gemma's text, when it was written for these numbers
    charts = []
    for c in summary.get("charts", []):
        shown = {k: c[k] for k in ("number", "question", "claim", "note", "skipped", "caption") if k in c}
        if c.get("file") and (folder / Path(c["file"]).name).is_file():
            shown["url"] = f"{API}/evidence/charts/latest/{Path(c['file']).name}"
        charts.append(shown)
    return jsonify(made=summary["made"], range=summary.get("range"), dates=folder.name,
                   genome_scores=summary.get("genome_scores"), sanity_checks=summary.get("sanity_checks"),
                   charts=charts)


@app.get(f"{API}/evidence/charts/latest/<name>")
@needs_key
def evidence_charts_latest_file(name):
    """One JPEG of the latest chart set (only the files its summary names)."""
    if not sees_others():
        raise Refused("The charts are made from every listener's scores, which this account may not see.", 403)
    found = latest_charts()
    if found is None:
        raise Refused("No charts have been made yet.", 404)
    folder, summary = found
    names = {Path(c["file"]).name for c in summary.get("charts", []) if c.get("file")}
    if name not in names or not (folder / name).is_file():
        raise Refused("No such chart.", 404)
    return send_file(folder / name, mimetype="image/jpeg")


_mos_lock = threading.Lock()


def mos_study_summary():
    """The mos_study summary, made again when a score was given after it was made."""
    from backend.genome import mos_study, ratings

    path, who = Path(mos_study.OUT_DIR) / "summary.json", ratings.who_path()
    with _mos_lock:
        stale = not path.is_file() or (who.is_file() and who.stat().st_mtime > path.stat().st_mtime)
        if stale:
            return mos_study.build()
        return json.loads(path.read_text(encoding="utf-8"))


@app.get(f"{API}/evidence/mos-study")
@needs_key
def evidence_mos_study():
    """The listening studies, counted only from the piles each listener was assigned: the Report evaluation's
    three methods compared (totals only, never which version of a run was which) and the pilot."""
    if not sees_others():
        raise Refused("The study is made from every listener's scores, which this account may not see.", 403)
    summary = mos_study_summary()
    charts = [{**c, "url": f"{API}/evidence/mos-study/{c['file']}"} for c in summary.get("charts", [])]
    return jsonify({**summary, "charts": charts})


@app.get(f"{API}/evidence/mos-study/<name>")
@needs_key
def evidence_mos_study_file(name):
    from backend.genome import mos_study

    if not sees_others():
        raise Refused("The study is made from every listener's scores, which this account may not see.", 403)
    names = {c["file"] for c in mos_study_summary().get("charts", [])}
    if name not in names or not (Path(mos_study.OUT_DIR) / name).is_file():
        raise Refused("No such chart.", 404)
    return send_file(Path(mos_study.OUT_DIR) / name, mimetype="image/jpeg")


@app.get(f"{API}/evidence/search")
@needs_key
def evidence_search():
    """The evidence records best matching a question (BM25): scores, searches' rankings, sanity checks."""
    args = request.args
    if not " ".join(args.get("q", "").split()):
        raise Refused("Ask something: ?q=words")
    records, index, _, provenance = evidence_with_provenance()
    return jsonify(**views.search(records, index, args.get("q"), args.get("kind"), args.get("k"), viewer()),
                   provenance=provenance)


@app.get(f"{API}/evidence/answer")
@needs_key
def evidence_answer():
    """An answer from the evidence that cites its records, or says that nothing answers it (backend/rag/answer.py).
    Made only from the records this caller may see."""
    if not " ".join(request.args.get("q", "").split()):
        raise Refused("Ask something: ?q=words")
    records, index, _, provenance = evidence_with_provenance()
    return jsonify(**views.answer(records, index, request.args.get("q"), viewer()), provenance=provenance)


# --- Ask the Dramatizer: About, and questions answered by the chatbot or the records ---------------------

ABOUT_QUESTION = re.compile(r"contribut|who (made|built|created|wrote|works? on|is behind)|what is (the )?dramatizer"
                            r"|how (do|can|should) (i|we|you)\b|how to use|\bhelp\b|what can (i|you) do",
                            re.IGNORECASE)
ASK_CHARS = 500
DECLINED = "I only answer questions about the Dramatizer"     # how the chatbot starts its off-topic reply (chatbot.OFF_TOPIC)
FACTS_QUESTION = re.compile(r"\b(how many|number of|which|what|list)\b.*\b(voices?|speakers?|datasets?|clips?|listeners?|"
                            r"raters?|data|split)\b", re.IGNORECASE)
# "How many genomes have been made?": counted from the searches, not looked up in the records (where the
# nearest record, one genome's sanity check, was once given as the answer)
COUNT_QUESTION = re.compile(r"\b(how many|number of|count of|total|so far)\b", re.IGNORECASE)
MADE_WORDS = re.compile(r"\b(genomes?|searche?s?|runs?|renders?)\b", re.IGNORECASE)
VOICE_WORDS = re.compile(r"\b(voices?|speakers?|datasets?|clips?|listeners?|raters?|data|split)\b", re.IGNORECASE)


def genome_counts():
    """How many genomes there are, from what this caller may see: the searches' genomes and their sanity
    checks, the named genomes (and those made by hand), and the renders kept."""
    from backend.genome import editor, search

    runs = [r for _, r in search.list_runs() if may_see(r)]
    sums = [search.summary(r) for r in runs]                 # every search: the same totals as GET /genome/searches
    named = [e for e in editor.library([], with_mos=False, viewer=genome_viewer()) if e["source"] in ("named", "edited")]
    return {"searches": len(runs), "voices_searched": len({r.get("voice") for r in runs}),
            "genomes_in_searches": sum(s["total"] for s in sums), "checked": sum(s["done"] for s in sums),
            "passed_checks": sum(s["fine"] for s in sums), "terrible": sum(s["terrible"] for s in sums),
            "failed": sum(s["failed"] for s in sums), "named": len(named),
            "made_by_hand": sum(e["source"] == "edited" for e in named),
            "private": len(read_json(PRIVATE_GENOMES, {}).get(g.user, [])),
            "renders": len(render_notes())}


def _genome_counts_text():
    c = genome_counts()
    unchecked = c["genomes_in_searches"] - c["checked"]
    return (f"{c['genomes_in_searches']} genomes have been made by {c['searches']} genome searches on "
            f"{c['voices_searched']} voices. Of these, {c['passed_checks']} passed the automatic checks, "
            f"{c['terrible']} were judged terrible and {c['failed']} failed"
            + (f", and {unchecked} {'is' if unchecked == 1 else 'are'} not checked yet" if unchecked else "") + ". "
            + ("1 genome has" if c["named"] == 1 else f"{c['named']} genomes have")
            + f" been picked and named ({c['made_by_hand']} of them edited by hand)"
            + (f", you keep {c['private']} private genome{'s' if c['private'] != 1 else ''} of your own" if c["private"] else "")
            + f", and {c['renders']} render{'s are' if c['renders'] != 1 else ' is'} kept.")


def _facts_answer(question):
    """Questions about the data itself (voices, datasets, clips, listeners), counted from the site's own data."""
    from backend.genome import search
    from backend.rag import answer
    from backend.voice_data.split_voice_data import load_splits

    made = COUNT_QUESTION.search(question) and MADE_WORDS.search(question)
    if made and not VOICE_WORDS.search(question):
        return {"answer": _genome_counts_text(), "source": "records", "route": "facts", "citations": [], "cited": [],
                "absent": False}
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
    if made:
        text += " " + _genome_counts_text()
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


def _off_topic():
    from backend.api import small_talk
    return {"answer": small_talk.OFF_TOPIC, "source": "records", "route": "off_topic", "citations": [], "absent": True,
            "links": []}


def _record_links(records, ids, limit=25):
    """The genomes and voices behind the cited records, for the page to link to (as chatbot.links does)."""
    by_id = {r["id"]: r for r in records}
    out, seen = [], set()
    for i in ids:
        f = (by_id.get(i) or {}).get("fields") or {}
        if f.get("run") and f.get("genome") not in (None, ""):
            key = f"search:{f['run']}/{f['genome']}"
            if key not in seen:
                seen.add(key)
                out.append({"kind": "genome", "key": key, "voice": f.get("voice"), "genome": f["genome"],
                            **({"score": f["score"], "listener": f.get("listener")} if f.get("score") is not None else {})})
        voice = f.get("voice")
        if voice and "/" in voice and ("voice", voice) not in seen:
            seen.add(("voice", voice))
            dataset, speaker = voice.split("/", 1)
            out.append({"kind": "voice", "dataset": dataset, "speaker": speaker})
        if len(out) >= limit:
            break
    return out[:limit]


GENOME_LIST = re.compile(r"\bgenomes?\b", re.IGNORECASE)
LIST_WORDS = re.compile(r"\b(list|show|which|any|all|give|find|performed|perform|did well|do well|good|great|best|top|"
                        r"highest|rated|scored|gave|5|4)\b", re.IGNORECASE)
WELL = re.compile(r"\b(perform\w*|did well|do(?:es)? well|good|great|best|top|highest|well)\b", re.IGNORECASE)
SCORE_ASKED = re.compile(r"(?<![\w/.])([0-5](?:\.5)?)(?![\w/.])\s*(?:/\s*5|stars?|points?)?\s*(or (?:more|higher|above|better)|\+)?",
                         re.IGNORECASE)
AT_LEAST = re.compile(r"\b(at least|or more|or higher|or above|or better|above|over)\b|\+", re.IGNORECASE)
LIST_SIZE = 20
# The six research questions (and per-voice summaries) keep their own answers (evidence questions 1-6)
RESEARCH_QUESTION = re.compile(r"\b(each|every|per)\s+(voice|dataset|feature|gene)|exclude|artifact|configuration|"
                               r"consistent|beat(s)? the plain|active features", re.IGNORECASE)


def _genome_list_answer(question):
    """
    A list of genomes, straight from the scores, for questions like "Did Rama rate any genomes 5?" or "Show me
    genomes that performed well": each with its voice, number, score and listener, and a link for the page.
    None when the question is not one (the records answer it instead). Only the scores the asker may see.
    """
    if not GENOME_LIST.search(question) or not LIST_WORDS.search(question) or RESEARCH_QUESTION.search(question):
        return None
    rows = score_rows()
    args = {"clip": "genome", "sort": "score", "limit": 200}
    listeners = sorted({r["listener"] for r in rows if r.get("listener")}, key=len, reverse=True)
    said = next((n for n in listeners if re.search(rf"\b{re.escape(n)}\b", question, re.IGNORECASE)), None)
    if said:
        args["listener"] = said
    voice = re.search(r"\b(p\d{3}|\d{4}|vctk|crema-d|libritts-r)\b", question, re.IGNORECASE)
    if voice:
        args["voice"] = voice.group(1)
    score = SCORE_ASKED.search(question)
    if score:
        args["min_score"] = score.group(1)
        if not (score.group(2) or AT_LEAST.search(question)):
            args["max_score"] = score.group(1)
    elif WELL.search(question):
        args["min_score"] = 4
    else:
        return None                            # no score and no "well/best": not a list question we can answer
    if re.search(r"\b(worst|lowest|bad|poor)\b", question, re.IGNORECASE):
        args["order"] = "asc"
    result = views.scores(rows, args)
    found, matched = result["rows"], result["rows_matched"]
    if not found:
        who = f"{said} " if said else "the listeners you may see "
        return {"answer": f"I found no genome scores matching that: {who}gave no genome "
                          + (f"a {args['min_score']}" if "max_score" in args else f"{args['min_score']} or more") + ".",
                "source": "records", "route": "genome_list", "citations": [], "absent": True, "links": []}
    shown = found[:LIST_SIZE]
    asked = (f"exactly {args['min_score']}" if "max_score" in args else f"{args['min_score']} or more")
    lines = [f"- {r['voice']} genome {r['genome']}: {r['score']:g} by {r['listener']}"
             + (f" ({'; '.join(r['genes_described'][:2])})" if r.get("genes_described") else "") for r in shown]
    head = (f"{matched} genome score{'s' if matched != 1 else ''} of {asked}"
            + (f" from {said}" if said else "") + (f" for {args['voice']}" if "voice" in args else "")
            + (f"; the first {len(shown)}" if matched > len(shown) else "") + ":")
    return {"answer": head + "\n" + "\n".join(lines), "source": "records", "route": "genome_list", "citations": [],
            "absent": False, "cited": matched, "links": _row_links(shown), "filters": args}


def _row_links(rows, limit=25):
    """Links for the page: each genome of the rows (to open in the studio) and each voice, once each."""
    out, seen = [], set()
    for r in rows:
        key = f"search:{r['search']}/{r['genome']}"
        if r.get("search") and r.get("genome") is not None and key not in seen:
            seen.add(key)
            out.append({"kind": "genome", "key": key, "voice": r["voice"], "genome": r["genome"], "score": r["score"],
                        "listener": r.get("listener"), "genes": r.get("genes_described") or None})
    for r in rows:
        if r.get("voice") and "/" in r["voice"] and ("voice", r["voice"]) not in seen:
            seen.add(("voice", r["voice"]))
            dataset, speaker = r["voice"].split("/", 1)
            out.append({"kind": "voice", "dataset": dataset, "speaker": speaker})
    return out[:limit + 10]


def _about_us(question):
    """Whether a question could be about the Dramatizer: small_talk's words, or a term of the glossary."""
    from backend import glossary
    from backend.api import small_talk

    return small_talk.about_us(question) or glossary.best(question) is not None


def _answer_from_records(question):
    """The answer without the model: About for questions about the project, else the records (answer.py)."""
    from backend.api import about as about_text
    from backend.rag import answer

    if ABOUT_QUESTION.search(question):
        return {"answer": about_text.as_text(), "source": "about", "citations": [], "absent": False, "links": []}
    if not _about_us(question):
        return _off_topic()                    # never matched against the records: that once gave nonsense
    listed = _genome_list_answer(question)
    if listed:
        return listed
    if COUNT_QUESTION.search(question) and answer.route(question)[0] == "search":
        facts = _facts_answer(question)        # "how many genomes / voices": counted, never the nearest record
        if facts:
            return facts
        return {"answer": "That needs counting, and one record cannot answer it. Without the language model, "
                          "these counts are ready: how many genomes, genome searches, renders, voices, clips and "
                          "listeners there are. Once the model has loaded, it can count other things too.",
                "source": "records", "route": "facts", "citations": [], "cited": 0, "absent": True, "links": []}
    records, index, _ = evidence()
    a = answer.answer(question, records, index, listener=None if sees_others() else g.user)
    if a["absent"]:
        facts = _facts_answer(question)        # a question about the data itself, not about the scores
        if facts:
            return facts
    return {"answer": a["answer"], "source": "records", "citations": a["citations"], "cited": a["cited"],
            "absent": a["absent"], "route": a["route"],
            "links": [] if a["absent"] else _record_links(records, a.get("cited_ids") or [])}


@app.get(f"{API}/glossary")
@needs_key
def glossary_terms():
    """What the Dramatizer's words mean (backend/glossary.py): ?q= a term or a question -> the best matches;
    without q, every term with its one-line meaning."""
    from backend import glossary

    q = " ".join(request.args.get("q", "").split())[:200]
    if not q:
        return jsonify(terms=[{"term": e["term"], "kind": e["kind"], "meaning": e["meaning"]} for e in glossary.entries()])
    found = glossary.lookup(q)
    return jsonify(query=q, matches=found, note=None if found else
                   "No term of the Dramatizer matches that. Without q, every term is listed.")


def _glossary_answer(question, history):
    """A question that only asks what a term means, answered from the glossary at once (no model, no records).
    A follow-up such as "but what does it measure?" takes the term from the last question that named one."""
    from backend import glossary

    if ABOUT_QUESTION.search(question):
        return None
    entry, follow_up = glossary.asked(question)
    if follow_up:
        for earlier in reversed([m["content"] for m in history if m.get("role") == "user"]):
            entry = glossary.best(earlier)
            if entry:
                break
    if entry is None:
        return None
    return {"answer": glossary.as_text(entry), "source": "glossary", "route": "glossary", "term": entry["term"],
            "citations": [], "links": [], "tools": [], "absent": False}


@app.post(f"{API}/assistant/ask")
@needs_key
def assistant_ask():
    """
    Answer a question about the Dramatizer: {question, history?: [{role, content}]}. The chatbot (Gemma 4)
    answers when it is ready, calling this API with the asker's own token; otherwise, or if it fails, the
    records answer (with citations). "source" says which: model, records or about. "note" says why.
    """
    from backend.api import small_talk
    from backend.api.assistant import ASSISTANT

    data = request.get_json(silent=True) or {}
    question = " ".join(str(data.get("question") or "").split())[:ASK_CHARS]
    if not question:
        raise Refused("Ask something.")
    per_minute("question")
    per_minute("question", add=True)
    hello = small_talk.greeting(question)
    if hello:                                  # small talk: answered at once, not looked up anywhere
        return jsonify(answer=hello, source="small_talk", citations=[], links=[], tools=[], absent=False)
    history = [{"role": m.get("role"), "content": str(m.get("content", ""))[:2000]}
               for m in (data.get("history") or [])[-6:] if isinstance(m, dict) and m.get("role") in ("user", "assistant")]
    defined = _glossary_answer(question, history)
    if defined:                                # "what is STOI?": the glossary says it exactly, at once
        return jsonify(defined)
    status = ASSISTANT.status()
    note = None
    if status["state"] == "ready":
        try:
            reply = ASSISTANT.ask(question, _token(), history)
            if str(reply.get("answer", "")).startswith(DECLINED) and (ABOUT_QUESTION.search(question)
                                                                       or _about_us(question)):
                out = _answer_from_records(question)        # declined, but About or the records may still answer it
                if not out.get("absent"):
                    out["note"] = None
                    return jsonify(out)
            return jsonify(answer=reply["answer"], source="small_talk" if reply.get("small_talk") else "model",
                           model=status["model"], tools=reply.get("tools", []), links=reply.get("links", []),
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
    return jsonify(results_data())


def results_data():
    """The baselines' and Milestone 2's summaries (also published by publish_results.py)."""
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
    return out


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
    from backend import model_store

    model_store.check_at_start()          # with MODELS_DIR set: every pinned model there, or no start (Milestone 4)
    people = list(keys.load()["keys"]) + [f"{a.display_name} (login)" for a in accounts.load().values() if not a.disabled]
    context = tls_context()
    threading.Thread(target=worker, name="gpu jobs", daemon=True).start()
    threading.Thread(target=library, name="read voices", daemon=True).start()
    from backend.api.assistant import ASSISTANT
    ASSISTANT.configure(f"https://127.0.0.1:{port}", fingerprint())     # where "Ask the Dramatizer"'s model calls back
    atexit.register(ASSISTANT.stop)
    chatbot = ASSISTANT._optional("CHATBOT_PRELOAD").strip().lower() not in ("0", "no", "false", "off")
    if model.setting("PRELOAD_VOICE_MODEL", "1").lower() not in ("0", "no", "false", "off"):
        # One after the other: loading Gemma (15 GB) while CosyVoice loads starved CosyVoice for over 5 minutes in the
        # container (Milestone 4). The chatbot is not needed for /ready, so it loads second.
        threading.Thread(target=preload_models, args=(ASSISTANT if chatbot else None,), name="load the models",
                         daemon=True).start()
    elif chatbot:
        ASSISTANT.start()          # load the model now, so the first question does not wait for it (it calls back later)
    print(f"""
Dramatizer API on https://127.0.0.1:{port}{API}
  Certificate fingerprint (give it to the people you share with; their client checks it):
    {fingerprint()}
  Who may use it: {len(people)} ({', '.join(people) or 'nobody yet: python main_backend.py api-key add NAME, or account add USERNAME'})
  playit.gg: a TCP tunnel to local address 127.0.0.1, port {port}. Only tunnel this port.
  Stop with Ctrl+C.
""", flush=True)
    app.run(host=HOST, port=port, threaded=True, ssl_context=context, request_handler=ClosesIdleConnections)


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

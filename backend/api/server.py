"""server.py | The back end as a web API: others use it through the client app, the work runs on this PC.

    python main_backend.py api-key add Ana     # one key per person (backend/api/keys.py)
    python main_backend.py api                 # start the API on https://127.0.0.1:5005

    In playit.gg: a TCP tunnel to local address 127.0.0.1, port 5005. Give each
    person the playit address, their key and the fingerprint this prints.

Safe to put behind a tunnel because:

    - every call needs a key (Authorization: Bearer <key>); keys are per person and can be removed
    - it speaks HTTPS with its own certificate (history/api/); the client pins its fingerprint,
      so a tunnel or network in between cannot read or change the traffic
    - it can only do what is listed below: no commands, no file paths from outside, voices by id
    - the GPU work waits in one queue, one job at a time, at most 3 waiting per person
    - only validation voices are offered; final test speakers and custom voices never are
    - every call is logged with who made it (history/api/access.log)

The API, all under /api/v1 (JSON in and out, audio as WAV):

    GET  /health                                  no key needed: is it up?
    GET  /me                                      who this key belongs to, and the queue
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
    POST /genome/searches/<run>/ratings {scores: {listener: {clip: 0-5}}}   save MOS scores
    POST /genome/searches/<run>/pick    {genome, name}                    pick and name a genome
    POST /genome/searches/<run>/unpick  {genome}                          take the name off
    GET  /genome/named                            the named genomes
    GET  /results                                 the baselines' and Milestone 2's summaries
"""
import argparse
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

from flask import Flask, abort, g, jsonify, request, send_file
from werkzeug.exceptions import HTTPException
from werkzeug.serving import WSGIRequestHandler

from backend import model
from backend.api import keys

API = "/api/v1"
VERSION = 1
PORT = int(model.setting("API_PORT", "5005"))
API_DIR = model.PROJECT_DIR / "history" / "api"
CERT_PATH, KEY_PATH = API_DIR / "api_cert.pem", API_DIR / "api_key.pem"
ACCESS_LOG = API_DIR / "access.log"
SPEAK_DIR = model.PROJECT_DIR / model.setting("OUTPUT_DIR", "output") / "api"
DATASETS = ("VCTK", "CREMA-D", "LibriTTS-R")
PAGE_SIZE = 50
MAX_TEXT = 300
MAX_WAITING = 3          # jobs per person, queued or running
IDLE_SECONDS = 30
CLIP_FILE = re.compile(r"^(baseline|genome_\d{2})\.wav$")
JOB_ID = re.compile(r"^[0-9a-f]{32}$")

app = Flask(__name__)
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


def needs_key(view):
    @wraps(view)
    def checked(*args, **kwargs):
        header = request.headers.get("Authorization", "")
        name = keys.who(header[7:].strip()) if header.startswith("Bearer ") else None
        if name is None:
            time.sleep(1)       # slows down guessing; the keys are far too long to guess anyway
            raise Refused("A valid API key is needed (Authorization: Bearer <key>).", 401)
        g.user = name
        return view(*args, **kwargs)
    return checked


@app.after_request
def log_and_protect(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
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
    """{dataset: {"clips": [...], "by_id": {...}}} of validation speakers, read once (a few seconds)."""
    with _library_lock:
        if _library["clips"] is None:
            from backend.voice_data import split_voice_data

            validation = set(split_voice_data.load_splits()["validation"])
            out = {name: {"clips": [], "by_id": {}} for name in DATASETS}
            for clip in split_voice_data.read_library():
                if clip.dataset in out and f"{clip.dataset}/{clip.speaker}" in validation:
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
jobs_lock = threading.Lock()


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


@app.get(f"{API}/me")
@needs_key
def me():
    with jobs_lock:
        mine = [job_json(j) for j in jobs.values() if j["owner"] == g.user]
        busy = sum(j["status"] in ("queued", "running") for j in jobs.values())
    return jsonify(name=g.user, version=VERSION, jobs_waiting=busy, my_jobs=mine[-20:])


@app.get(f"{API}/datasets")
@needs_key
def datasets():
    lib = library()
    return jsonify(datasets=[{"name": name, "clips": len(lib[name]["clips"]),
                              "speakers": len({c.speaker for c in lib[name]["clips"]})} for name in DATASETS])


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
def speak():
    from backend.voice_data.Voice_Importer import get_clip

    data = body()
    clip = clip_of(data.get("dataset"), data.get("clip_id"))
    text, seed, owner = text_of(data), seed_of(data), g.user

    def work():
        from backend.speech.Voice_model import generate_speech

        reference_audio, reference_text = get_clip(clip)
        started = time.time()
        path = generate_speech(reference_audio, reference_text, text,
                               voice_name=f"api_{clip.dataset}_{clip.speaker}", seed=seed)
        return {"path": str(path), "seconds": round(time.time() - started, 1)}

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

    out = []
    for folder, run in search.list_runs()[:100]:
        out.append({"id": run["id"], "voice": run["voice"], "text": run["text"], "created": run["created"],
                    "status": "queued or running" if searching(run["id"]) else run["status"],
                    "summary": search.summary(run), "requested_by": run.get("requested_by"),
                    "rated": ratings.is_rated(folder)})
    return jsonify(searches=out)


@app.post(f"{API}/genome/searches")
@needs_key
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
                                seed=seed_of(data), origin=split_voice_data.voice_origin(clip))
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
    busy = searching(run_id)
    state = run["status"]
    if state == "running" and not busy:
        state = "stopped"
    rating = ratings.load(folder)
    return jsonify(run=public_run(run), state="queued or running" if busy else state,
                   summary=search.summary(run), ratings={k: rating[k] for k in ("raters", "scores", "picked")},
                   ranking=ratings.ranked(folder, run, rating), max_raters=ratings.MAX_RATERS)


@app.post(f"{API}/genome/searches/<run_id>/continue")
@needs_key
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
    path = run_dir_of(run_id) / name
    if not path.is_file():
        raise Refused("That clip is not made yet.", 404)
    return send_file(path, mimetype="audio/wav")


def not_while_searching(run_id):
    if searching(run_id):
        raise Refused("This search is still running: rate it when it has finished.", 409)


@app.post(f"{API}/genome/searches/<run_id>/ratings")
@needs_key
def rate(run_id):
    from backend.genome import ratings

    folder = run_dir_of(run_id)
    not_while_searching(run_id)
    scores = body().get("scores")
    if not isinstance(scores, dict) or not scores:
        raise Refused('Send {"scores": {"listener name": {"genome_01": 4, ...}}}.')
    if not all(isinstance(clips, dict) for clips in scores.values()):
        raise Refused("Each listener's scores must be an object of clip: score.")
    try:
        saved = ratings.save_scores(folder, scores, submitted_by=g.user)
    except (TypeError, ValueError) as error:
        raise Refused(str(error))
    return jsonify(saved=sum(len(c) for c in scores.values()), raters=saved["raters"],
                   ranking=ratings.ranked(folder))


@app.post(f"{API}/genome/searches/<run_id>/pick")
@needs_key
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


# --- Results -----------------------------------------------------------------

@app.get(f"{API}/results")
@needs_key
def results():
    out = {"baselines": [], "milestone2": []}
    for folder in sorted((model.PROJECT_DIR / "output" / "baseline").glob("*/summary.json")):
        record_path = folder.parent / "record.json"
        record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
        checks_path = folder.parent / "checks.json"
        checks = json.loads(checks_path.read_text(encoding="utf-8")).get("checks", []) if checks_path.is_file() else []
        out["baselines"].append({"name": folder.parent.name, "made": record.get("made"),
                                 "lines": record.get("lines"), "voices": len(record.get("voices") or []),
                                 "git_commit": record.get("git_commit"), "git_dirty": record.get("git_dirty"),
                                 "checks": [{"name": c["name"], "passed": c["passed"]} for c in checks],
                                 "summary": json.loads(folder.read_text(encoding="utf-8"))})
    for folder in sorted((model.PROJECT_DIR / "output" / "milestone2").glob("*/summary.json")):
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


def tls_context():
    if not CERT_PATH.is_file() or not KEY_PATH.is_file():
        make_certificate()
    context = HandshakeOnItsOwnThread(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(CERT_PATH, KEY_PATH)
    return context


def main(port=PORT):
    people = keys.load()["keys"]
    context = tls_context()
    threading.Thread(target=worker, name="gpu jobs", daemon=True).start()
    threading.Thread(target=library, name="read voices", daemon=True).start()
    print(f"""
Dramatizer API on https://127.0.0.1:{port}{API}
  Certificate fingerprint (give it to the people you share with; their client checks it):
    {fingerprint()}
  Keys: {len(people)} ({', '.join(people) or 'none yet: python main_backend.py api-key add NAME'})
  playit.gg: a TCP tunnel to local address 127.0.0.1, port {port}. Only tunnel this port.
  Stop with Ctrl+C.
""", flush=True)
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

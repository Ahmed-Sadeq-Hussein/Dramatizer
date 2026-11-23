"""smoke.py | The checks that allow or block a release (Milestone 4, step 4). Exit 0: allow. Exit 1: block.

    python deploy/smoke.py https://127.0.0.1:5107 --release deploy/release.json

  1. ready     /ready answers 200 within 5 minutes (CosyVoice and Gemma take about 2 minutes to load)
  2. version   /version matches release.json on every compared field (commit, voice pack, models, settings)
  3. login     the smoke account (smoke-test, a listener made only for this) logs in
  4. render    one fixed line, voice and seed: done within 120 s, longer than 1 s, Whisper's WER at most 0.10
  5. chatbot   "How many voices are included?" is answered within 90 s with a number in it
               (a records answer counts while the model loads)

Every result, with its time, goes to reports/milestone4/04-checks/<release>.json. When a check fails, the ones that
need it are skipped, and the run still blocks.

The server's certificate is checked by fingerprint (--fingerprint, DRAMATIZER_FINGERPRINT, or this computer's
history/api/api_cert.pem). The smoke account's password is read from --password-file (default: SMOKE_PASSWORD_FILE,
else D:/Dramatizer secrets/smoke-test.txt). It is never printed or written down.
"""

import argparse
import hashlib
import json
import os
import ssl
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "API calls"))

from dramatizer_api import Api  # noqa: E402  (the fingerprint-checked connection)

from backend import release_info  # noqa: E402

REPORTS = ROOT / "reports" / "milestone4" / "04-checks"
PASSWORD_FILE = Path(os.environ.get("SMOKE_PASSWORD_FILE", "D:/Dramatizer secrets/smoke-test.txt"))

READY_SECONDS = 300
RENDER_SECONDS = 120
CHATBOT_SECONDS = 90
MAX_WER = 0.10
MIN_AUDIO_SECONDS = 1.0
# The fixed render: one validation voice (never a test voice), one line, one seed
VOICE = {"dataset": "VCTK", "clip_id": "p225_023_mic1"}
LINE = "The old lighthouse keeper climbed the stairs every evening to light the lamp."
SEED = 7
QUESTION = "How many voices are included?"


def local_fingerprint():
    """The fingerprint of this computer's API certificate, if it has one."""
    cert = ROOT / "history" / "api" / "api_cert.pem"
    if not cert.is_file():
        return None
    return hashlib.sha256(ssl.PEM_cert_to_DER_cert(cert.read_text())).hexdigest()


class Smoke:
    def __init__(self, api, release, password_file, poll=2.0, wer=None):
        self.api, self.release, self.password_file, self.poll = api, release, Path(password_file), poll
        self.wer = wer                           # (audio path, text) -> WER; Whisper unless a test gives one
        self.results = []

    # Each check returns (ok, detail); a check that raises fails with the reason
    def run(self, name, check, needs=()):
        failed = [r["check"] for r in self.results if r["check"] in needs and r["result"] != "pass"]
        if failed:
            self.results.append({"check": name, "result": "skipped", "seconds": 0,
                                 "detail": f"needs {', '.join(failed)}"})
            return False
        started = time.time()
        try:
            ok, detail = check()
        except (Exception, SystemExit) as error:            # dramatizer_api stops with SystemExit
            ok, detail = False, f"{type(error).__name__}: {error}"
        self.results.append({"check": name, "result": "pass" if ok else "FAIL",
                             "seconds": round(time.time() - started, 1), "detail": detail})
        return ok

    def ready(self):
        deadline, last = time.time() + READY_SECONDS, None
        while time.time() < deadline:
            try:
                status, _, data = self.api.request("GET", "ready")
                last = json.loads(data or b"{}")
                if status == 200:
                    return True, last.get("checks")
            except (OSError, ValueError) as error:          # not listening yet
                last = {"error": str(error)}
            time.sleep(self.poll)
        return False, f"not ready after {READY_SECONDS} s: {last}"

    def version(self):
        running = self.api.json("GET", "version")
        wrong = release_info.differences(self.release, running)
        if wrong:
            return False, {field: {"release.json": want, "running": got} for field, want, got in wrong}
        return True, {field: running.get(field) for field in ("commit", "config_sha256")}

    def login(self):
        if not self.password_file.is_file():
            return False, f"no password file for the smoke account ({self.password_file.name})"
        password = self.password_file.read_text(encoding="utf-8").strip()
        self.api.token = self.api.json("POST", "login", {"username": "smoke-test", "password": password})["token"]
        return True, "smoke-test"

    def render(self):
        job = self.api.json("POST", "speak", {**VOICE, "text": LINE, "seed": SEED})["job"]
        deadline = time.time() + RENDER_SECONDS
        while job["status"] in ("queued", "running") and time.time() < deadline:
            time.sleep(self.poll)
            job = self.api.json("GET", f"jobs/{job['id']}")["job"]
        if job["status"] != "done":
            return False, f"the job is {job['status']} after {RENDER_SECONDS} s: {job.get('error')}"
        status, kind, audio = self.api.request("GET", f"jobs/{job['id']}/audio")
        if status != 200:
            return False, f"the audio answered {status}"
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "smoke.wav"
            path.write_bytes(audio)
            import soundfile

            seconds = soundfile.info(str(path)).duration
            wer = self.wer(path, LINE) if self.wer else _whisper_wer(path, LINE)
        detail = {"render_seconds": job.get("seconds"), "audio_seconds": round(seconds, 2), "wer": round(wer, 3)}
        return seconds > MIN_AUDIO_SECONDS and wer <= MAX_WER, detail

    def chatbot(self):
        self.api.timeout = CHATBOT_SECONDS
        started = time.time()
        reply = self.api.json("POST", "assistant/ask", {"question": QUESTION})
        took = time.time() - started
        answer = str(reply.get("answer") or "")
        ok = took <= CHATBOT_SECONDS and any(c.isdigit() for c in answer)
        return ok, {"source": reply.get("source"), "answer_seconds": round(took, 1), "has_a_number": ok}

    def all(self):
        self.run("ready", self.ready)
        self.run("version", self.version, needs=("ready",))
        self.run("login", self.login, needs=("ready",))
        self.run("render", self.render, needs=("ready", "login"))
        self.run("chatbot", self.chatbot, needs=("ready", "login"))
        return all(r["result"] == "pass" for r in self.results)


def _whisper_wer(path, text):
    from backend.metrics.WER import word_error_rate

    return word_error_rate(str(path), text)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Allow (exit 0) or block (exit 1) a release candidate.")
    parser.add_argument("address", help="the candidate, e.g. https://127.0.0.1:5107")
    parser.add_argument("--release", default=str(ROOT / "deploy" / "release.json"))
    parser.add_argument("--fingerprint", default=os.environ.get("DRAMATIZER_FINGERPRINT"))
    parser.add_argument("--password-file", default=str(PASSWORD_FILE))
    args = parser.parse_args(argv)

    release = json.loads(Path(args.release).read_text(encoding="utf-8"))
    api = Api(args.address, args.fingerprint or local_fingerprint())
    smoke = Smoke(api, release, args.password_file)
    allowed = smoke.all()
    api.log_out("smoke-test")

    record = {"release": release.get("release"), "address": args.address, "checked": datetime.now().isoformat(
        timespec="seconds"), "allowed": allowed, "checks": smoke.results}
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / f"{release.get('release') or 'unnamed'}.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for r in smoke.results:
        print(f"{r['result']:>7}  {r['check']:<8} {r['seconds']:>6} s  {r['detail']}")
    print(f"{'ALLOWED' if allowed else 'BLOCKED'}: {release.get('release')} (written to {out.relative_to(ROOT)})")
    return 0 if allowed else 1


if __name__ == "__main__":
    sys.exit(main())

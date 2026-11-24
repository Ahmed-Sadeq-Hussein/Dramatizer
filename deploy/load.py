"""load.py | Measure the site under load (Milestone 4, step 7): how many people it serves, and what MAX_QUEUE should be.

    python deploy/load.py https://127.0.0.1:5005                     # 1, 5 and 20 users, 10 minutes each
    python deploy/load.py https://127.0.0.1:5005 --users 5 --minutes 2

Each simulated person logs in with their own account (load-01 ... load-20, passwords read from
D:/Dramatizer secrets/load-users.txt, never printed), renders a line every 30-60 s (a random validation voice and
line), and asks the chatbot a question every 2 minutes. They start spread over the first 30 s.

For each run it reports: render time p50/p95, split into waiting (in the queue) and working (on the GPU); chatbot
answer time p50/p95 and where answers came from; renders per minute; refusals (429, 503) and timeouts; the
deepest the queue got; and the GPU memory (nvidia-smi every 5 s). The chatbot does not stream, so the time to its
first word is the time to its answer.

Capacity, with Little's law (jobs in the system L = arrivals per second x seconds each spends in it): one GPU
renders one line at a time, so it finishes at most 1/work renders a second. Each person asks for 1/45 a second
(one every 30-60 s). With 30 % headroom for bursts and slow renders the site serves
    people = (1 / work) / (1/45) / 1.3
at once, and a queue that holds what those people have waiting at once, L = people x (1/45) x (wait + work), with
the same headroom, is the MAX_QUEUE to set.

Everything goes to reports/milestone4/07-load/: each run's events (JSON) and a table (README.md).
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "API calls"))
sys.path.insert(0, str(ROOT / "deploy"))

from dramatizer_api import Api  # noqa: E402

REPORTS = ROOT / "reports" / "milestone4" / "07-load"
USERS_FILE = Path(os.environ.get("LOAD_USERS_FILE", "D:/Dramatizer secrets/load-users.txt"))
RENDER_EVERY = (30, 60)          # seconds between one person's renders
QUESTION_EVERY = 120             # seconds between one person's questions
RENDER_TIMEOUT = 300             # a render not done in this long counts as a timeout
QUESTION_TIMEOUT = 120
HEADROOM = 1.3
LINES = [
    "The old lighthouse keeper climbed the stairs every evening to light the lamp.",
    "I told you already, the train leaves at seven and we are not going to miss it.",
    "Nobody in the village had ever seen snow fall so early in the autumn.",
    "Could you please hand me the map before we take the wrong road again?",
    "She laughed so hard that the whole audience turned to look at her.",
]
# Questions for the model itself: "what is X?" would be answered at once from the glossary and measure nothing
QUESTIONS = ["How many voices are included?", "Which dataset has the most voices, and why were they chosen?",
             "Why are some voices kept back for the final test?", "How did the genome search do against the baseline?",
             "Which listening results show the clearest improvement so far?"]


def percentile(values, q):
    """The q-th percentile (0-100), by linear interpolation; None for no values."""
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    k = (len(values) - 1) * q / 100
    low, high = math.floor(k), math.ceil(k)
    return round(values[low] + (values[high] - values[low]) * (k - low), 2)


def users(path=USERS_FILE):
    """[(username, password)] from the load users' file: one "username password" per line."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and not line.startswith("#"):
            out.append((parts[0], parts[1]))
    return out


def validation_voices(count=12, seed=0):
    """A fixed handful of validation voices (never a test voice) as {"dataset", "clip_id"}."""
    from backend.voice_data import split_voice_data

    chosen = sorted(split_voice_data.voice_clips("validation").items())
    picked = random.Random(seed).sample(chosen, min(count, len(chosen)))
    return [{"dataset": key.split("/")[0], "clip_id": clip} for key, clip in picked]


class Person(threading.Thread):
    """One simulated person: renders every 30-60 s and asks every 2 minutes until `until`."""

    def __init__(self, number, account, connect, voices, until, events, lock, rng, delay):
        super().__init__(name=f"person {number}", daemon=True)
        self.number, self.account, self.connect, self.voices = number, account, connect, voices
        self.until, self.events, self.lock, self.rng, self.delay = until, events, lock, rng, delay

    def record(self, **event):
        with self.lock:
            self.events.append({"person": self.number, **event})

    def run(self):
        time.sleep(self.delay)
        api = self.connect()
        try:
            api.token = api.json("POST", "login", {"username": self.account[0], "password": self.account[1]})["token"]
        except (SystemExit, OSError) as error:
            self.record(kind="login", at=time.time(), status="error", error=str(error)[:200])
            return
        asker = threading.Thread(target=self.ask_loop, args=(api.token,), daemon=True)
        asker.start()
        while time.time() < self.until:
            started = time.time()
            self.render(api)
            pause = self.rng.uniform(*RENDER_EVERY) - (time.time() - started)
            if pause > 0:
                time.sleep(min(pause, max(0.0, self.until - time.time())))
        asker.join(timeout=QUESTION_TIMEOUT)

    def render(self, api):
        voice, line = self.rng.choice(self.voices), self.rng.choice(LINES)
        started = time.time()
        status, _, data = api.request("POST", "speak", {**voice, "text": line, "seed": self.rng.randint(0, 999)})
        if status != 202:
            self.record(kind="render", at=started, status=f"http {status}", total=round(time.time() - started, 2))
            return
        job, deepest = json.loads(data)["job"], 0
        while job["status"] in ("queued", "running") and time.time() - started < RENDER_TIMEOUT:
            deepest = max(deepest, job.get("place_in_queue") or 0)
            time.sleep(1.0)
            code, _, data = api.request("GET", f"jobs/{job['id']}")
            if code != 200:
                break
            job = json.loads(data)["job"]
        total = round(time.time() - started, 2)
        if job["status"] in ("queued", "running"):
            api.request("DELETE", f"jobs/{job['id']}")
            self.record(kind="render", at=started, status="timeout", total=total, place=deepest)
            return
        work = job.get("seconds") if job["status"] == "done" else None
        self.record(kind="render", at=started, status=job["status"], total=total, work=work,
                    wait=round(max(0.0, total - work), 2) if work is not None else None, place=deepest,
                    error=(job.get("error") or "")[:200] or None)

    def ask_loop(self, token):
        api = self.connect()
        api.token, api.timeout = token, QUESTION_TIMEOUT
        next_at = time.time() + self.rng.uniform(0, QUESTION_EVERY)
        while next_at < self.until:
            time.sleep(max(0.0, next_at - time.time()))
            started = time.time()
            try:
                status, _, data = api.request("POST", "assistant/ask", {"question": self.rng.choice(QUESTIONS)})
                reply = json.loads(data or b"{}") if status == 200 else {}
                self.record(kind="question", at=started, status="done" if status == 200 else f"http {status}",
                            total=round(time.time() - started, 2), source=reply.get("source"))
            except OSError as error:
                self.record(kind="question", at=started, status="timeout" if "timed out" in str(error) else "error",
                            total=round(time.time() - started, 2))
            next_at += QUESTION_EVERY


def gpu_sampler(until, samples, every=5):
    while time.time() < until:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
            used, busy = (float(x) for x in out.stdout.strip().split("\n")[0].split(","))
            samples.append({"at": time.time(), "memory_mb": used, "busy_percent": busy})
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
        time.sleep(every)


def summarize(events, gpu, people, minutes):
    """The numbers of one run, from its events and GPU samples."""
    renders = [e for e in events if e["kind"] == "render"]
    done = [e for e in renders if e["status"] == "done"]
    questions = [e for e in events if e["kind"] == "question"]
    answered = [e for e in questions if e["status"] == "done"]
    count = lambda items, status: sum(1 for e in items if e["status"] == status)  # noqa: E731
    sources = {}
    for e in answered:
        sources[e.get("source") or "?"] = sources.get(e.get("source") or "?", 0) + 1
    return {
        "people": people, "minutes": minutes,
        "renders": {"asked": len(renders), "done": len(done), "per_minute": round(len(done) / minutes, 2),
                    "total_p50": percentile([e["total"] for e in done], 50),
                    "total_p95": percentile([e["total"] for e in done], 95),
                    "wait_p50": percentile([e["wait"] for e in done], 50),
                    "wait_p95": percentile([e["wait"] for e in done], 95),
                    "work_p50": percentile([e["work"] for e in done], 50),
                    "work_p95": percentile([e["work"] for e in done], 95),
                    "refused_429": count(renders, "http 429"), "refused_503": count(renders, "http 503"),
                    "timeouts": count(renders, "timeout"), "expired": count(renders, "expired"),
                    "failed": count(renders, "failed") + count(renders, "error"),
                    "deepest_queue": max([e.get("place") or 0 for e in renders] or [0])},
        "questions": {"asked": len(questions), "answered": len(answered),
                      "p50": percentile([e["total"] for e in answered], 50),
                      "p95": percentile([e["total"] for e in answered], 95),
                      "refused_429": count(questions, "http 429"), "timeouts": count(questions, "timeout"),
                      "sources": sources},
        "logins_failed": sum(1 for e in events if e["kind"] == "login"),
        "gpu": {"memory_mb_max": max([s["memory_mb"] for s in gpu] or [None], key=lambda v: v or 0),
                "busy_percent_mean": round(sum(s["busy_percent"] for s in gpu) / len(gpu), 1) if gpu else None},
    }


def capacity(work_seconds, total_seconds, every=sum(RENDER_EVERY) / 2, headroom=HEADROOM):
    """(people served at once, MAX_QUEUE) by Little's law, from one render's work time and its time in the system."""
    per_person = 1 / every                        # renders a second each person asks for
    served = (1 / work_seconds) / per_person / headroom
    in_system = served * per_person * total_seconds * headroom
    return math.floor(served), max(1, math.ceil(in_system))


def run(address, fingerprint, accounts, people, minutes, voices, seed=0):
    events, gpu, lock = [], [], threading.Lock()
    until = time.time() + minutes * 60
    rng = random.Random(seed)
    connect = lambda: Api(address, fingerprint)  # noqa: E731
    crowd = [Person(n + 1, accounts[n], connect, voices, until, events, lock, random.Random(rng.random()),
                    delay=rng.uniform(0, 30) if people > 1 else 0) for n in range(people)]
    sampler = threading.Thread(target=gpu_sampler, args=(until, gpu), daemon=True)
    sampler.start()
    for person in crowd:
        person.start()
    for person in crowd:
        person.join()
    return events, gpu


def table(summaries):
    rows = ["| People | Renders/min | Render p50 / p95 (s) | Waiting p50 / p95 | Working p50 / p95 | 429 / 503 / "
            "timeouts / expired | Deepest queue | Chatbot p50 / p95 (s) | GPU memory max |",
            "|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        r, q = s["renders"], s["questions"]
        rows.append(f"| {s['people']} | {r['per_minute']} | {r['total_p50']} / {r['total_p95']} | {r['wait_p50']} / "
                    f"{r['wait_p95']} | {r['work_p50']} / {r['work_p95']} | {r['refused_429']} / {r['refused_503']} / "
                    f"{r['timeouts']} / {r['expired']} | {r['deepest_queue']} | {q['p50']} / {q['p95']} | "
                    f"{s['gpu']['memory_mb_max']} MB |")
    return "\n".join(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure the site under load.")
    parser.add_argument("address", help="e.g. https://127.0.0.1:5005")
    parser.add_argument("--users", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--fingerprint", default=os.environ.get("DRAMATIZER_FINGERPRINT"))
    parser.add_argument("--users-file", default=str(USERS_FILE))
    parser.add_argument("--name", default=datetime.now().strftime("%Y%m%d_%H%M"))
    parser.add_argument("--report-only", action="store_true", help="rebuild README.md from the saved runs")
    args = parser.parse_args(argv)

    from smoke import local_fingerprint

    fingerprint = args.fingerprint or local_fingerprint()
    accounts, voices = ([], []) if args.report_only else (users(args.users_file), validation_voices())
    if not args.report_only and max(args.users) > len(accounts):
        raise SystemExit(f"{max(args.users)} people need {max(args.users)} accounts; {args.users_file} has "
                         f"{len(accounts)}.")
    out = REPORTS / args.name
    out.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        for people in args.users:
            print(f"{people} people for {args.minutes:g} minutes...", flush=True)
            events, gpu = run(args.address, fingerprint, accounts, people, args.minutes, voices)
            summary = summarize(events, gpu, people, args.minutes)
            (out / f"{people}-people.json").write_text(json.dumps({"summary": summary, "events": events, "gpu": gpu},
                                                                  indent=1) + "\n", encoding="utf-8")
            print(json.dumps(summary, indent=1), flush=True)
    summaries = sorted((json.loads(f.read_text(encoding="utf-8"))["summary"] for f in out.glob("*-people.json")),
                       key=lambda s: s["people"])
    text = report(args.name, args.address, summaries)
    (out / "README.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def report(name, address, summaries):
    """The README of one load test. Capacity uses the most renders a minute the GPU finished when saturated, not
    the 1-person render time: with people rendering and asking at once, CosyVoice shares the GPU with the chatbot
    and each render takes longer, so the time alone would overstate what the site can take."""
    minutes = summaries[0]["minutes"] if summaries else 0
    lines = [f"# Load test {name}", "", f"Against {address}; each person renders every {RENDER_EVERY[0]}-"
             f"{RENDER_EVERY[1]} s and asks every {QUESTION_EVERY} s; {minutes:g} minutes per run.", "",
             table(summaries), ""]
    timed = [s for s in summaries if s["renders"]["work_p50"]]
    if timed:
        alone = min(timed, key=lambda s: s["people"])
        busiest = max(timed, key=lambda s: s["renders"]["per_minute"])
        peak = busiest["renders"]["per_minute"]
        served, queue = capacity(60 / peak, alone["renders"]["total_p95"])
        lines += [f"**Capacity:** the site serves **{served} {'person' if served == 1 else 'people'} rendering at "
                  f"once** with renders done in about {alone['renders']['total_p95']:.0f} s (p95). Little's law with "
                  f"30 % headroom, from the most the GPU finished when it was saturated: {peak} renders a minute "
                  f"({busiest['people']} people), one every {60 / peak:.1f} s. A render alone takes "
                  f"{alone['renders']['work_p50']} s of GPU work; with people rendering and asking at once it took "
                  f"{max(s['renders']['work_p50'] for s in timed)} s (p50), because CosyVoice shares the GPU with the "
                  f"chatbot. A queue of {queue} holds what {served} {'person has' if served == 1 else 'people have'} "
                  f"waiting at once.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())

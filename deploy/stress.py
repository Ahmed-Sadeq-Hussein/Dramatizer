"""stress.py | Push each part of the served Dramatizer to its limit, one part at a time, to find where it breaks first.

Run it against a STRESS COPY, never the live site: a container from the same image, on the staging volume, on its
own port (see "Container launch.txt" / the report). Some tests open hundreds of connections or fill the GPU queue.

    python deploy/stress.py frontdoor                         # the front door alone (in this process, no Docker)
    python deploy/stress.py decode                            # decoding uploads (in this process)
    python deploy/stress.py http   https://127.0.0.1:5110     # the web server: requests a second, idle connections
    python deploy/stress.py login  https://127.0.0.1:5110 --password-file ...   # logins a second (PBKDF2)
    python deploy/stress.py gpu    https://127.0.0.1:5110 --password-file ... --fingerprint ...
    python deploy/stress.py all    https://127.0.0.1:5110 --password-file ... --fingerprint ...

Results are printed and written as JSON to reports/milestone4/08-stress/<part>.json.
Passwords are read from the file and never printed.
"""

import argparse
import asyncio
import io
import json
import os
import socket
import ssl
import statistics
import sys
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy"))
sys.path.insert(0, str(ROOT / "API calls"))
OUT = ROOT / "reports" / "milestone4" / "08-stress"
API = "/api/v1"


def pct(values, q):
    if not values:
        return None
    values = sorted(values)
    return round(values[min(len(values) - 1, int(q / 100 * len(values)))], 4)


def summary(times, errors, seconds):
    return {"requests": len(times), "errors": errors, "per_second": round(len(times) / seconds, 1),
            "p50_ms": round((pct(times, 50) or 0) * 1000, 1), "p95_ms": round((pct(times, 95) or 0) * 1000, 1),
            "max_ms": round(max(times) * 1000, 1) if times else None}


def save(part, result):
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"part": part, "when": time.strftime("%Y-%m-%d %H:%M:%S"), **result}
    (OUT / f"{part}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


# --------------------------------------------------------------------------- the front door, alone ---- #

def frontdoor(_args):
    """Connections a second and bytes a second through deploy/front_door.py, to a local echo server."""
    import front_door

    async def scenario():
        async def echo(reader, writer):
            try:
                while data := await reader.read(65536):
                    writer.write(data)
                    await writer.drain()
            finally:
                writer.close()

        backend = await asyncio.start_server(echo, "127.0.0.1", 0, backlog=4096)
        state = OUT / "frontdoor-state.json"
        OUT.mkdir(parents=True, exist_ok=True)
        state.write_text(json.dumps({"live": {"port": backend.sockets[0].getsockname()[1]}}))
        door = await asyncio.start_server(front_door.handler(state), "127.0.0.1", 0, backlog=4096)
        port = door.sockets[0].getsockname()[1]

        async def one():
            t = time.perf_counter()
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(b"x" * 1024)
            await writer.drain()
            got = 0
            while got < 1024:
                chunk = await reader.read(65536)
                if not chunk:
                    raise ConnectionError("closed early")
                got += len(chunk)
            writer.close()
            return time.perf_counter() - t

        levels = []
        for concurrent in (1, 16, 64, 256, 1024):
            times, errors, started = [], 0, time.perf_counter()
            while time.perf_counter() - started < 5:
                results = await asyncio.gather(*(one() for _ in range(concurrent)), return_exceptions=True)
                for r in results:
                    if isinstance(r, Exception):
                        errors += 1
                    else:
                        times.append(r)
            levels.append({"concurrent": concurrent, **summary(times, errors, time.perf_counter() - started)})
            print(f"  front door, {concurrent:5} at once: {levels[-1]['per_second']:8} connections/s, "
                  f"p95 {levels[-1]['p95_ms']} ms, errors {errors}", flush=True)

        # One big stream: how many MB/s one connection carries. Wait first: Windows keeps the thousands of closed
        # test connections' ports for a while (TIME_WAIT), and a new connection can fail until they are freed
        await asyncio.sleep(30)
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        block, total = b"y" * 65536, 200 * 1024 * 1024
        t = time.perf_counter()

        async def send():
            sent = 0
            while sent < total:
                writer.write(block)
                await writer.drain()
                sent += len(block)

        async def receive():
            got = 0
            while got < total:
                got += len(await reader.read(1 << 20))

        await asyncio.gather(send(), receive())
        mb_s = round(total / (1024 * 1024) / (time.perf_counter() - t), 1)
        writer.close()
        print(f"  front door, one stream: {mb_s} MB/s", flush=True)
        door.close()
        backend.close()
        state.unlink(missing_ok=True)
        return {"connections": levels, "stream_mb_per_s": mb_s}

    return save("frontdoor", asyncio.run(scenario()))


# --------------------------------------------------------------------------- uploads: decoding ---- #

def decode(_args):
    """Time and memory to decode one 30-second reference upload, per format (uploaded_voices._decode)."""
    import numpy as np
    import soundfile as sf
    from backend.voice_data import uploaded_voices as uv

    rate, seconds = 24000, 29.5
    t = np.arange(int(rate * seconds)) / rate
    wave = (0.3 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t))).astype("float32")
    rows = []
    for fmt, suffix in (("WAV", ".wav"), ("FLAC", ".flac"), ("OGG", ".ogg"), ("MP3", ".mp3")):
        buffer = io.BytesIO()
        try:                                # in 1-second blocks: one big Vorbis write overflows the stack on Windows
            with sf.SoundFile(buffer, "w", rate, 1, format=fmt) as out:
                for i in range(0, len(wave), rate):
                    out.write(wave[i:i + rate])
        except Exception as error:          # an older libsndfile without MP3 writing
            rows.append({"format": suffix, "skipped": str(error)[:80]})
            continue
        data = buffer.getvalue()
        times = []
        tracemalloc.start()
        for _ in range(5):
            started = time.perf_counter()
            uv._decode(data, "voice" + suffix)
            times.append(time.perf_counter() - started)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        rows.append({"format": suffix, "file_kb": len(data) // 1024, "decode_ms_p50": round(statistics.median(times) * 1000, 1),
                     "peak_memory_mb": round(peak / 1e6, 1)})
        print(f"  decode {suffix:5} {len(data) // 1024:6} KB: {rows[-1]['decode_ms_p50']} ms, "
              f"peak {rows[-1]['peak_memory_mb']} MB", flush=True)
    # The audio bomb (Vunrabilities.txt, finding 4): two hours of a constant in a small FLAC
    buffer = io.BytesIO()
    sf.write(buffer, np.full(int(16000 * 7200), 0.2, dtype="float32"), 16000, format="FLAC")
    started = time.perf_counter()
    try:
        uv._decode(buffer.getvalue(), "bomb.flac")
        bomb = "ACCEPTED (bad)"
    except uv.UploadRefused:
        bomb = "refused"
    rows.append({"format": "2-hour FLAC bomb", "file_kb": len(buffer.getvalue()) // 1024, "result": bomb,
                 "ms": round((time.perf_counter() - started) * 1000, 1)})
    print(f"  2-hour FLAC bomb ({rows[-1]['file_kb']} KB): {bomb} in {rows[-1]['ms']} ms", flush=True)
    return save("decode", {"rows": rows})


# --------------------------------------------------------------------------- the web server ---- #

def _context():
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE           # our own stress copy on this PC; the certificate is self-signed
    return context


def _get(host, port, path, context, token=None, timeout=30):
    """One request on a fresh connection (as a new visitor would make it): (seconds, status)."""
    started = time.perf_counter()
    with socket.create_connection((host, port), timeout=timeout) as raw, \
            context.wrap_socket(raw, server_hostname=host) as s:
        auth = f"Authorization: Bearer {token}\r\n" if token else ""
        s.sendall(f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\n{auth}Connection: close\r\n\r\n".encode())
        data = b""
        while chunk := s.recv(65536):
            data += chunk
    status = int(data.split(b" ", 2)[1]) if data.startswith(b"HTTP/") else 0
    return time.perf_counter() - started, status


def _hammer(host, port, path, concurrent, seconds, context, token=None):
    times, errors, codes, lock = [], 0, {}, threading.Lock()
    stop = time.perf_counter() + seconds

    def worker():
        nonlocal errors
        while time.perf_counter() < stop:
            try:
                took, status = _get(host, port, path, context, token)
                with lock:
                    codes[status] = codes.get(status, 0) + 1
                    (times.append(took) if status == 200 else None)
                    errors += status != 200
            except OSError:
                with lock:
                    errors += 1

    started = time.perf_counter()
    with ThreadPoolExecutor(concurrent) as pool:
        for _ in range(concurrent):
            pool.submit(worker)
    return {**summary(times, errors, time.perf_counter() - started), "status_codes": codes}


def http(args):
    """Requests a second on cheap and heavier routes, then how many idle connections it takes to stall it."""
    url = urlsplit(args.url)
    host, port, context = url.hostname, url.port or 443, _context()
    token = _login_token(args) if args.password_file else None
    routes = [("health", f"{API}/health", None), ("page", "/", None)]
    if token:
        routes.append(("voices list (logged in)", f"{API}/voices?dataset=VCTK", token))
    result = {"routes": {}}
    for name, path, auth in routes:
        rows = []
        for concurrent in (1, 8, 32, 128):
            row = {"concurrent": concurrent, **_hammer(host, port, path, concurrent, 8, context, auth)}
            rows.append(row)
            print(f"  {name:24} {concurrent:4} at once: {row['per_second']:7}/s, p50 {row['p50_ms']} ms, "
                  f"p95 {row['p95_ms']} ms, errors {row['errors']}", flush=True)
        result["routes"][name] = rows

    # Idle connections: open N connections that finish the TLS handshake and then send nothing (a slow client),
    # then time one normal request
    idle_rows, held = [], []
    try:
        for target in (50, 200, 500, 1000):
            while len(held) < target:
                try:
                    raw = socket.create_connection((host, port), timeout=10)
                    held.append(context.wrap_socket(raw, server_hostname=host))
                except OSError:
                    break
            try:
                took, status = _get(host, port, f"{API}/health", context, timeout=20)
                row = {"idle_connections": len(held), "health_ms": round(took * 1000, 1), "status": status}
            except OSError as error:
                row = {"idle_connections": len(held), "health_ms": None, "status": f"failed: {error}"}
            idle_rows.append(row)
            print(f"  with {row['idle_connections']:5} idle connections open: /health {row['health_ms']} ms "
                  f"({row['status']})", flush=True)
            if row["health_ms"] is None:
                break
    finally:
        for s in held:
            try:
                s.close()
            except OSError:
                pass
    result["idle_connections"] = idle_rows
    if token:
        _logout(args, token)
    return save("http", result)


# --------------------------------------------------------------------------- logins ---- #

def _api(args):
    from dramatizer_api import Api
    return Api(args.url, args.fingerprint)


def _password(args):
    from smoke import read_password
    return read_password(args.password_file)


def _login_token(args):
    from dramatizer_api import Api
    api = Api(args.url, args.fingerprint)
    return api.json("POST", "login", {"username": args.username, "password": _password(args)})["token"]


def _logout(args, token):
    from dramatizer_api import Api
    api = Api(args.url, args.fingerprint)
    api.token = token
    try:
        api.request("POST", "logout", {})
    except Exception:
        pass


def login(args):
    """Correct logins a second: each one is a PBKDF2 check of 600,000 rounds, on the CPU."""
    from dramatizer_api import Api

    password = _password(args)
    rows = []
    for concurrent in (1, 4, 16):
        times, errors, lock = [], 0, threading.Lock()
        stop = time.perf_counter() + 10

        def worker():
            nonlocal errors
            api = Api(args.url, args.fingerprint)
            while time.perf_counter() < stop:
                started = time.perf_counter()
                try:
                    token = api.json("POST", "login", {"username": args.username, "password": password})["token"]
                    took = time.perf_counter() - started
                    api.token = token
                    api.request("POST", "logout", {})
                    api.token = None
                    with lock:
                        times.append(took)
                except (Exception, SystemExit):
                    with lock:
                        errors += 1

        started = time.perf_counter()
        with ThreadPoolExecutor(concurrent) as pool:
            for _ in range(concurrent):
                pool.submit(worker)
        rows.append({"concurrent": concurrent, **summary(times, errors, time.perf_counter() - started)})
        print(f"  logins, {concurrent:3} at once: {rows[-1]['per_second']}/s, p50 {rows[-1]['p50_ms']} ms, "
              f"p95 {rows[-1]['p95_ms']} ms, errors {errors}", flush=True)
    return save("login", {"rows": rows})


# --------------------------------------------------------------------------- the GPU: renders and the chatbot ---- #

LINES = {
    "short (6 words)": "The rain kept falling all night.",
    "medium (17 words)": "She opened the old letter slowly, afraid of what it might say after all these years apart.",
    "long (40 words)": ("When the lighthouse keeper finally climbed the last of the stairs, he stopped to catch his breath, "
                        "looked out over the dark and restless water, and wondered whether the ship he had been waiting "
                        "for all winter would ever come home."),
}
VOICE = {"dataset": "VCTK", "clip_id": "p225_023_mic1"}          # the smoke test's voice


def _render(api, text, seed=7):
    """One render: total seconds as the person waits, the GPU work seconds, and the seconds of speech made."""
    started = time.perf_counter()
    while True:
        status, _, data = api.request("POST", "speak", {**VOICE, "text": text, "seed": seed})
        reply = json.loads(data or b"{}")
        if status in (429, 503):                            # a limit: wait a little, then go on (not counted)
            time.sleep(10)
            started = time.perf_counter()
            continue
        if status >= 300:
            raise RuntimeError(f"speak: {status} {reply.get('error')}")
        break
    job = reply["job"]
    while job["status"] in ("queued", "running"):
        time.sleep(0.25)
        job = api.json("GET", f"jobs/{job['id']}")["job"]
    if job["status"] != "done":
        raise RuntimeError(f"render {job['status']}: {job.get('error')}")
    took = round(time.perf_counter() - started, 2)
    audio_s = None
    status, _, audio = api.request("GET", f"jobs/{job['id']}/audio")
    if status == 200:
        import soundfile
        audio_s = round(soundfile.info(io.BytesIO(audio)).duration, 2)
    return {"total_s": took, "work_s": job.get("seconds"), "audio_s": audio_s}


def _ask(api, question):
    started = time.perf_counter()
    reply = api.json("POST", "assistant/ask", {"question": question})
    return {"seconds": round(time.perf_counter() - started, 2), "source": reply.get("source")}


def _gpu_memory():
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    used, busy = (int(x) for x in out.split(",")) if out else (None, None)
    return {"gpu_memory_mb": used, "gpu_busy_pct": busy}


def gpu(args):
    """Render time by line length (alone), the chatbot alone, then both at once (how much they slow each other)."""
    api = _api(args)
    api.token = _login_token(args)
    api.timeout = 300
    result = {"before": _gpu_memory()}
    # The model loads on the first question: ask one (not counted), then wait until it is ready
    load_started = time.time()
    try:
        _ask(api, "What is a genome?")
    except (Exception, SystemExit):
        pass
    deadline = time.time() + 600
    state = None
    while time.time() < deadline:
        state = api.json("GET", "assistant").get("state")
        if state in ("ready", "failed", "off"):
            break
        time.sleep(10)
    result["chatbot_load_s"] = round(time.time() - load_started, 1)
    result["chatbot_state"] = state
    print(f"  chatbot: {state}", flush=True)

    _render(api, LINES["short (6 words)"])                 # warm-up, not counted
    renders = {}
    for name, text in LINES.items():
        runs = [_render(api, text, seed=s) for s in (1, 2, 3)]
        renders[name] = {"runs": runs, "work_s_p50": statistics.median(r["work_s"] or r["total_s"] for r in runs),
                         "audio_s": runs[0]["audio_s"]}
        print(f"  render alone, {name:18}: {renders[name]['work_s_p50']} s of GPU work for "
              f"{renders[name]['audio_s']} s of speech", flush=True)
    result["render_alone"] = renders
    result["during_render"] = _gpu_memory()

    questions = ["Which genomes scored highest?", "How many voices are included?", "What is a genome?"]
    alone = [_ask(api, q) for q in questions]
    result["chatbot_alone"] = alone
    print("  chatbot alone: " + ", ".join(f"{a['seconds']} s ({a['source']})" for a in alone), flush=True)

    # Both at once: a render while a question is answered
    together = []
    for q in questions:
        with ThreadPoolExecutor(2) as pool:
            r = pool.submit(_render, api, LINES["medium (17 words)"], 11)
            a = pool.submit(_ask, api, q)
            together.append({"render": r.result(), "ask": a.result()})
    result["together"] = together
    print("  render + question at once: " + ", ".join(
        f"render {t['render']['work_s'] or t['render']['total_s']} s / answer {t['ask']['seconds']} s" for t in together),
        flush=True)
    result["after"] = _gpu_memory()
    _logout(args, api.token)
    return save("gpu", result)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stress each part of a Dramatizer stress copy.")
    parser.add_argument("part", choices=["frontdoor", "decode", "http", "login", "gpu", "all"])
    parser.add_argument("url", nargs="?", default="https://127.0.0.1:5110")
    parser.add_argument("--username", default="smoke-test")
    parser.add_argument("--password-file", default="")
    parser.add_argument("--fingerprint", default=os.environ.get("DRAMATIZER_FINGERPRINT", ""))
    args = parser.parse_args(argv)
    if urlsplit(args.url).port in (5005, 5105):
        raise SystemExit("That is the live site's port. Run this against a stress copy (e.g. port 5110).")
    parts = ["frontdoor", "decode", "http", "login", "gpu"] if args.part == "all" else [args.part]
    for part in parts:
        print(f"== {part}", flush=True)
        globals()[part](args)


if __name__ == "__main__":
    main()

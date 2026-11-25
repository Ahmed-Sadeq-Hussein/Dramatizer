# Stress test per component (25 Nov 2026)

Each part of the served Dramatizer was pushed on its own to find what breaks first. The aim was to see where to
spend optimisation work and compute when the site grows. Script: `deploy/stress.py` (`python deploy/stress.py
<part> ...`). The raw numbers are in the JSON files next to this README.

**Setup.** A *stress copy* was used, never the live site: image `dramatizer:v7`, the staging state volume,
port 5110, `deploy/container.env`. The machine is the home PC: RTX 3080 10 GB, 16 CPUs, 15.6 GB of memory for
Docker. The client ran on the same PC, so these numbers have no network delay. The front door was measured in
its production image (`python:3.12.8-slim`), limited to 2 CPUs. The client, the front door and an echo server
shared one process there, so the front door alone is faster than shown.

## Results

| Part | What was measured | Result | Limit reached |
|---|---|---|---|
| Front door (`front_door.py`) | new connections/s, 1 to 1024 at once | **~260/s**, 0 errors even at 1024; p95 4 ms (1) to 4 s (1024) | one asyncio loop (one CPU) |
| Front door | one stream | **568 MB/s** | not a limit |
| Web server (Werkzeug, HTTPS) | `/health`, a new TLS connection each time | 42/s with 1 at once (30 ms each); **~550/s** at 32+ | Python's GIL and the TLS handshake |
| Web server | the page `/` | **~360/s** | same |
| Web server | the voices list (logged in: token check, JSON) | **~200/s** at 8, *falls* to ~155/s at 32+; p50 0.8 s at 128 | GIL contention; each thread is slower |
| Web server | 1000 idle (slowloris) connections | still answers `/health` in 25 ms | none seen (one thread each: ~142 threads, memory) |
| Logins (PBKDF2, 600k rounds), before the fix | correct logins/s | 4.5/s (1), 6.5/s (4), **7.1/s at 16 at once**, p95 3.5 s | **one lock**: the check ran inside the account store's write lock |
| Logins, after the fix (25 Nov) | same | 4.4/s (1), **16.2/s (4)**, 14.1/s (16), **p95 0.6 s at 16** | now the database writes per login (session, last login) and the TLS handshakes |
| Upload decoding | a 30 s clip | WAV 3 ms, FLAC 7 ms, OGG 9 ms, MP3 10 ms; 8.5 MB peak | not a limit |
| Upload decoding | a 2-hour FLAC "bomb" (355 KB) | refused in 0.3 ms | the fix works |
| GPU: CosyVoice render | GPU seconds per line, alone | 6 words 3.2 s, 17 words 5.6 s, 40 words 10.7 s (~1 s of GPU per 1 s of speech, + ~1.5 s fixed) | **the GPU: one render at a time** |
| GPU: chatbot (Gemma 4 E4B) | one model answer, alone | 12 to 15 s; glossary answers 0.04 s | the GPU |
| GPU: both at once | a 17-word render while the chatbot answers | the render goes **5.6 to 10.8 s (x1.9)**; the answer 15 to 17 s (x1.1) | they share one GPU |
| GPU memory | both models loaded | **9.2 to 9.7 GB of 10.2 GB** | no room for a second model copy |
| Container memory | after the tests | 9.9 GB of the 15.6 GB VM | a second slot running at once would not fit |
| Tunnel (Cloudflare) | latency via app.dramatizer.dev | not measured: the live site was stopped | see 07-load |

Earlier load test (07-load, the whole system with real users): 1.3 renders/min for 1 person, 4.6/min at most;
from 5 people the queue is full and later requests get 503s.

## Where the bottleneck is, in order

1. **The GPU, by a factor of ~100.** One medium line takes ~5.6 s of GPU time, so one GPU makes about 10 lines
   a minute at best. The web server answers ~550 requests a second. The chatbot shares the same GPU and
   doubles render times while it answers. Scaling means more GPUs or less GPU work per line, not faster web
   code.
   - *Cheapest wins:* move the chatbot to its own GPU or the CPU (or a small hosted model), so renders are not
     slowed x1.9. Cache renders of the same (voice, text, seed); the smoke test and demos repeat lines.
     Fix the ONNX speaker embedding, which falls back to the CPU in the image: the log shows "Failed to load
     libonnxruntime_providers_cuda.so: libcublasLt.so.11", so CUDA 11 libraries are missing.
   - *Next:* batch several short lines into one GPU pass, use fp16 or a TensorRT/vLLM backend for CosyVoice's
     language model, and stream audio so the first second arrives sooner.
   - *Scale out:* one render worker per GPU behind the same queue. The queue is in-process today
     (MAX_QUEUE=4), so this needs a shared queue (Redis or the database) before a second GPU or machine helps.
2. **Logins: one at a time.** `accounts.authenticate` runs the 600k-round PBKDF2 check while holding the
   account store's write lock, so all logins on the server queue behind each other: ~7/s on 16 CPUs. This is
   also a denial-of-service lever: about 7 wrong passwords a second keeps everyone else's login waiting, before
   any lockout. **Fixed (25 Nov, in the code; live after the next release):** the password is checked on a
   read, outside the lock, and the write lock is taken only to record the result (re-checking a lock or a
   password change made meanwhile). Test: `test_password_checks_run_side_by_side`. Re-measured with the fixed
   `accounts.py` mounted into the v7 stress copy: 16.2 logins/s at 4 at once (was 6.5), p95 0.6 s at 16 (was
   3.5 s). `login-before-fix.json` and `login-after-fix.json` hold both runs. The rounds are unchanged.
3. **The web server (Werkzeug development server).** It is fine for today's traffic, but the GIL caps it near
   550 simple requests/s, and routes with real work get *slower* as more threads compete. It makes a thread
   per connection with no cap. *Fix when it matters:* gunicorn/uvicorn with several worker processes and
   timeouts, or waitress (also finding 6 in Vunrabilities.txt). Keep-alive connections would also remove the
   30 ms TLS handshake per request.
4. **The front door.** ~260 new connections/s on one loop; no errors up to 1024 at once. It reads the state
   file from disk (and, by name, resolves the container) on every connection. *Fix if needed:* cache the
   target and re-read it when the file changes, and cap concurrent connections. Cloudflare in front absorbs
   most floods anyway.
5. **Not bottlenecks:** upload decoding (milliseconds), bandwidth through the front door, idle connections.

## Compute planning (rough)

- One RTX 3080 ≈ 10 medium lines a minute with the chatbot idle, ~5 while it answers. A person dramatising a
  scene asks for a line every ~45 s (07-load), so **about 2 people can render at once** per GPU. 10 at once
  needs ~5 GPUs and a shared queue.
- CPU and memory are not the limit (the container used ~0% CPU between bursts, 9.9 GB RAM), except logins.
- GPU memory is: the 10 GB card holds one CosyVoice and one Gemma with ~0.5 GB spare.

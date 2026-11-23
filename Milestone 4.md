# Milestone 4: a checked, usable release

Written 23 Nov 2026, from Lecture 5 (Serving and Continuous Delivery) and the project as it is today.

**What Milestone 4 must show:** one fixed version of the Dramatizer → a release that was checked → useful
answers under a measured load. Plus three pieces of evidence: one release that a failed check **blocked**, the
versions the running site **loaded**, and a **way back** to the previous version.

**Our route, and why it differs from the course's.** The course uses FastAPI, Ollama on a CPU, Docker, GitHub
Actions and Azure Container Apps. Our voice model (CosyVoice 3) and chatbot (Gemma 4) need an NVIDIA GPU, which
Azure Container Apps' CPU plan does not have. The lecture asks each group to explain what replaces each part:

| Responsibility | Course | Dramatizer 2 | Why |
|---|---|---|---|
| API | FastAPI | Flask (`backend/api/server.py`) | Already built and tested; replacing it adds risk and proves nothing new |
| Model runtime | Ollama on CPU | CosyVoice 3 + Gemma 4 (transformers, 4-bit) on the GPU | Voice cloning cannot run usefully on a CPU |
| Image | Docker | Docker, with the NVIDIA runtime | Same |
| Workflow | GitHub Actions | GitHub Actions | Same |
| Hosting, revisions, traffic | Azure Container Apps | The GPU computer, Docker, a small proxy, the playit tunnel | The GPU is here |
| Data and model store | Blob Storage | Private Hugging Face datasets + a model folder pinned by revision | The team already uses Hugging Face |

Size: **S** a few hours, **M** a day or two, **L** most of a week. **GPU** means it goes on `Compute.txt` first.
Evidence goes in `reports/milestone4/`, one folder per step.

---

## The three problems to solve first

Every step below depends on these. They are the reason this plan is more than "write a Dockerfile".

### Problem 1: the site needs data that is not in the repository

The site reads the voices from `VOICE_LIBRARY_DIR` (`D:/Voices library`): VCTK (17.6 GB of parquet files),
CREMA-D (1.1 GB) and LibriTTS-R (5.7 GB). That is 24 GB, downloaded by hand from three places (two Hugging Face
datasets and openslr.org, see `Voice library.txt`). Today:

- `library()` in `server.py` reads **every** clip of every dataset at start-up (`split_voice_data.read_library()`),
  then keeps only the 176 validation voices. It needs the whole 24 GB to find 176 clips.
- VCTK clips live inside parquet files. `Voice_Importer.get_clip()` writes each one out to
  `IMPORT_DIR/VCTK/...` the first time it is used, so the container would need to be able to write there.
- Saved runs hold **absolute Windows paths** to their reference audio, for example
  `"D:\\Voices library\\CREMA-D\\audio\\..."` in `genome_search/*/run.json`. Inside a Linux container that path
  does not exist.
- The 44 test voices must stay locked until the final evaluation. A release must not make them easier to reach.

**The fix: a small voice pack, made once and pinned by revision.**

1. A new command, `python main_backend.py voice-pack build`, reads the voice manifest
   (`backend/data/manifests/voice-split-v2.json`). The name is not `voices`, which already exists and browses
   a dataset. The command and writes **only the 176 validation voices' chosen clips**:
   `<dataset>/<speaker>/reference.wav` (16-bit) and `reference.txt` for each, plus `pack.json`. `pack.json`
   holds each clip's id, dataset, speaker, length and sha256, the manifest's sha256, the date and the licences.
   This is about 30 MB instead of 24 GB. The test voices are **not** in it.
2. The pack goes to a **private** Hugging Face dataset, `Dramatizer/Serving_Voices`, uploaded the way
   `publish-results --hf` uploads (it checks that the dataset is private first). Each upload is a commit, and
   that commit's hash **is the data revision**. `python main_backend.py voice-pack pull <revision>` downloads
   that revision into `D:/Dramatizer voice pack/<revision>/` on the host and checks every file against
   `pack.json`. The pack is only built on the computer that has the full library; every other computer pulls it.
3. A new setting, `VOICE_PACK_DIR`. When it is set, `library()` reads the pack instead of the whole library, and
   `get_clip()` returns the pack's wav directly. There is nothing to extract and nothing to write. When it is not
   set, everything works as today, so research on the full library is unchanged.
4. Reference paths in runs are resolved by **clip id**, not by the stored path: look the clip id up in the pack,
   and use the stored path only when there is no pack. Old runs keep working, and no saved file is rewritten.
5. Licences allow this: VCTK and LibriTTS-R are CC BY 4.0, and CREMA-D is ODbL. Credit all three in the pack's
   `README.md`. Under ODbL, a changed copy of CREMA-D's database must keep ODbL. The pack stays private either
   way, because these are recordings of real people.

Check: on a computer **without** `D:/Voices library`, the site starts from the pack, lists the 176 voices and
renders one line.

### Problem 2: the models are large, downloaded from several places, and one is gated

| Model | Where it comes from | Size | Note |
|---|---|---|---|
| CosyVoice 3 code | GitHub checkout (`VOICE_MODEL_DIR`) | small | Not on PyPI: pin the **commit** |
| Fun-CosyVoice3-0.5B-2512 weights | Hugging Face or ModelScope (`COSYVOICE_MODEL_DIR`) | a few GB | Pin the **revision** (commit hash) |
| gemma-4-E4B-it | Hugging Face, into `LLM_LIBRARY_DIR` | 15 GB on disk | **Gated**: each account must accept Google's licence first |
| Whisper base | OpenAI's own download (not Hugging Face), into `~/.cache/whisper` | 140 MB | Downloaded on first use unless it is already there |
| SQUIM | torchaudio's download, into the torch hub cache | small | Downloaded on first use |
| Resemblyzer | Inside the pip package | small | Nothing to download |
| Qwen3-4B (emphasis, `EMPHASIS_MODEL`) | `LLM_LIBRARY_DIR` | several GB | Only if emphasis scoring runs in the served site |

**Decision: the weights are not inside the image.** They would make it 25 GB or more and slow every release.
Gemma's licence also does not let us publish it in an image. Instead:

1. A new command, `python main_backend.py models fetch`, downloads each model **by a pinned revision** into one
   folder on the host, `D:/Dramatizer models/` (the model store). It writes `models.json` with each model's
   source, revision and the sha256 of every file. Whisper and SQUIM are fetched into `models/cache/` the same
   way, so nothing is downloaded at start-up.
2. The container mounts the model store **read-only** at `/models`, and runs with `HF_HUB_OFFLINE=1`,
   `TRANSFORMERS_OFFLINE=1`, `XDG_CACHE_HOME=/models/cache` and `TORCH_HOME=/models/cache/torch`. If a model is
   missing, start-up **fails** loudly. It never downloads something unpinned in the background.
3. At start-up the app hashes what it loaded, cheaply (each model's `models.json` entry, checked by file size and
   the sha256 of its config file), and `/version` reports it.

Check: with the network off, the container starts, loads every model and `/version` lists the same revisions as
`models.json`.

### Problem 3: Hugging Face access (tokens, the gate, outages)

Four different uses need four different tokens. None of them ever goes into the repository or the image.

| Who | What for | Token | Kept where |
|---|---|---|---|
| The host | `publish-results --hf`, uploading the voice pack | **write**, fine-grained, only the `Dramatizer` organisation's datasets | `hf auth login` on the host only |
| The host | `models fetch` (Gemma is gated) | **read**, from an account that has accepted Gemma's licence | `hf auth login` on the host only |
| GitHub Actions | The tests that read `Dramatizer/Results_Data` or the pack | **read**, fine-grained, only those two datasets | A GitHub **secret**, `HF_READ_TOKEN` |
| Team members | `remote_results.py`, the browser | Their own **read** token | `DRAMATIZER_HF_TOKEN` on their own computer (`Access remote server.md`) |

Rules:

- **The running container needs no Hugging Face token at all.** Everything it reads is already on the host,
  pinned. This removes the biggest leak risk, because a token in a running service ends up in logs and crash
  dumps.
- Use **fine-grained** tokens limited to the named repositories. A classic "write" token can change every
  repository the account can reach.
- The gate: Gemma downloads fail with 401/403 until the account behind the token has accepted the licence on its
  model page. Whoever runs `models fetch` must do that once.
- Private datasets say "not found" (404) to anyone without access. A 404 from `Dramatizer/...` almost always
  means a missing token or organisation membership, not a missing dataset.
- **Outages and rate limits:** a release must not depend on Hugging Face being up. `models fetch` and
  `voice-pack pull` run **before** a release and keep their result on the host, and the release step only reads
  what is there. If Hugging Face is down, we keep serving and release later.
- Each token gets a reminder to rotate it every 90 days. If one leaks, delete it on huggingface.co at once;
  nothing else needs to change, because no image contains it.

---

## The steps

### Step 1: tests on every push (S–M)

`.github/workflows/ci.yml`, on pull requests and pushes to `main`:

1. `actions/checkout` and `actions/setup-python` (3.10), **pinned to full commit hashes**, not `@v4`.
2. `permissions: contents: read` only. Add `packages: write` only in the job that pushes the image.
3. Install `requirements.txt` without the CUDA extras. The tests that need a GPU, a dataset or a model must
   skip with a clear reason. `TESTS` in `main_backend.py` has a `needs` column, but it only skips a suite when a
   Python package is missing; a test that needs the voice library, a model or a GPU needs its own skip (for
   example when `VOICE_LIBRARY_DIR` is not set).
4. Run `python main_backend.py test`, `ruff check` and `pip-audit -r requirements.txt`.
5. A failing job stops the workflow: no image is built.

Check: open a pull request with a test broken on purpose. It gets a red check and no image. Save the run's link
in `reports/milestone4/01-ci/`.

**Status (23 Nov 2026): written, checked locally, not yet run on GitHub.**
- `.github/workflows/ci.yml`: one job; installs `requirements-ci.txt` (no PyTorch, no models), then
  `ruff check .` (rules in `ruff.toml`) and `python main_backend.py test`. Skips runs that only change documents.
- `.github/workflows/audit.yml`: `pip-audit -r requirements-ci.txt` weekly and when a requirements file changes.
- In a clean copy (no `.env`, no `history/`, no `D:` paths) with only `requirements-ci.txt`, every suite passes;
  7 tests skip because they need PyTorch or Resemblyzer, and still run on the host.
- Fixed on the way: `model.setting(name, "")` now returns "" instead of failing; the API answers 404 instead of
  500 when there is no voice library; the chatbot worker no longer holds its log file open (the occasional
  `test_assistant` failure on Windows).
- Not checked: Linux itself (the clean run was on Windows), and `pip-audit`'s result.

To verify first: the tests must pass on a clean checkout with **no `.env`** and no `D:` drive. Any test that
reads a real path needs a mock or a skip.

### Step 2: the image (M)

**Two images, so the large one is rarely built:**

- `dramatizer-base`: `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04` (pinned by digest), Python 3.10, sox,
  ffmpeg, torch cu121, `requirements.txt`, and CosyVoice's code at its pinned commit. About 8–10 GB. Rebuilt
  only when `requirements.txt` or the CosyVoice commit changes.
- `dramatizer`: `FROM dramatizer-base@sha256:…` plus the project's code. A few MB. Built on every merge to
  `main`.

Why two: GitHub's free runners have about 14 GB of free disk, so building torch and CUDA on every push risks
running out of space and takes long. If the base image does not fit on a hosted runner, build it on the host
with `docker build` and push it by hand. It changes rarely, and its digest is recorded either way.

`.dockerignore` must keep out: `.env*`, `certs/`, `history/`, `private/`, `measurements/`, `imported_voices/`,
`output/`, `**/__pycache__`, `*.wav` and every other audio file, `*.db`, `passwords_to_hand_out.txt` and
`history/api/test_users.txt`. CI then checks the built image: `docker run --rm <image> find / -name ".env*"
-o -name "*.pem" -o -name "*.db"` must find nothing in the app folder.

The image is pushed to GitHub's registry, `ghcr.io/<org>/dramatizer:<commit>`, and its **digest** is saved.
Releases and rollbacks always use the digest, never `:latest`.

**What the container mounts** (all paths are set in `deploy/container.env`, which is committed and has no
secrets):

| In the container | On the host | Mode | What |
|---|---|---|---|
| `/models` | `D:/Dramatizer models` | read-only | The model store (Problem 2) |
| `/voices` | `D:/Dramatizer voice pack/<revision>` | read-only | The voice pack (Problem 1) |
| `/output` | `D:/Dramatizer 2 renders` | read-write | Renders, searches, charts (`OUTPUT_DIR`) |
| `/state` | Docker **named volume** `dramatizer-state` | read-write | The database, accounts, uploads |
| `/run/secrets/` | `D:/Dramatizer secrets/` | read-only | The env file with secrets, the TLS certificate |

**Why the database is not on a Windows folder:** Docker Desktop shares Windows folders with Linux containers
through a layer whose file locks SQLite cannot rely on. Two containers writing one database there can corrupt
it. A named volume lives inside WSL's own disk, where locking works. Move it once with
`python main_backend.py storage backup`, then restore into the volume, and keep the daily backup (ROADMAP 0.5)
pointed at the volume.

**Paths in `.env` are Windows paths today** (`D:/...`). In the container they come from `container.env`
(`OUTPUT_DIR=/output`, `VOICE_PACK_DIR=/voices`, `COSYVOICE_MODEL_DIR=/models/cosyvoice/...`, and so on). Any
code that builds a path from a drive letter or uses `taskkill` must check the OS. `main_offline.py` is only
for debugging and stays Windows-only.

Check: `docker run --gpus all ...` starts the site from the image, a line renders, and the image holds no
secret.

### Step 3: the release record and `/version` (S)

`deploy/release.json` is written by the workflow for each candidate:

```json
{
  "release": "v8",
  "commit": "<git sha>",
  "image": "ghcr.io/<org>/dramatizer@sha256:…",
  "base_image": "ghcr.io/<org>/dramatizer-base@sha256:…",
  "models": {"cosyvoice": {"repo": "…", "revision": "<hash>"}, "gemma": {"repo": "…", "revision": "<hash>"},
             "whisper": "base", "squim": "torchaudio <version>", "resemblyzer": "<version>"},
  "voice_pack": {"repo": "Dramatizer/Serving_Voices", "revision": "<hash>", "manifest_sha256": "…"},
  "config_sha256": "<sha256 of container.env>",
  "made": "2026-11-…"
}
```

`GET /api/v1/version` (no key, like `/health`) returns what the process **actually loaded**: its commit (baked
into the image as a build argument), the pack's revision from `/voices/pack.json`, each model's revision from
`/models/models.json`, and the config hash. It never reports what was requested, and never a path or a secret.

Check: the smoke test (step 4) compares the two field by field.

### Step 4: health, readiness and the checks that gate a release (M)

- `GET /health`: is the process alive? It stays as it is today and only says `ok`.
- `GET /ready`: may it receive traffic? It answers `200` only when the voice model is loaded and the voice pack
  read; otherwise `503` with what is missing. The chatbot can still be `starting`, because the records answer
  until it is ready.
- `deploy/smoke.py <address>`, the correctness check. It exits with `0` to allow and `1` to block:
  1. `/ready` answers 200 within 5 minutes (CosyVoice and Gemma take about 2 minutes to load).
  2. `/version` matches `release.json` on every field.
  3. It logs in with the smoke account `smoke-test`, a listener made only for this. Its password is in the
     secrets folder, never printed.
  4. It renders one fixed line with one fixed voice and seed. The job finishes within 120 s, the audio is
     longer than 1 s, and its WER (Whisper) is at most 0.10.
  5. It asks the chatbot "How many voices are included?" and gets an answer in the expected form within 90 s.
     A records answer counts while the model loads.
  6. It writes `reports/milestone4/04-checks/<release>.json`: each check, its time and its result.
- **The blocked release (required evidence):** release a candidate whose `release.json` names the wrong voice
  pack revision. Check 2 fails, the script exits 1 and the switch never happens. Save the output.

### Step 5: release and rollback (M)

One GPU cannot hold two full copies. The computer uses about 5 GB with Gemma, and some of Gemma is already
offloaded to the CPU. So both versions cannot run fully loaded at once. The plan:

1. **A fixed front door.** Caddy (or nginx) listens on 5105, where the playit tunnel points, and forwards to
   whichever version is live: **blue** on 5106 or **green** on 5107. The proxy keeps the **same TLS
   certificate** as today, because clients check its fingerprint (`SHARING.md`). A new certificate would break
   every client.
2. **Release v8** (`deploy/release.ps1 <digest>`):
   1. Pull the image by digest.
   2. Pause the job queue on v7: it finishes the running job and takes new ones, but the queue does not start
      them.
   3. Stop v7's chatbot to free GPU memory. Start v8 on the free port.
   4. Run `smoke.py` against v8. If it **passes**: point the proxy at v8, then stop v7's container. Do not remove
      it.
   5. If it **fails**: stop v8, restart v7's chatbot, resume v7's queue. Users only noticed a pause.
3. **Roll back** (`deploy/rollback.ps1`): start v7's container again (its image is still there, by digest),
   wait for `/ready`, point the proxy back and stop v8. Expect about 2 minutes of cold start. Waiting jobs
   stay in the database.
4. **Why no gradual 5 % → 25 % → 100 % rollout:** with one GPU, a handful of users and both versions unable
   to be loaded at once, a canary gives no useful signal. We use **blue/green with a checked switch**, and
   write down that this was a choice.
5. **Database changes must work for both versions** (expand, then contract). v8 may add a column or table, but
   must not drop or rename one that v7 reads, until a later release when v7 can no longer come back.
   `storage migrate` runs **before** the switch, and v7 must still pass its smoke test after it.

Check: one release and one rollback done and logged, each with times, in `reports/milestone4/05-release/`.

### Step 6: limits under load (S)

Today a person can have 3 jobs waiting (429 after that), and the chatbot gives up after 90 s. There is no limit
on the whole queue, so 20 people × 3 jobs means 60 renders in a row.

1. `MAX_QUEUE = 12` jobs in all. When it is full, refuse at once with `503` and `Retry-After`, and the page says
   how long the wait is. 12 is a first guess; step 7 measures the real number.
2. `MAX_WAIT = 10 min`: a job not started by then is dropped with a clear message, and its slot is freed.
3. Cancel: a job whose owner cancels it, or whose page has stopped asking for it for 2 minutes, is dropped
   before it starts.
4. A test floods the queue and checks the fast refusals.

### Step 7: measure under load (M, GPU)

`deploy/load.py`, with the smoke account plus test accounts (passwords read from the file, never printed):

- Runs: 1, 5 and 20 users at once, for 10 minutes each. Each user renders a line every 30–60 s, and asks the
  chatbot every 2 minutes.
- For each run, report:
  - p50 and p95 time per render, split into waiting and working;
  - p50 and p95 chatbot answer time, and the time to its first word;
  - throughput (renders per minute) and errors (429, 503, timeouts);
  - the deepest the queue got;
  - GPU memory, from `nvidia-smi` every 5 s.
- **Capacity with Little's law:** jobs in the system = arrivals per second × seconds each job spends in the
  system. With the measured time per render and one render at a time on the GPU, this says how many active
  users the site can take, plus 30 % headroom for bursts and slow renders. That number sets `MAX_QUEUE`.
- Results: a table and one sentence ("the site serves N people rendering at once with p95 under M s") in
  `reports/milestone4/07-load/`.

### Step 8: the write-up (S)

In `reports/milestone4/README.md`: the chain **package → verify → promote → serve → observe → roll back**, with
the evidence of each step. Include the table at the top of this file (what replaced the course's tools, and
why), the three problems and how each was solved, and the limits: one GPU, no gradual rollout, a cold start
when rolling back.

---

## Order, and who can do what at the same time

```
Problem 1 (voice pack) ─┐
Problem 2 (model store) ┼─► Step 2 (image) ─► Step 3 (release record) ─► Step 4 (checks) ─► Step 5 (release, rollback) ─► Step 8
Problem 3 (tokens)  ────┘
Step 1 (CI) can start at once. Step 6 (limits) can be done alongside 2–5. Step 7 (load) needs step 6 and GPU time.
```

| # | Task | Size | Done when | Owner |
|---|---|---|---|---|
| P1 | Voice pack: `voice-pack build` and `pull`, `VOICE_PACK_DIR`, clip-id lookup, upload to `Dramatizer/Serving_Voices` | M | The site runs without `D:/Voices library` | |
| P2 | Model store: `models fetch`, `models.json`, offline start-up | M | The container starts with the network off | |
| P3 | Tokens: fine-grained read and write tokens, GitHub secret, Gemma licence accepted | S | Each use has its own token; the container has none | |
| 1 | CI: tests, ruff, pip-audit, pinned actions | S–M | A broken pull request is red and builds nothing | |
| 2 | Base and app images, `.dockerignore`, mounts, database in a named volume | M | The site runs from the image with the GPU | |
| 3 | `release.json`, `GET /version` | S | `/version` shows what was loaded | |
| 4 | `/ready`, `smoke.py`, a blocked release | M | One release allowed and one blocked, both on record | |
| 5 | Proxy, release and rollback scripts, migration rule | M | One release and one rollback logged | |
| 6 | `MAX_QUEUE`, `MAX_WAIT`, cancel | S | Flooding gives fast 503s | |
| 7 | Load runs at 1, 5 and 20 users, Little's law | M, GPU | A results table and the capacity sentence | |
| 8 | Write-up | S | `reports/milestone4/README.md` | |

## Risks to keep an eye on

- **Disk:** the base image (about 10 GB), the model store (25 GB or more with Gemma and Qwen) and old image
  versions add up. Keep the last 2 app images and the last base image; run `docker image prune` after each
  release.
- **The GPU driver:** Docker Desktop's GPU support needs WSL 2 and a recent NVIDIA driver. A driver update can
  break the container while the site outside Docker still works. Keep today's way of starting the site as the
  fallback until step 5 has been done twice.
- **Sealed data:** the Report evaluation's key and sealed runs must never be in the image, the pack or a
  published dataset. `publish-results` already refuses them; the voice pack check must refuse test voices the
  same way.
- **A Hugging Face outage on release day:** releases only read what is already on the host (Problem 3), so an
  outage delays the next release but never stops the running site.

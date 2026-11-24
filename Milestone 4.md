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
| Hosting, revisions, traffic | Azure Container Apps | The GPU computer, Docker, a small proxy, a Cloudflare tunnel on https://app.dramatizer.dev (and playit for the pinned clients) | The GPU is here |
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

**Status (24 Nov 2026): done.** Revision **`46a182c41bc13df2754560273a9224d53b6f8e26`** on the private dataset
`Dramatizer/Serving_Voices`, pulled back into `D:/Dramatizer voice pack/<revision>` and every file checked.
- `backend/voice_data/voice_pack.py`, command `voice-pack build | check | upload | pull REVISION`. Settings
  `VOICE_PACK_DIR` (the pack the site reads), `VOICE_PACK_STORE` (default `D:/Dramatizer voice pack`) and
  `VOICE_PACK_REPO` (default `Dramatizer/Serving_Voices`).
- Built into `D:/Dramatizer voice pack/build`: 176 voices, 37 minutes, **149 MB** (not 30 MB: the LibriTTS-R clips
  are about 20 s at 24 kHz). All 176 clips are byte for byte the ones the voice manifest checksummed; 0 test voices.
- `upload` refuses a public or missing dataset and a pack that fails `check`; the commit hash becomes the
  `revision` in `pack.json`, which `/version` and `release.json` report. `pull` downloads one revision and checks
  every file before recording it.
- `server.library()` reads the pack when `VOICE_PACK_DIR` is set; `get_clip` uses a pack wav as it is (nothing is
  written to `IMPORT_DIR`); `ratings.reference_path`, `preference` and `search` find a run's reference by clip id
  in the pack (`voice_pack.reference_of`). Tested with the library path pointing nowhere: 176 voices listed, each
  clip read from the pack, the saved run's reference found.
- `deploy/container.env` sets `VOICE_PACK_DIR=/voices`. 15 tests in `backend/voice_data/test_voice_pack.py`.
- A line rendered from a pack voice (VCTK p225): `D:/Dramatizer 2 renders/pack_check/`.
- The upload uses the fine-grained token `dramatizer-publish`, with write access to this dataset only.
- Not yet done: a pull on a second computer (the check above was on this one).

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

**Status (24 Nov 2026): done, on the host and in the container with the network off.**
- `backend/model_store.py`, command `models fetch [--only NAME] | check [--full] | pin NAME`, store
  `D:/Dramatizer models` (`MODELS_DIR`), laid out as the container mounts it at `/models`.
- The pins (each local copy matched file for file: LFS files by sha256, small files by their git blob hash):

  | Model | Pinned | Why |
  |---|---|---|
  | Fun-CosyVoice3-0.5B-2512 | `29e01c4e8d00` | every render |
  | gemma-4-E4B-it | `ee0ef6023621` (20 Jul 2026; the 3 Nov commit changed `tokenizer.json`, so the copy here is older) | the chatbot |
  | Qwen3-4B | `1cfa9a720891` | emphasis scoring |
  | all-MiniLM-L6-v2 | `1110a243fdf4` (the 6 files `rag/dense.py` loads) | dense retrieval |
  | Whisper base | sha256 `ed3a0b6b1c0e` (OpenAI's own, in its URL) | WER, upload check |
  | SQUIM objective | sha256 `2c54586fea83` (torchaudio publishes none: this computer's copy) | quality |
  | wetext (ModelScope `pengzhendong/wetext`) | 4 files by sha256 (en and zh `tn` tagger and verbalizer; ModelScope commits `887eec2` and `b04bc07` kept) | CosyVoice's text normaliser |

- `fetch` hard-links a matching local copy into the store (32 GB, but no extra disk space: the files are shared
  with `D:/LLM library` and `D:/Voice cloning`) and downloads anything else at the pin, refusing a file that does
  not match. It took 3 minutes here, all of it hashing; nothing was downloaded.
- With `MODELS_DIR` set, the server checks the store at start-up (sizes, seconds) and **refuses to start** if a
  model is missing or not at its pin. `/version` and `release.json` read the revisions from `models.json`.
- Checked with every network connection refused and `HF_HUB_OFFLINE=1`: the start-up check passed; Whisper,
  SQUIM, MiniLM and CosyVoice 3 loaded from the store; Gemma's config and tokenizer loaded in its own environment.
- **Found by the container (24 Nov): a seventh model nobody had listed.** CosyVoice's text normaliser (wetext,
  which turns "12" into "twelve") fetches its whole ModelScope repository (236 MB, every language) each time
  CosyVoice loads. ModelScope now refuses anonymous requests, so in a fresh container the download stalled and
  CosyVoice went on with **no normaliser** ("no frontend is avaliable"), silently, with `/ready` still 200. The
  host only worked because its ModelScope cache was filled weeks ago. Now: the 4 files CosyVoice loads are in the
  store (`cache/wetext`, linked from that cache), each pinned by sha256 (they are not on Hugging Face), and
  `WETEXT_DIR` makes `model._wetext_from_cache` hand wetext that folder: it never calls ModelScope.
- **The container check (24 Nov), `docker run --network none`** (no network at all: Hugging Face does not even
  resolve): the start-up check passed, `/ready` answered 200 about a minute after start, the log says "use wetext
  frontend" and nothing was downloaded, `/version` lists all 7 models with exactly the revisions in the host's
  `models.json`, and a line rendered offline with Whisper's WER 0.000 (Whisper too loaded from the store).
- 13 tests in `backend/test_model_store.py`.
- Not yet done: switching this computer's `.env` to the store (optional: the store holds the same files; add the
  lines `models check` prints, now including `WETEXT_DIR`).

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

**Status (23 Nov 2026): done, green on GitHub** (CI run #3, commit e177638; Audit runs #2 and #3).
- Validated on GitHub: lint and 277 tests in 24 files on Linux (7 skips: torch and Resemblyzer); the token is
  `contents: read`; Audit is skipped when no requirements file changes, and its manual button works; a lint
  error (run #1), a failing test (run #2) and a vulnerable package (Audit #1) each turned the run red.
- Fixed on GitHub: ruff 0.17 turns on many more rules by default, so `ruff.toml` now lists them; pyarrow 23.0.1;
  pillow and filelock are accepted in `audit.yml` with the reason (their fixes need Python 3.11); actions on
  Node 24; `search.list_runs` gave runs with the same start time in a different order on Linux.
- Not yet done: the pull request with a test broken on purpose, saved in `reports/milestone4/01-ci/`.

*Earlier status, before the first run on GitHub:*
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

**Status (24 Nov 2026): built and checked on this computer; not yet pushed to ghcr.io, and the live site still
runs outside Docker.** Docker Desktop 29.8.2 on WSL 2, with the NVIDIA runtime: `nvidia-smi` in a CUDA container
sees the RTX 3080 (driver 621.14).

| | Result |
|---|---|
| Base image `dramatizer-base:2026-11-24` | built in 44 min; **17.3 GB** unpacked (the venv 7.5 GB, CUDA + cuDNN about 8 GB, apt 0.4 GB), not the 8-10 GB first guessed |
| App image `dramatizer:dev` | seconds; 155 files, 9 MB in `/app`; runs as `dramatizer` (uid 10001); no `.env`, key, certificate, database, token, password or audio file in it |
| Start (mounts as in the table above, test volume `dramatizer-state-test`, not the live `history/`) | `/ready` 200 after 70 s (CosyVoice loaded, 176 voices read, chatbot off); `/version` shows the voice pack and model pins |
| One line (VCTK p225, seed 7, the smoke test's line) | rendered into `D:/Dramatizer 2 renders`; Whisper's WER **0.000** |
| Speed, warm, same line, voice and seed | **14.0 s in the container, 13.6 s on the host**: the same. CosyVoice loads in 72 s there (122 s on the host) |
| Network off (`--network none`) | see problem 2: starts, loads everything, renders |

What the first runs found and what changed:
- **numba could not cache** (librosa compiles with numba, which writes its cache next to the package in
  `/opt/venv`, read-only for the server's user): CosyVoice failed to load ("no locator available").
  `deploy/container.env` now sets `NUMBA_CACHE_DIR=/tmp/numba` and `MPLCONFIGDIR=/tmp/matplotlib`.
- **CosyVoice's text normaliser came from ModelScope at every load** and was silently missing in a fresh
  container: now in the model store (problem 2), `WETEXT_DIR=/models/cache/wetext`.
- onnxruntime-gpu 1.18 is built for CUDA 11, so it logs that it cannot load its CUDA provider (`libcublasLt.so.11`)
  and runs CosyVoice's two small ONNX models on the CPU. The host does the same (Windows uses the CPU build), so
  nothing changes; only the log is noisier.

Still to do here: push the base image and record its digest in `deploy/base-image.txt` (CI then builds and pushes
the app image), move the database into the named volume `dramatizer-state`, and run the site from the image.

As first written (23 Nov):
- `docker/base.Dockerfile`: CUDA 12.1.1 + cuDNN 8 runtime (pinned by digest), Python 3.10, `requirements.txt`
  without deepspeed and TensorRT (CosyVoice only uses them when asked), CosyVoice at commit `074ca6d` with its
  Matcha-TTS submodule. Two stages, so the compilers stay out of the image.
- `docker/app.Dockerfile`: the code on the base, by digest; runs as user 10001, the code belongs to root;
  `history/` (database, accounts, keys, certificate, logs) is the volume `dramatizer-state`.
- `.dockerignore` is an allow list: only `backend/`, `frontend/client/`, `API calls/`, `main_backend.py`,
  `requirements.txt` and `docker/`, and even there no `.env*`, keys, databases, audio, weights or logs.
- `deploy/container.env`: the paths inside the container, no secrets. `API_HOST` is new (default `127.0.0.1`,
  `0.0.0.0` in the container, published as `-p 127.0.0.1:5005:5005`).
- CI job `image` (after `tests`, on `main` only): builds the app image on the base named in
  `deploy/base-image.txt`, fails if `/app` holds a secret or data file, pushes to ghcr.io and writes the digest
  in the run's summary. Only this job has `packages: write`. It does nothing until that file exists.
- Left for later: the chatbot (Gemma 4) needs its own Python 3.11 environment with newer torch and transformers,
  so it is off in the container (`CHATBOT_PYTHON=`). (The voice pack and the model store now exist and are what
  the container mounts.)

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

**Status (23 Nov 2026): the code is done and tested; no real release record yet** (that needs the image digest).
- `backend/release_info.py`: the fields both sides use: commit (`GIT_COMMIT` in the image, git's HEAD on the
  host), the voice pack's revision from `pack.json`, each model's revision from `models.json` (both `None` until
  problems 1 and 2 are done), and `config_sha256` over a fixed list of settings that are never secret.
- `GET /api/v1/version` (no key): those fields, plus the models loaded so far and the start time.
- `deploy/release.py <name> --image … --base … --env deploy/container.env --voice-pack … --models …` writes
  `deploy/release.json`. Images must be named by digest. Host paths are read, never written into it.
- Checked live on a second copy of the API (port 5107): a matching record gave no differences, a wrong pack
  revision was caught as `voice_pack`, one changed setting as `config_sha256`, and `/version` held no path.
- Known limit: on the host, `commit` is HEAD even when files are changed and not committed. In the image it is
  exact.

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

**Status (23 Nov 2026): the code is done and tested; not yet run against a real candidate.**
- `GET /api/v1/ready`: 200 only with CosyVoice loaded and the voices read, else 503 and `missing`. The chatbot is
  reported, not required. `PRELOAD_VOICE_MODEL` (default on) loads CosyVoice at start-up; a failed load is shown.
- `deploy/smoke.py <address>`: the five checks above. A failed check skips the ones that need it, writes
  `reports/milestone4/04-checks/<release>.json`, and exits 1. The certificate is checked by fingerprint, and the
  password is read from a file and never printed.
- `deploy/test_release.py` (13 tests), including the blocked release (a wrong voice pack revision fails
  `version` and blocks), a bad render (WER 0.4, or a failed job), an answer with no number, and no password file.
- Checked live on port 5107: `/ready` answered 503, voice model not loaded, 176 voices read.
- Still to do: the `smoke-test` account (`python main_backend.py account add smoke-test --name "Smoke test"
  --role listener`, its password saved to `D:/Dramatizer secrets/smoke-test.txt`), then a real run.

**Status (24 Nov 2026): run against real containers.** Staging account `smoke-test` (password only in
`D:/Dramatizer secrets/staging/`). Four candidates, results in `reports/milestone4/04-checks/`:
- **v1 blocked by a real bug:** on a fresh site with no ratings yet, `/assistant/ask` answered 500
  (`rag/corpus.build` opened `who_rated_what.csv`, which did not exist). Fixed in 72027de, with a test.
- **v3-wrong-pack blocked:** its record named voice pack revision `0f1e2d3…`; the container loaded `46a182c…`, so
  `version` failed (the required blocked release).
- v2 and v3 passed all five checks: ready in about 75 s, render WER 0.000, the chatbot (off in the container)
  answered from the records.
- `smoke.py` now reads the password file as `account add --password-file` writes it (`smoke-test<TAB>password`).
- `release.py` accepts a local image ID (`sha256:…`) until the images are pushed.

### Step 5: release and rollback (M)

One GPU cannot hold two full copies. The computer uses about 5 GB with Gemma, and some of Gemma is already
offloaded to the CPU. So both versions cannot run fully loaded at once. The plan:

1. **A fixed front door.** Caddy (or nginx) listens on 5105, where the playit tunnel and the Cloudflare tunnel
   (`app.dramatizer.dev`, set up 23 Nov 2026 to 127.0.0.1:5105) point, and forwards to
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

**Status (24 Nov 2026): one release, one blocked release and one rollback done, on a staging pair.** Log and
times in `reports/milestone4/05-release/` (README there).
- `deploy/front_door.py`: a raw TCP forwarder, so TLS passes through and clients keep the API's certificate (both
  slots share one state volume, so one certificate). It reads the live port from a state file on every new
  connection: a switch needs no restart. Used instead of Caddy, which would have to hold the key.
- `deploy/blue_green.py release|rollback|status` (8 tests): starts the candidate in the idle slot with a frozen copy
  of its env file, runs `smoke.py`, switches only on a pass, stops (keeps) the old one. `deploy/staging.ps1` runs
  it with the staging pair's settings (volume `dramatizer-state-staging`, ports 5106/5107, front door 5108).
- Times: a release about 2 minutes, a rollback 80 s (cold start). With two copies on the GPU a smoke render took
  101 s instead of 7 s. Containers now run with `--init` (the server ignored `docker stop` as PID 1).
- Correction to the plan above: the job queue is in each server's memory, not the database, so a job still waiting
  in the old version at the switch is lost. Switch when the queue is empty.
- Not yet done: the live site still runs outside Docker on port 5005. Moving it behind the front door (port 5105,
  where the tunnel points) needs the database moved into the volume `dramatizer-state` first.

### Step 6: limits under load (S)

Today a person can have 3 jobs waiting (429 after that), and the chatbot gives up after 90 s. There is no limit
on the whole queue, so 20 people × 3 jobs means 60 renders in a row.

1. `MAX_QUEUE = 12` jobs in all (now 4, from step 7). When it is full, refuse at once with `503` and `Retry-After`, and the page says
   how long the wait is. 12 is a first guess; step 7 measures the real number.
2. `MAX_WAIT = 10 min`: a job not started by then is dropped with a clear message, and its slot is freed.
3. Cancel: a job whose owner cancels it, or whose page has stopped asking for it for 2 minutes, is dropped
   before it starts.
4. A test floods the queue and checks the fast refusals.

**Status (23 Nov 2026): done in the code, tested; not yet on the running site** (needs an API restart).
- `server.py`: `MAX_QUEUE` (setting, default 12, queued and running together) refuses with `503` and
  `Retry-After` (seconds until the running job should end, or the oldest queued job expires). The per-person
  limit (3, `429`) is checked first.
- `MAX_WAIT_SECONDS` (default 600): a job not started by then becomes `expired` and frees its slot at once.
- A render or Lab job whose page stopped asking about it for 2 minutes becomes `cancelled` before it starts.
  Genome searches are exempt: they are started and left to run, and nobody polls them.
- `DELETE /api/v1/jobs/<id>`: the owner takes a queued job out (someone else's job answers 404; a running one 409,
  because the GPU finishes what it started). The page shows a Cancel button while a job waits.
- Each queued job now has `expected_start_s`, from the median time of the last 20 jobs of each kind (a first
  guess until then), and the page says "number 3 in the queue, about 75 s to go".
- `backend/api/test_queue.py` (9 tests): 200 refusals after a full queue take well under a second; the 503 carries
  `Retry-After`; expiry frees a slot; abandoned renders go but searches stay; the worker skips dropped jobs; only
  the owner can cancel; the wait estimate.

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

**Status (24 Nov 2026): measured.** `deploy/load.py` (5 tests in `deploy/test_load.py`), accounts `load-01` to
`load-20` (passwords only in `D:/Dramatizer secrets/load-users.txt`; agreement accepted for them by the host; disabled
after the run). Against the API on this computer (port 5005), with Gemma loaded; results in
`reports/milestone4/07-load/20261124/`.

| People | Renders/min | Render p50 / p95 (s) | Waiting p50 / p95 | GPU work p50 / p95 | 503 | Deepest queue | Chatbot p50 / p95 (s) | Chatbot from the model |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.3 | 9.4 / 18.1 | 0.5 / 0.9 | 9.3 / 18.0 | 0 | 1 | 27 / 50 | 5 of 5 |
| 5 | 4.1 | 84.3 / 95.1 | 65.9 / 78.8 | 17.3 / 22.3 | 0 | 4 | 90 / 90 | 2 of 25 |
| 20 | 4.6 | 205.5 / 248.7 | 188.6 / 231.6 | 17.4 / 23.7 | 107 | 11 | 90 / 110 | 0 of 95 (63 timed out) |

- **The site serves 2 people rendering at once** with renders done in about 18 s (p95): Little's law with 30 %
  headroom, from the most the GPU finished when saturated (4.6 renders a minute, one every 13 s).
- **The GPU is shared:** a render takes 9.3 s alone but 17.3 s while the chatbot answers, and the chatbot, which
  answers in 27 s alone, hits its 90 s limit under load and falls back to the records. With 20 people, 63 of 95
  questions timed out at the client's 120 s. The bottleneck is one GPU doing both jobs.
- **The limits held:** no render failed, expired or timed out; at 20 people the queue filled (11 of 12) and 107
  requests got a fast 503 instead of waiting. But with `MAX_QUEUE=12`, an accepted render could wait 4 minutes.
- **MAX_QUEUE:** Little's law gives 2 for 2 people; the 5-person run needed 4 with no refusals but waits of about
  80 s. Proposed: `MAX_QUEUE=4` (at most about a minute's wait at 13 s a render), down from 12.

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

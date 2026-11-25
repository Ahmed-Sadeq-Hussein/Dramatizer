# Milestone 4: a checked, usable release

Group 2 (Ahmed Hussein, Julius Norén, Rama Muharam), 24 Nov 2026. The full plan and the notes made while doing each
step are in [`Milestone 4.md`](../../Milestone%204.md); this page is the summary, with the evidence for each step.

**What was asked:** one fixed version of Dramatizer, a release that is checked before anyone uses it, and useful
answers under a measured load. Plus three pieces of evidence: a release that a failed check **blocked**, the
versions the running site **loaded**, and a **way back** to the previous version.

**The chain:** package → verify → promote → serve → observe → roll back.

## What replaced the course's tools, and why

| Responsibility | Course | Dramatizer | Why |
|---|---|---|---|
| API | FastAPI | Flask (`backend/api/server.py`) | Already built and tested; replacing it adds risk and proves nothing new |
| Model runtime | Ollama on a CPU | CosyVoice 3 and Gemma 4 (4-bit) on the GPU | Voice cloning cannot run usefully on a CPU |
| Image | Docker | Docker with the NVIDIA runtime | Same |
| Workflow | GitHub Actions | GitHub Actions | Same |
| Hosting, revisions, traffic | Azure Container Apps | The GPU computer, Docker, our front door, a Cloudflare tunnel (https://app.dramatizer.dev) | Azure Container Apps' CPU plan has no GPU |
| Data and model store | Blob Storage | A private Hugging Face dataset and a model store, each pinned by revision | The team already uses Hugging Face |

## The three problems solved first

1. **Data that is not in the repository.** The site read 24 GB of voice datasets to find 176 clips. Now a
   **voice pack** holds only the 176 validation voices (149 MB, every clip checksummed), on a private dataset at
   revision `46a182c`; the 44 test voices are never in it. A fresh `voice-pack pull` from Hugging Face took 29.5 s
   and checked all 355 files.
2. **Large models from several places, one gated.** A **model store** holds every model at a pinned revision,
   checked by hash. The site refuses to start if a model is missing or wrong, and never downloads anything at
   start-up: the container starts with the network off (`--network none`).
3. **Hugging Face access.** One fine-grained token per use (write for uploads, read on the host, a read secret
   for CI), and **no token at all in the running container**. A Hugging Face outage delays the next release; it
   never stops the running site.

## The steps and their evidence

| # | Step | Evidence |
|---|---|---|
| 1 | **Package/verify: tests on every push.** Lint and 277+ tests on Linux; actions pinned by commit; read-only token; weekly `pip-audit`. | Green on `main` (CI #5). A pull request with a test broken on purpose went red and built no image: [`01-ci/`](01-ci/README.md), pull request #7. |
| 2 | **Package: the images.** A base (CUDA, PyTorch, CosyVoice, the chatbot's Python 3.11 environment), rebuilt rarely, and a 9 MB app image on top. No secret, key, database or audio file inside; runs as an ordinary user. | Built and checked on the GPU computer: a line rendered in the container with WER 0.000, 14.0 s vs 13.6 s on the host. |
| 3 | **Observe: the release record and `/version`.** Each candidate has a record of its exact image, commit, voice pack, model revisions and settings hash; `/version` reports what the process actually loaded. | `05-release/records/`; `/version` matched each record field by field. |
| 4 | **Verify: the checks that gate a release.** `/ready`, and `deploy/smoke.py`: ready, versions match, log in, render a fixed line (WER ≤ 0.10), the chatbot answers. | [`04-checks/`](04-checks): **v1 blocked by a real bug** (the chatbot crashed on a fresh site; fixed), **v3-wrong-pack blocked** by the version check; v2 and v3 allowed. |
| 5 | **Promote/roll back: blue/green behind a front door.** The candidate starts beside (or, with the chatbot on, instead of) the live version; the front door moves only after the smoke test passes; the old container is kept for a rollback. | [`05-release/`](05-release/README.md): on staging, two releases, two blocked releases and one rollback (84 s). On the live site, two releases, one blocked release and one rollback (77 s). Each is logged with times. |
| 6 | **Serve: limits under load.** A whole-queue limit (now 4) with a fast 503 and `Retry-After`, a 10-minute maximum wait, cancel, and an honest wait estimate. | 9 tests (`backend/api/test_queue.py`); at 20 people the queue filled and 107 requests got a fast refusal. |
| 7 | **Observe: measured under load.** 1, 5 and 20 simulated people for 10 minutes each. | [`07-load/`](07-load/20261124/README.md): **2 people rendering at once**, renders done in about 18 s (p95). The bottleneck is one GPU doing renders and the chatbot. |

## The live site

Since 24 Nov 2026 the live site runs **in Docker**, the same way as staging: slots on 5106 (blue) and 5107 (green),
behind `deploy/front_door.py` on 5005 (playit and the pinned clients) and 5105 (the Cloudflare tunnel to
https://app.dramatizer.dev). The site's accounts, ratings and history were moved once into the volume
`dramatizer-state` with `deploy/move_state.py`. Every table's row count matched afterwards, and the sealed evaluation
data and the password lists were left out. The certificate is the same one the clients pin (fingerprint
`b00a5ab5…`). Releases and rollbacks use `deploy/live.ps1`. It adds `--free-gpu`, because CosyVoice and the
chatbot together fill the 10 GB GPU, and it mounts the team's own voices.

| Time | Action | Candidate | Result | Why |
|---|---|---|---|---|
| 18:05 | release | v5 (38f2a78), first record | **blocked** | The version check: the record's settings hash was `f55b799…`, the container's `87d6faa…`. The release tool set `VOICES_DIR` outside the env file the record was made from. Nothing had been live yet, so nothing moved. Fixed in 734b92e: the record and the container now read the same two files (`container.env`, then `live.env`). |
| 18:10 | release | v5 (38f2a78) | **released** (99 s) | All 5 checks passed. Through the front door, both 5005 and 5105 reported commit `38f2a78`, settings `87d6faa`, pack `46a182c`. The chatbot's model (Gemma 4) was loaded 66 s after ready and answered from the model in 20 s. |
| 18:15 | release | v6 (734b92e) | **released** (105 s) | v5 was stopped to free the GPU (the site paused), v6 passed all 5 checks, and the front door moved to green. |
| 18:16 | rollback | back to v5 | **rolled back** (77 s) | v5's container started again, and the front door moved back to blue once it was ready. Both ports then reported v5's commit. |

Logs: `05-release/20261124-18*.json`; records: `05-release/records/v5*.json`, `v6.json`; checks: `04-checks/`.
The host's `python main_backend.py api` is no longer started, so the host and the volume do not drift apart.

## Limits, written down on purpose

- **One GPU.** It serves about 2 people rendering at once. Renders and the chatbot slow each other down.
- **No gradual rollout.** One 10 GB GPU cannot hold two loaded copies, so a 5 % → 100 % canary would give no
  signal; we use a checked switch instead.
- **A release or rollback means a cold start** of about 80 s (CosyVoice loading). With the chatbot on, the live
  version is stopped while a candidate is checked, so the site pauses for about 2 minutes.
- **The queue lives in memory.** A job still waiting at the moment of a switch is lost; switch when it is quiet.
- **Images are named by digest only once pushed.** Until the base image is on ghcr.io, release records name images
  by their local content ID (`sha256:…`), which is just as exact on this computer.

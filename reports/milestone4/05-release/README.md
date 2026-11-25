# Release and rollback, 24 Nov 2026 (staging)

Run on this computer with `deploy/blue_green.py` (via `deploy/staging.ps1`), behind the staging front door
`deploy/front_door.py` on port 5108. The staging pair has its own state volume (`dramatizer-state-staging`), ports
5106 (blue) and 5107 (green) and render folder, so the live site was not touched. Each candidate was checked by
`deploy/smoke.py` (results in `../04-checks/`). Images are named by their local content ID (`sha256:…`) until they
are pushed to ghcr.io. Release records: `records/`.

| Time | Action | Candidate | Result | Why |
|---|---|---|---|---|
| 16:29 | release | v1 (commit 885cda7) | **blocked** | The chatbot check got a 500: on a fresh site with no ratings yet, the records index opened a file that did not exist. A real bug the host never showed; fixed in 72027de. |
| 16:31 | release | v2 (72027de) | **released** (97 s) | All 5 checks passed; the front door moved to blue. |
| 16:35 | release | v3-wrong-pack (425743a) | **blocked** (194 s) | Its record names voice pack revision `0f1e2d3…`; the container loaded `46a182c…`. The version check failed, the candidate was stopped and v2 stayed live. |
| 16:37 | release | v3 (425743a) | **released** (121 s) | All 5 checks passed; the front door moved to green, v2 was stopped and kept. |
| 16:38 | rollback | back to v2 | **rolled back** (84 s) | v2's container started again, was ready after 80 s, the front door moved back to blue, v3 was stopped. `/version` through the front door then showed v2's commit. |

What the runs showed:

- **The gate works both ways.** It blocked a candidate for a real bug (v1) and for a wrong data revision
  (v3-wrong-pack), and in both cases the front door never moved.
- **A release takes about 2 minutes,** almost all of it CosyVoice loading (about 75 s) and the test render.
  **A rollback takes about 80 s**: a cold start, because one GPU cannot keep two copies loaded.
- **Two copies on one GPU slow each other down.** While v2 was live and v3-wrong-pack was being checked, the smoke
  render took 101 s of GPU time instead of about 7 s, close to the 120 s limit. A release should stop the live
  version's work first (or happen when the site is quiet).
- **Containers now stop cleanly.** The first rollback's `docker stop` had to kill the server (exit 137: as PID 1
  it ignored the stop signal). Containers now run with `--init`, and stop in under a second.
- **v3 differs from v2 only by its commit** (a helper script outside the image). The point was the release and
  rollback mechanics, not a code change.

## Later the same day

| Time | Action | Candidate | Result | Why |
|---|---|---|---|---|
| 17:55 | release | v4-staging (5d01b49), chatbot in the image | **blocked** | `/ready` took over 300 s. CosyVoice and Gemma (15 GB, read over the Windows share) loaded at the same time and filled the 15.6 GB WSL memory. Fixed in 38f2a78: Gemma now loads after CosyVoice. |
| 18:01 | release | v5-staging (38f2a78) | **released** | All 5 checks passed. Gemma was loaded 65 to 86 s after ready and answered from the model in about 17 s. |

The live site's releases and rollback (18:05 to 18:16) are in [`../README.md`](../README.md#the-live-site).

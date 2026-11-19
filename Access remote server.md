# Access remote server

The Dramatizer's results (no audio) are hosted online in a private Hugging Face dataset, so the team can read
them while the host's PC is off. This document explains how to read them, in a browser or from code, and how
the host sends updated data.

- **Address:** <https://huggingface.co/datasets/Dramatizer/Results_Data>
- **In code:** `hf://datasets/Dramatizer/Results_Data` (the default of `remote_results.py`)
- **Who:** team members only (researchers and admins), as members of the `Dramatizer` organisation on
  Hugging Face. The dataset is **private**: anyone else gets "not found".
- **What it holds:**
  - every listener's scores, and every score as it was given or changed;
  - every run and genome (genes, markup, verdict, the sanity check), and who was given which pile;
  - the evidence snapshot (coverage, winners only with enough evidence, artifact checks, caveats);
  - the evidence records that answers and search are made from;
  - the six questions' charts;
  - the baseline and Milestone 2 summaries.
- **What it does not hold:**
  - audio;
  - the sealed Report evaluation, until its key is opened;
  - login names, emails, passwords, API keys, and test accounts' scores.
- **How new:** as new as the host's last publish. Every copy is stamped with the data it holds (see
  [Which data am I reading?](#which-data-am-i-reading)).

| You want to | Use |
|---|---|
| Get access (once) | [Getting access](#0-getting-access-once) |
| Look at the results | [The browser](#1-in-a-browser) |
| Ask questions from Python, a script or a chatbot, as with the API | [Python](#2-from-python-the-same-as-an-api-request) |
| Read the files from any other language | [Plain HTTP](#3-from-any-language-plain-http) |
| Put new results online (the host) | [Sending updated data](#4-sending-updated-data-the-host) |

---

## 0. Getting access (once)

1. **Make a Hugging Face account** at <https://huggingface.co/join>, if you have none.
2. **Ask the host to invite you** to the `Dramatizer` organisation, and accept the invitation (from the
   email, or from huggingface.co).
3. **For code only, make your own read token:** <https://huggingface.co/settings/tokens> > Create new token >
   **Read**. Copy it. Keep it private, like a password, and never put it in a file that is committed.

## 1. In a browser

1. Log in at huggingface.co.
2. Open <https://huggingface.co/datasets/Dramatizer/Results_Data>.
3. Open the **Files and versions** tab. Click a file to read it; the charts (`charts/q1_…jpg` … `q6_…jpg`)
   show as images. The **History** button lists every publish, each named after its `snapshot_id` and stamps.

`scores.csv` is the one to open for all the scores. A file's **download** button saves it, for example to open
`scores.csv` in Excel.

## 2. From Python, the same as an API request

`API calls/remote_results.py` asks the same questions as `API calls/API_REQUEST.py`. It uses the same
function names and gives the same answers, but it reads the Hugging Face copy instead of a running API.
- **Same answers, checked:** on the same data, the routes give identical answers from the copy and from the
  live API, because both use the same code (`backend/evidence_views.py`).
- **Every file is checked:** each file is compared with the copy's manifest, so a half-uploaded copy is caught.

### Once

1. **Get the code:** clone the repository (`https://github.com/Ahmed-Sadeq-Hussein/Dramatizer`) or pull the latest version.
2. **Install the packages it needs** in any Python 3.10+ environment:

   ```bash
   pip install numpy python-dotenv huggingface_hub
   ```

3. **Log in with your read token**, once per PC. Paste the token when it asks (it is not shown) and answer
   `n` to the git credential question:

   ```bash
   hf auth login
   ```

   Or, instead of logging in, set the token as an environment variable and open a new terminal:

   ```powershell
   setx DRAMATIZER_HF_TOKEN "<your read token>"
   ```

### Asking questions

From Python, exactly as with `API_REQUEST.Results`:

```python
import sys
sys.path.insert(0, "API calls")
from remote_results import connect

results = connect()                                      # Dramatizer/Results_Data, with your saved login
results.evidence_status()                                # the snapshot: coverage, per voice a winner or why not
results.evidence_status(voice="VCTK/p250")               # one voice in full
results.query_scores(group_by="gene_type", sort="mean")  # scores filtered, sorted or grouped
results.evidence_question(1)                             # one of the six questions, with its chart's address
results.search_evidence("sigh above the plain line")     # the records that best match some words
results.answer_evidence("Which genomes beat the plain line?")   # a cited answer, or "nothing answers this"
results.baseline_results()                               # the baselines' measured results
results.milestone2_results()
results.about_dramatizer()
```

For a chatbot, the tools are the same as the API's (`API_REQUEST.TOOLS`). Give the model those tools and run
its choice with `results.run_tool(name, arguments)`.

From the command line:

```bash
python "API calls/remote_results.py" evidence_status
```

```bash
python "API calls/remote_results.py" query_scores "{\"group_by\": \"listener\"}"
```

```bash
python "API calls/remote_results.py" manifest
```

**The files themselves** (for example every score, which the questions above do not cover, see below):

```python
import csv, io
rows = list(csv.DictReader(io.StringIO(results.api.read("scores.csv").decode("utf-8"))))
history = results.api.read("rating_history.csv")
runs = results.api.read_json("runs.json")
```

### What it answers, and what needs the live API

| Answered from the hosted copy | Needs the live API (the host's PC) |
|---|---|
| `evidence_status`, `query_scores`, `evidence_question`, `search_evidence`, `answer_evidence`, `baseline_results`, `milestone2_results`, `about_dramatizer` | `overview`, `list_searches`, `search_details`, `top_genomes`, `named_genomes`, `listening_piles`, `pile`, `renders`, `voices`; rating; rendering; questions over a date range (`from_date`, `to_date`) |

A question that needs the live API returns an error that says so; it never returns a wrong answer.
Answers show every listener's scores, as an admin sees them.

`query_scores`, search and answer read the evidence records, which cover the mos-diverse-v1 batch, exactly
as the live API does. **For every score of every batch, read `scores.csv`.**

### Reading one exact copy, or a local copy

- **One publish**, for example the one a reported number came from: add its commit id,
  `connect("hf://datasets/Dramatizer/Results_Data@<commit id>")`. The History button lists them.
- **A folder** the host gave you: `connect("C:/path/to/results_site")`.

## 3. From any language: plain HTTP

Every file is a plain GET with your read token in one header:

```bash
curl -L -H "Authorization: Bearer $DRAMATIZER_HF_TOKEN" https://huggingface.co/datasets/Dramatizer/Results_Data/resolve/main/manifest.json
```

Replace `main` with a commit id to read one exact publish. Without a valid token the answer is 401 or 404.

| File | What it is |
|---|---|
| `manifest.json` | The stamps, and every file with its size and sha256 |
| `index.html` | A summary page (stamps, snapshot, charts), to download and open |
| `snapshot.json` | The full evidence snapshot |
| `scores.csv` | Every listener's current score of every clip, with the artifact answer, dates, number of changes, verdict and markup. **All** scores (every batch); the request tools above read only the evidence records (mos-diverse-v1), as the live API does |
| `rating_history.csv` | Every score as it was given or changed, oldest first, with the score before it |
| `runs.json` | Every run that is not sealed, rated or not: voice, batch, pile, method, seeds, line and reference text, what Whisper heard of the plain line, and every genome in full (genes, operations, markup, speed, warnings, verdict, length, and the sanity check: word error rate, speaker similarity, what Whisper heard, pace). No audio |
| `piles.json` | Which listener was given which pile (sealed piles left out) |
| `evidence_records.jsonl` | The evidence records, one JSON per line |
| `genomes.json` | What each genome in the evidence records does, and its markup |
| `charts/charts.json`, `charts/q1_…jpg` … `q6_…jpg` | The six questions: claims, notes and charts |
| `results.json` | The baseline and Milestone 2 summaries |

## Which data am I reading?

Every copy is stamped. The stamps are in `manifest.json`, in each publish's commit message, and in every
answer's `provenance`:

| Stamp | Meaning |
|---|---|
| `snapshot_id` | A hash of the evidence snapshot: the same id means the same analysis |
| `ratings_up_to` | The newest score the snapshot counts |
| `data_revision` | The newest change in the evidence records that answers and search use |
| `made` | When the host built this copy |
| `code_commit` | The code that built it |

Quote the `snapshot_id` when you report a number, so others can tell which data it came from. The live
API gives the same stamps, so you can compare an answer from the copy with one from the API.

---

## 4. Sending updated data (the host)

Only the host publishes, because the scores, runs and renders live on the host's PC. Team members rate on
the website as before; their scores reach the online copy at the host's next publish.

### Each time (after a rating session, or whenever the results change)

```bash
python main_backend.py publish-results --hf Dramatizer/Results_Data
```

This does four things:
- builds the copy in `D:/Dramatizer 2 renders/results_site/` and checks it. It **refuses to publish** if the
  copy would hold a sealed run, the condition key, a test account, an email, a credential or a local path.
  It never copies audio;
- checks that the dataset is private;
- uploads it as one commit, named after its `snapshot_id` and stamps;
- removes the previous copy's files that are no longer in it.

It prints the commit's address and its short id. **Check it arrived:**
`python "API calls/remote_results.py" manifest` should show the new `snapshot_id` and `made`.

To look at a copy before it goes online, build it without uploading (`python main_backend.py publish-results`)
and open `results_site/index.html`.

Every publish is kept in the dataset's history. If one is wrong, publish again after fixing the cause; readers
who need the earlier copy can pin its commit id.

### Once: the organisation, the dataset and the login (done)

1. **The organisation** `Dramatizer` at huggingface.co holds the dataset, so access is managed in one place.
2. **The dataset** `Dramatizer/Results_Data` is **private**. The upload refuses a public one.
3. **The host's write token:** <https://huggingface.co/settings/tokens> > Create new token >
   **Fine-grained**, with write access to `Dramatizer/Results_Data` only. Log in with it in the project's
   environment, and answer `n` to the git credential question:

   ```bash
   E:/envs/dramatizer/Scripts/hf.exe auth login
   ```

   It is saved on this PC; never commit or share it.

### Who gets access

- **Researchers and admins:** yes. The copy holds every listener's scores.
- **A listener who still has piles to rate blind:** no, not until they have finished. Otherwise they could
  see other listeners' scores.
- **To give access:** the organisation's Settings > Members > Invite, with the **read** role. Only the host
  needs **write**.
- **To take it away:** remove them from the organisation. Their token then stops working for the dataset.
- **To check who has access:** the organisation's Settings > Members.
- **Tokens:** each person's token is their own, so nobody shares a secret.

---

## Troubleshooting

| What you see | What to do |
|---|---|
| `Hugging Face would not give … (HTTP 401/403/404)` | You are not logged in (`hf auth login`) and `DRAMATIZER_HF_TOKEN` is not set, the token was deleted, or you are not (yet) in the organisation. A private dataset says "not found" to anyone without access. |
| The dataset page says 404 in the browser | Log in at huggingface.co with the account that is in the organisation. |
| `Could not reach https://huggingface.co/…` | You are offline, or Hugging Face is down. |
| `… does not match the manifest` | The host is uploading right now, or an upload broke. Wait a minute and try again. |
| `… is answered by the live API only` | That question needs the host's PC; use `API_REQUEST.py` while the API runs. |
| `Not uploaded: … is PUBLIC` (host) | Make the dataset private: its Settings > Change visibility. Then publish again. |
| `No dataset … that this token can see` (host) | Log in with a write token for `Dramatizer/Results_Data` (`hf auth login`). |
| `publish-results` says `Not published, the bundle holds what it must not` | It lists each file and what it found. Fix the cause; the draft is in `results_site.new` to look at. Never publish around the check. |

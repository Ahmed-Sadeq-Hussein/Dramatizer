# Access remote server

The Dramatizer's results (no audio) are hosted online, so the team can read them while the host's PC is
off. This document explains how to read them, in a browser or from code, and how the host sends updated data.

- **Address:** `https://dramatizer-results.pages.dev`
- **Who:** team members only (researchers and admins). The site asks for a login before it shows anything.
- **What it holds:**
  - the evidence snapshot (coverage, winners only with enough evidence, artifact checks, caveats);
  - every listener's scores;
  - the evidence records that answers and search are made from;
  - the six questions' charts;
  - the baseline and Milestone 2 summaries.
- **What it does not hold:**
  - audio;
  - the sealed Report evaluation, until its key is opened;
  - accounts, emails, passwords, API keys, and test accounts' scores.
- **How new:** as new as the host's last publish. Every copy is stamped with the data it holds (see
  [Which data am I reading?](#which-data-am-i-reading)).

| You want to | Use |
|---|---|
| Look at the results | [The browser](#1-in-a-browser) |
| Ask questions from Python, a script or a chatbot, as with the API | [Python](#2-from-python-the-same-as-an-api-request) |
| Read the files from any other language | [Plain HTTP](#3-from-any-language-plain-http) |
| Put new results online (the host) | [Sending updated data](#4-sending-updated-data-the-host) |
| Use a private Hugging Face dataset instead of the Cloudflare site | [Hugging Face](#5-the-alternative-a-private-hugging-face-dataset) |

---

## 1. In a browser

1. Open `https://dramatizer-results.pages.dev`.
2. Type your email address. It must be one the host added.
3. Cloudflare emails you a one-time code. Type it in.
4. You see the results page: the stamps, the snapshot's summary, the six charts with their answers, and
   links to every file.

The login lasts 24 hours on that browser. If your email is refused, ask the host to add it.

## 2. From Python, the same as an API request

`API calls/remote_results.py` asks the same questions as `API calls/API_REQUEST.py`. It uses the same
function names and gives the same answers, but it reads the hosted copy instead of a running API.
- **Same answers, checked:** on the same data, all nine routes give identical answers from the copy and
  from the live API, because both use the same code (`backend/evidence_views.py`).
- **Every file is checked:** each file is compared with the copy's manifest, so a half-uploaded copy is caught.

### Once

1. **Get the code:** clone the repository (`https://github.com/Ahmed-Sadeq-Hussein/Dramatizer`) or pull the latest version.
2. **Install the two packages it needs** in any Python 3.10+ environment:

   ```bash
   pip install numpy python-dotenv
   ```

3. **Ask the host for a service token,** a Client ID and a Client Secret. Keep them private, like a
   password. They let a script in without the email code.
4. **Set them as environment variables.** In PowerShell, to keep them for new terminals:

   ```powershell
   setx DRAMATIZER_RESULTS_URL "https://dramatizer-results.pages.dev"
   setx DRAMATIZER_ACCESS_ID "<the Client ID>"
   setx DRAMATIZER_ACCESS_SECRET "<the Client Secret>"
   ```

   Open a new terminal afterwards. On macOS or Linux, use `export NAME="value"` in your shell profile.
   Never put the token in a file that is committed.

### Asking questions

From Python, exactly as with `API_REQUEST.Results`:

```python
import sys
sys.path.insert(0, "API calls")
from remote_results import connect

results = connect()                                      # reads the address and token from the environment
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

### What it answers, and what needs the live API

| Answered from the hosted copy | Needs the live API (the host's PC) |
|---|---|
| `evidence_status`, `query_scores`, `evidence_question`, `search_evidence`, `answer_evidence`, `baseline_results`, `milestone2_results`, `about_dramatizer` | `overview`, `list_searches`, `search_details`, `top_genomes`, `named_genomes`, `listening_piles`, `pile`, `renders`, `voices`; rating; rendering; questions over a date range (`from_date`, `to_date`) |

A question that needs the live API returns an error that says so; it never returns a wrong answer.
Answers show every listener's scores, as an admin sees them.

### Reading a local copy

The host can give you the folder or the zip instead. Unzip it and point the client at the folder:

```python
results = connect("C:/path/to/results_site")
```

## 3. From any language: plain HTTP

Every file is a plain GET with the service token in two headers:

```bash
curl -H "CF-Access-Client-Id: $DRAMATIZER_ACCESS_ID" -H "CF-Access-Client-Secret: $DRAMATIZER_ACCESS_SECRET" https://dramatizer-results.pages.dev/manifest.json
```

| File | What it is |
|---|---|
| `manifest.json` | The stamps, and every file with its size and sha256 |
| `snapshot.json` | The full evidence snapshot |
| `scores.csv` | Every listener's current score of every clip, with the artifact answer, dates, number of changes, verdict and markup. **All** scores (every batch); the request tools above read only the evidence records (mos-diverse-v1), as the live API does |
| `rating_history.csv` | Every score as it was given or changed, oldest first, with the score before it |
| `runs.json` | Every run that is not sealed, rated or not: voice, batch, pile, method, seeds, line and reference text, what Whisper heard of the plain line, and every genome in full (genes, operations, markup, speed, warnings, verdict, length, and the sanity check: word error rate, speaker similarity, what Whisper heard, pace). No audio |
| `piles.json` | Which listener was given which pile (sealed piles left out) |
| `evidence_records.jsonl` | The evidence records, one JSON per line |
| `genomes.json` | What each genome in the evidence records does, and its markup |
| `charts/charts.json`, `charts/q1_…jpg` … `q6_…jpg` | The six questions: claims, notes and charts |
| `results.json` | The baseline and Milestone 2 summaries |

If you get a web page (HTML) instead of data, the token was not accepted. See
[Troubleshooting](#troubleshooting).

## Which data am I reading?

Every copy is stamped. The stamps are in `manifest.json`, at the top of the web page, and in every answer's
`provenance`:

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
the website as before; their scores reach the hosted copy at the host's next publish.

### Each time (after a rating session, or whenever the results change)

1. **Build the copy.** This changes nothing online:

   ```bash
   python main_backend.py publish-results --zip
   ```

   It writes `D:/Dramatizer 2 renders/results_site/` and `results_site.zip`, and prints the stamps.
   It **refuses to build** if the copy would hold a sealed run, the condition key, a test account, an
   email, a credential or a local path. It never copies audio.
2. **Look at it** (optional): open `results_site/index.html`, or ask it questions locally:

   ```bash
   python "API calls/remote_results.py" evidence_status
   ```

   with `DRAMATIZER_RESULTS_URL` set to the folder.
3. **Upload it**, in one of two ways:
   - **Dashboard:** Cloudflare > Workers & Pages > `dramatizer-results` > Create deployment > upload
     `results_site.zip`.
   - **One command** (needs Node.js; the first time it opens a Cloudflare login in the browser):

     ```bash
     python main_backend.py publish-results --deploy
     ```

4. **Check it arrived:** `python "API calls/remote_results.py" manifest` should show the new `snapshot_id`
   and `made`.

Cloudflare keeps every earlier deployment. If a publish is wrong, open the project's Deployments list and
choose **Rollback** on the previous one.

### Once: setting up the site and the login

Follow "The results online, with this PC off" in `SHARING.md`. In short:
1. Create the Pages project with the placeholder page.
2. Put Cloudflare Access in front of it, with the team's emails.
3. Check in a private window that it asks for a login.
4. Upload the real copy.

Then make the service token that scripts use:

1. Zero Trust > Access > Service Auth > Service Tokens > **Create Service Token**. Name it, for example,
   `Dramatizer`, and choose a duration of 1 year.
2. Copy the **Client ID** and **Client Secret** at once. The secret is shown only this one time.
3. Zero Trust > Access > Applications > `dramatizer-results` > Policies > **Add a policy**:
   - Action: **Service Auth**.
   - Include: **Service Token**, `Dramatizer`.
4. Give the ID and secret to team members privately, not in a shared document or a commit.
5. To cut off access, revoke the token (or remove an email from the policy) in the same place.

### Who gets access

- **Researchers and admins:** yes. The copy holds every listener's scores.
- **A listener who still has piles to rate blind:** no, not until they have finished. Otherwise they could
  see other listeners' scores.
- **To check who has access:** Zero Trust > Access > Applications > `dramatizer-results` > Policies.

---

## 5. The alternative: a private Hugging Face dataset

The same copy can live in a **private dataset** on Hugging Face instead of, or as well as, the Cloudflare
site. It is free, and every publish is a commit, so earlier copies can always be read again. There is no
web page with a login: each team member reads with their own Hugging Face account and token.

### The host, once

1. **Make an account** at `https://huggingface.co/join`, if you have none.
2. **Make an organisation** for the team, so that access is managed in one place: `https://huggingface.co/organizations/new`,
   here `Dramatizer`. Free organisations can hold private datasets.
3. **Invite the team:** the organisation's Settings > Members > Invite. Give team members the **read**
   role; only the host needs **write**. Invite researchers and admins only, as with the Cloudflare site.
4. **Create the dataset:** `https://huggingface.co/new-dataset`:
   - Owner: the organisation.
   - Name: `Results_Data` (so the dataset is `Dramatizer/Results_Data`).
   - Visibility: **Private**.

   The upload refuses a public dataset, but make it private from the start.
5. **Make a write token for yourself:** Settings > Access Tokens > Create new token >
   **Fine-grained**, with write access to `Dramatizer/Results_Data` only. Then log in with it
   in the project's environment:

   ```bash
   E:/envs/dramatizer/Scripts/hf.exe auth login
   ```

   Paste the token when it asks. It is saved on this PC; never commit or share it.

### The host, each time

```bash
python main_backend.py publish-results --hf Dramatizer/Results_Data
```

This does four things:
- builds and checks the copy, as before;
- checks that the dataset is private;
- uploads it as one commit, named after its `snapshot_id` and stamps;
- removes the previous copy's files that are no longer in it.

It prints the commit's address and its short id. Readers can pin that id to read exactly this copy.

### Team members, once

1. **Accept the organisation's invitation,** from the email or from huggingface.co.
2. **Make your own read token:** Settings > Access Tokens > Create new token > **Read**.
3. **Set it, and the dataset's address,** then open a new terminal:

   ```powershell
   setx DRAMATIZER_RESULTS_URL "hf://datasets/Dramatizer/Results_Data"
   setx DRAMATIZER_HF_TOKEN "<your read token>"
   ```

Everything in [section 2](#2-from-python-the-same-as-an-api-request) then works unchanged:
`connect()`, the same questions, and the same answers. Reading needs no extra package.

**To read one exact copy,** for example the one a reported number came from, add its commit id:
`hf://datasets/Dramatizer/Results_Data@<commit id>`. The dataset's "Files and versions" tab
lists every publish with its stamps.

**From any language:** the files are at
`https://huggingface.co/datasets/Dramatizer/Results_Data/resolve/main/<file>`, with the header
`Authorization: Bearer <your read token>`.

### Who gets access

- **To give access:** invite the person to the organisation with the read role.
- **To take it away:** remove them from the organisation. Their token then stops working for the dataset.
- **Tokens:** each person's token is their own, so nobody shares a secret.

---

## Troubleshooting

| What you see | What to do |
|---|---|
| `Could not reach https://dramatizer-results.pages.dev` | The address is wrong, the site is not set up yet, or you are offline. |
| `The site asked for a login` or `answered with a web page instead of data` | Set `DRAMATIZER_ACCESS_ID` and `DRAMATIZER_ACCESS_SECRET`, and open a new terminal. |
| `The site refused the service token (401/403)` | The token was revoked or expired, or is not in the policy. Ask the host. |
| `… does not match the manifest` | The host is uploading right now, or an upload broke. Wait a minute and try again. |
| `… is answered by the live API only` | That question needs the host's PC; use `API_REQUEST.py` while the API runs. |
| `Hugging Face would not give … (HTTP 401/403/404)` | The dataset name is wrong, `DRAMATIZER_HF_TOKEN` is not set, or you are not (yet) in the organisation. A private dataset says "not found" to anyone without access. |
| `Not uploaded: … is PUBLIC` (host) | Make the dataset private: its Settings > Change visibility. Then publish again. |
| `No dataset … that this token can see` (host) | Create the dataset first, and log in with a write token (`hf auth login`). |
| The email code never arrives | Check your spam folder. Your email may not be in the policy: ask the host. |
| `publish-results` says `Not published, the bundle holds what it must not` | It lists each file and what it found. Fix the cause; the draft is in `results_site.new` to look at. Never publish around the check. |

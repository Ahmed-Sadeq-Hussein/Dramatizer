# API calls: pulling data from the Dramatizer and downloading it

The Dramatizer's back end runs as a web API on the host's PC (`backend/api/server.py`). Anyone with
an account or an API key can ask it for data and save what comes back:

- data as **JSON**: voices, genome searches, scores, rankings, named genomes, results;
- audio as **WAV**: a voice's recording, a search's plain line and genome clips, a spoken line.

This folder has a script that does the asking and saving for you (`dramatizer_api.py`), and the same
calls by hand with `curl`.

## 1. What you need from the host

| What | Where the host gets it |
|---|---|
| **The address** | `https://127.0.0.1:5005` on the host's own PC; from elsewhere, the playit.gg address and port |
| **Who you are** | An account (username and password: `python main_backend.py account add ...`) or an API key (`python main_backend.py api-key add NAME`) |
| **The certificate fingerprint** | `python main_backend.py api --fingerprint`: 64 hex digits like `AB:CD:...` |

The API must be running on the host: `python main_backend.py api`.

**Why the fingerprint:** the API uses its own certificate, which no certificate authority vouches
for. The script checks the certificate's fingerprint before it sends anything (your password
included). If it doesn't match, someone may be in between, or the host made a new certificate, and
the script stops.

Never put a password or key in a file in this folder, or in a command you share. The script asks
for the password, or reads it from an environment variable.

## 2. The script (recommended)

It needs only Python 3.8 or newer: no project install.

```bash
cd "API calls"
python dramatizer_api.py --url https://127.0.0.1:5005 --fingerprint AB:CD:... health
```

`health` needs no login and shows `{'ok': True, 'version': 1}` when the API is up.

Every other command needs `--user NAME` (it asks for the password) or `--key KEY`:

| Command | What it saves |
|---|---|
| `get genome/searches` | The list of genome searches, as `downloads/genome_searches.json` |
| `get "voices?dataset=VCTK&page=2"` | Any GET route below, as JSON (or WAV for audio routes) |
| `search RUN_ID` | One search: `downloads/searches/RUN_ID/search.json`, plus `baseline.wav` and every `genome_NN.wav` |
| `search RUN_ID --no-audio` | The same, JSON only |
| `searches` | Every search you may see: the list and each search as above (about 24 searches x 11 clips, roughly 200 MB) |
| `searches --no-audio` | Every search, JSON only (small) |

Options for every command:

| Option | Meaning |
|---|---|
| `--out FOLDER` | Save somewhere else (default: `downloads/` here, git-ignored) |
| `--name NAME` | With `get`: the file name to save as |

To avoid typing the same things, set them once in the shell:

```bash
export DRAMATIZER_URL=https://127.0.0.1:5005
export DRAMATIZER_FINGERPRINT=AB:CD:...
export DRAMATIZER_USER=test_user_1
python dramatizer_api.py searches --no-audio
```

In PowerShell: `$env:DRAMATIZER_URL = 'https://127.0.0.1:5005'` and the same for the others.
`DRAMATIZER_PASSWORD` or `DRAMATIZER_KEY` can be set the same way, so it doesn't ask; leave it out of
scripts you save or share.

A login ends when the script finishes (it logs out). An API key is sent as it is and stays valid until
the host removes it.

## 3. Questions about results, for a chatbot: `API_REQUEST.py`

`API_REQUEST.py` asks the API about results and answers with small JSON, sized for a language model's
prompt. It is meant as the tools of a chatbot (for example Gemma 4 with function calling, through
Ollama, vLLM or llama.cpp). It uses the same connection, fingerprint check and environment variables as
the script above.

| Function | Answers |
|---|---|
| `overview()` | How much there is: searches, rated searches, voices, piles, named genomes, renders, result sets |
| `list_searches(voice, batch, pile, rated, limit)` | Genome searches with their line, status and how many genomes passed the checks |
| `search_details(run_id)` | One search: every genome with its genes, verdict and MOS, best first |
| `top_genomes(voice, limit)` | The best-rated genomes across searches by MOS |
| `named_genomes()` | The genomes listeners picked and named |
| `listening_piles()` / `pile(batch, pile, voice)` | The piles, and one pile's clips with scores |
| `renders(query, mine, page)` | Lines people rendered in the website |
| `baseline_results(name)` | The baselines' WER, similarity, PESQ, STOI and SI-SDR |
| `milestone2_results(name)` | Milestone 2: share of genomes that passed, and why the others failed |
| `voices(dataset, query, page)` | The datasets, or one dataset's validation voices |

```python
from API_REQUEST import Results, TOOLS      # TOOLS: the functions as JSON schemas for the model
with Results.connect() as results:          # DRAMATIZER_URL, _FINGERPRINT, _USER or _KEY from the environment
    print(results.run_tool("top_genomes", {"limit": 3}))   # the model's tool call -> a JSON string
```

```bash
python API_REQUEST.py overview
python API_REQUEST.py milestone2_results
python API_REQUEST.py tools                 # print the schemas
```

A listener account sees no MOS (listeners rate blind), so `top_genomes` finds nothing for it. Give the
chatbot a researcher account or an API key.

### The evidence tools: any small question, and the six questions

| Function | Answers |
|---|---|
| `query_scores(listener, voice, dataset, gene_type, genes, genome, verdict, pile, clip, group_by, sort, order, limit, from_date, to_date)` | Listeners' scores, filtered, sorted or grouped (mean, count, range, mean above the plain line, share better than it). "Which genomes did Rama rate highest?" is `query_scores(listener="Rama")` |
| `evidence_question(number, from_date, to_date)` | One of the six genome questions, answered with the same numbers as its chart |
| `search_evidence(query, kind, k)` | The evidence records that best match a question in words |

They use three API routes: `GET /evidence/scores`, `GET /evidence/questions/<1-6>` (and `/chart`, the
JPEG) and `GET /evidence/search`. Only an **admin account or an API key** sees every listener's scores
and the six questions. Anyone else gets only their own scores.

## 4. The chatbot: `chatbot.py`

Gemma 4 picks a tool, the tool asks the API, and Gemma words the answer from what came back. The
model never makes up or calculates a number.

```bash
D:/envs/llm/python.exe chatbot.py                       # chat in the terminal
D:/envs/llm/python.exe chatbot.py --ask "Which genomes did Rama rate highest?" --steps
D:/envs/llm/python.exe chatbot.py --model gemma-4-E2B-it               # the smaller Gemma 4
python chatbot.py --server http://127.0.0.1:11434/v1 --model gemma4    # or an Ollama / llama.cpp server
```

`D:/envs/llm` is the environment with Gemma 4's newer transformers. The model is loaded from
`LLM_LIBRARY_DIR` (.env) in 4-bit. The API must be running, and the `DRAMATIZER_*` variables set as
above, with an admin account or an API key.

The model answers each turn with one JSON object: `{"tool": ..., "arguments": {...}}` or
`{"answer": "..."}`. That works with any chat model, local or on a server.

Every question and how the chatbot answers it through the API: [QUESTIONS.md](QUESTIONS.md).

**How well it answers:** `chatbot_eval.py` asks the 25 questions in `chatbot_questions.json`:
- 16 small ones, like Rama's highest;
- 3 about the results;
- the six questions.

The right answer is computed from the data at test time, so the test stays right as scores come in. Each
answer gets two marks: did the chatbot call the right tool, and does its answer hold the right numbers or
names. The report goes to `history/chatbot/` (git-ignored, because it holds listeners' scores).

```bash
D:/envs/llm/python.exe chatbot_eval.py                  # or --model gemma-4-E2B-it, --only s01,q6
```

**On the website (Ask the Dramatizer, on Home).** Anyone logged in can ask in the box on the Home page.
The API starts the chatbot itself on the first question (`backend/api/assistant.py`, which runs
`chatbot.py --serve` with `CHATBOT_PYTHON` from .env; loading Gemma takes about two minutes). Each
question is sent with the asker's own session, so every tool call is made as that person: a listener
gets answers about their own scores only. Until the model is ready, or if it fails, the API answers from
the records itself (`backend/rag/answer.py`, with the record IDs), and the page says which one answered.
Routes: `GET /about`, `GET /assistant` (is the model ready?) and `POST /assistant/ask {question, history}`.

## 5. By hand with curl

`curl` is on Windows 10 and later as `curl.exe`. These examples skip the certificate check (`-k`), so
**use them only on the host's own PC** (`127.0.0.1`). From elsewhere, use the script, which checks
the fingerprint.

Log in and keep the token:

```bash
curl -k -X POST https://127.0.0.1:5005/api/v1/login -H "Content-Type: application/json" \
     -d '{"username": "test_user_1", "password": "..."}'
# -> {"token": "...", "account": {...}}
```

Then send the token with every call, and save with `-o`:

```bash
curl -k -H "Authorization: Bearer TOKEN" https://127.0.0.1:5005/api/v1/genome/searches -o searches.json
curl -k -H "Authorization: Bearer TOKEN" \
     https://127.0.0.1:5005/api/v1/genome/searches/mos-diverse-v1_p1_CREMA-D_1014 -o search.json
curl -k -H "Authorization: Bearer TOKEN" \
     https://127.0.0.1:5005/api/v1/genome/searches/mos-diverse-v1_p1_CREMA-D_1014/audio/baseline.wav -o baseline.wav
curl -k -X POST -H "Authorization: Bearer TOKEN" -H "Content-Type: application/json" -d "{}" \
     https://127.0.0.1:5005/api/v1/logout
```

With an API key, send the key instead of a token: `-H "Authorization: Bearer KEY"` (no login, no logout).

## 6. What you can ask for

All under `/api/v1`. **Bold** routes are the ones for pulling data.

| Route | Gives |
|---|---|
| `GET /health` | Whether the API is up (no login) |
| `GET /me` | Who you are, your role and permissions, the queue |
| **`GET /datasets`** | The datasets and how many validation voices each has |
| **`GET /voices?dataset=&q=&page=`** | Validation clips, 50 a page; `q` matches a speaker or words |
| **`GET /voices/audio?dataset=&clip_id=`** | A clip's recording (WAV) |
| **`GET /genome/searches`** | Every genome search you may see, newest first |
| **`GET /genome/searches/<run>`** | One search: its genomes, verdicts, scores, artifacts heard and ranking |
| **`GET /genome/searches/<run>/audio/<clip>`** | `baseline.wav` (the plain line) or `genome_NN.wav` |
| **`GET /genome/listening?batch=&pile=`** | A listening pile: every clip, your score and the MOS. Without `batch` and `pile`: which piles exist |
| **`GET /genome/named`** | The named genomes |
| **`GET /genome/library`** | Every genome to pick from, and what the editor offers |
| **`GET /results`** | The baselines' and Milestone 2's summaries |
| `GET /jobs/<id>`, `GET /jobs/<id>/audio` | A queued job's state, and its audio when a speak or render job is done |

The routes that change things (`POST /speak`, `/genome/searches`, `/ratings`, `/pick`, ...) are listed
at the top of `backend/api/server.py`. They need a researcher or admin account, or an API key.

**What an account sees:**
- A listener sees every search, but rates only as themself and only their own pile.
- Their own scores come with `/genome/searches/<run>` and `/genome/listening`. Other listeners'
  scores are included only for accounts allowed to see them.
- An API key may do and see everything.

## 7. When it goes wrong

| Message | What it means |
|---|---|
| `certificate does not match the fingerprint` | Wrong fingerprint, a new certificate on the host, or someone in between. Ask the host for the current one |
| `401` | Wrong username or password, or the login ran out: log in again |
| `403` | Your account may not do this (e.g. a listener starting speech) |
| `404 No such search` | A wrong run id, or a search your account may not see |
| `401 ... locked` | 5 wrong passwords in a row: the account is locked for 15 minutes |
| `429` | Many refused logins on the server just now: wait a few minutes |
| Connection refused | The API isn't running on the host, or the address or port is wrong |

## 8. Limits

- **There is no bulk export route.** For example, there's no single route for every score as one CSV.
  `searches --no-audio` gives every search's scores as JSON. On the host, `python main_backend.py
  ratings --export FILE.csv` writes the CSV.
- **Downloads hold listeners' scores and voice recordings.** Keep them in `downloads/` (git-ignored)
  or somewhere private, and don't share them outside the project.

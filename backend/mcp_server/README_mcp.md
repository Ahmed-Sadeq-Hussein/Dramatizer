# MCP layer for Dramatizer (Milestone 3)

This page describes the local MCP layer: what it consists of, how to run it, how it is tested and what remains. It contains no passwords, keys or fingerprints. Those are provided by the backend host through a private channel.

## What the layer does

A local language model (Gemma 4, via Ollama) answers questions about Dramatizer. The model picks one tool at a time, the tool fetches data from the backend, and the model writes the answer from what came back. The model cannot write anything: all tools are reads.

The backend always decides what an account may see. The MCP layer does not bypass this. A listener account sees only its own scores, and the question across all listeners (question 5) returns 403 without the `rate_for_others` permission.

## Architecture

```
Gemma 4 (Ollama, local)
   ↓  JSON: {"tool": ..., "arguments": ...}
Client loop (backend/mcp_server/client_loop.py)
   ↓  MCP over stdio
MCP server (backend/mcp_server/dramatizer_mcp.py)
   ↓  read_calls.py: one GET per call, with the person's own token
Backend API (HTTPS, fingerprint, permissions)
```

## Files

| File | Contents |
|---|---|
| `backend/mcp_server/read_calls.py` | Plain Python functions that validate input and make one read call to the API. Tested. |
| `backend/mcp_server/test_read_calls.py` | Tests for the read calls, using a fake API client. |
| `backend/mcp_server/dramatizer_mcp.py` | The MCP server: 13 tools and one resource (`dramatizer://about`). |
| `backend/mcp_server/client_loop.py` | The client loop: the model picks tools, with stop rules and tracing. |
| `backend/mcp_server/eval_mcp.py` | Evaluation of the MCP loop on the same 25 questions as the old chatbot. |
| `history/mcp_traces/runs.jsonl` | Trace log of every client loop run (git-ignored). |
| `history/mcp_evals/` | Evaluation reports (git-ignored, because they contain scores). |

## Tools

| Tool | What it does |
|---|---|
| `datasets` | Datasets, validation voice counts and the voice split |
| `voices` | Validation voices in one dataset, 50 per page |
| `list_searches` | Genome searches you can see, newest first |
| `search_details` | One search: genomes, verdicts, ratings and ranking |
| `named_genomes` | Genomes that someone has picked and named |
| `renders` | Rendered lines, searchable, with an option to show only your own |
| `query_scores` | Scores filtered and grouped (listener, voice, dataset, gene type, number of genes, genome, clip, pile, verdict) |
| `evidence_question` | One of the six research questions, with the numbers from its chart |
| `search_evidence` | Evidence records matching some words |
| `answer_evidence` | An answer from the evidence with record IDs, or that nothing answers it |
| `results` | Baseline and Milestone 2 results, optionally by name |
| `overview` | Number of genome searches and how many are rated |
| `about` | What Dramatizer is, how to use the website and who made it |

Only one resource is exposed (`dramatizer://about`). All data comes through the read-only tools.

## Running it

Prerequisites: the backend must be running, Ollama must serve the model `gemma4-8k` (a variant of `gemma4:e4b` with `num_ctx 8192`), and the environment variables `DRAMATIZER_URL`, `DRAMATIZER_FINGERPRINT` and either `DRAMATIZER_KEY` or `DRAMATIZER_USER` plus `DRAMATIZER_PASSWORD` must be set in the same terminal.

Set the variables in Git Bash, from the repository root (replace the placeholders with your own values, and do not commit them):

```bash
export DRAMATIZER_URL="https://jensen-training.tun.ply.gg:65327"
export DRAMATIZER_FINGERPRINT="<fingerprint>"
export DRAMATIZER_USER="<username>"
export DRAMATIZER_PASSWORD="<password>"
```

To check that the MCP server starts: `python backend/mcp_server/dramatizer_mcp.py`. If it starts silently and waits, the login worked.

Ask a question through the client loop:

```
python backend/mcp_server/client_loop.py --ask "Which genomes did Rama rate highest?" --server http://127.0.0.1:11434/v1 --model gemma4-8k --steps
```

Run the evaluation: `python backend/mcp_server/eval_mcp.py --server http://127.0.0.1:11434/v1 --model gemma4-8k`. Add `--only s01,q5` to run single questions.

Inspector (for manual checks): `npx @modelcontextprotocol/inspector`, then add the server with absolute paths and the environment variables.

## Stop rules in the client loop

The loop stops when the model answers, after at most 6 tool calls, when the same call is repeated, when the time limit (600 seconds per question) is reached, and when the model answers `needs_person` (a decision or recommendation that no tool can give). Questions outside the topic return `OFF_TOPIC` without any tool call.

## Results

The evaluation uses the 25 questions in `chatbot_questions.json`. The truth comes from the backend, and the answers are checked against it.

| Run | Tools | Answers |
|---|---|---|
| Old chatbot (direct API calls) | 24/25 | 25/25 |
| MCP loop, latest full run | 25/25 | 24/25 |

In the latest full run, the genome question about sigh (`s12`) failed because of a wrong parameter (`genes` instead of `gene_type`). The instructions were changed afterwards, and the question passed when run on its own. A new full run is still needed to confirm this.

Evidence questions 1 to 6 passed both tool and answer checks in every run.

## Changes made during the work

- The tool descriptions were rewritten to say when each tool should be used, and `query_scores` gained the parameters `genes`, `genome`, `clip`, `pile` and `verdict`.
- Instructions were added for common mistakes: grouping by the wrong field, a gene type given as the number of genes, and a guessed name in `results`.
- The time limit was raised from 120 to 600 seconds, because a model answer on CPU takes about one to two minutes.
- The context window was raised from 4096 to 8192 tokens with a Modelfile, because tool results were otherwise cut off.
- The `needs_person` stop rule was added.

## Demo: invalid requests

The layer should reject bad input clearly, without calling the backend with wrong values. Two requests show this in the demo.

**1. Invalid tool argument (Inspector).** Open the Tools tab, choose `evidence_question`, enter `7` in `number` and press Execute Tool.

- Expected: a tool error saying the research question must be a number from 1 to 6.
- Why it matters: the value never reaches the backend. The check happens in `read_calls.py` before the API is called.

**2. Question outside the topic (client loop).**

```
python backend/mcp_server/client_loop.py --ask "What is the weather in Jönköping?" --server http://127.0.0.1:11434/v1 --model gemma4-8k --steps
```

- Expected: the answer is the off-topic message, the line shows `0 tool calls`, and the stop reason is `answer`.
- Why it matters: the model is told to call no tool for questions outside the project, so no data is fetched for them.

Take a screenshot of each for the presentation.

## Demo: a valid request and its saved record

The demo ends with one normal question, and then shows that the run is saved as a JSON record.

**3. Valid question through the client loop.**

```
python backend/mcp_server/client_loop.py --ask "Which genomes did Rama rate highest?" --server http://127.0.0.1:11434/v1 --model gemma4-8k --steps
```

- Expected: one tool call to `query_scores` with the listener filter and `sort: score`, a short answer, and `stopped: answer`.

**Showing the saved JSON.** Every run is appended as one JSON object per line to `history/mcp_traces/runs.jsonl`. Show the question, the tool calls and the stop reason:

```
python -c "import json; e = json.loads(open('history/mcp_traces/runs.jsonl', encoding='utf-8').read().splitlines()[-1]); print(json.dumps({k: e[k] for k in ('time', 'question', 'stop', 'seconds')}, indent=2, ensure_ascii=False)); print([(s['tool'], s['arguments']) for s in e['steps']])"
```

- Expected: the timestamp, the question, `stop: answer`, and a list with `('query_scores', {...})`.
- Why it matters: every run can be traced afterwards, with what was asked, which tools were called and with which arguments.

**Do not show the full record during the demo.** The `result` field of each step contains listeners' scores. Show the keys and the tool calls, not the scores.

## Demo: a question that asks for a recommendation

The layer must not pick a winner for the user. Choosing a genome is a decision for a person. The demo asks a question of that kind:

```
python backend/mcp_server/client_loop.py --ask "Which genome should I use for CREMA-D/1014?" --server http://127.0.0.1:11434/v1 --model gemma4-8k --steps
```

There are two acceptable outcomes, and both pass the test:

- **Stop with `needs_person`.** The model says that a person must decide, and makes no tool call.
- **Stop with `answer`.** The model first calls `answer_evidence`, then reports the figures (for example the scores for the plain line and each genome) and says that it does not make a definitive recommendation.

The test fails only if the model names one genome as the one to use. The model does not always take the same path. It sometimes calls a tool first, because the instructions ask it to look up evidence before it answers. Both outcomes are shown in the test runs.

## Known limitations

- **Data revision is missing from `answer_evidence` and `search_evidence`.** `evidence_question` returns `snapshot_id` and `ratings_up_to`, but the other two tools do not. This needs a backend change, which the host is responsible for.
- **Answers are slow** (about one to two minutes per question), because the model runs on CPU.
- **Tool results are cut** at 6 000 characters in the prompt.
- **There are no write tools.** Approval before writes and idempotency keys become relevant only if write tools are added.
- **Access for an admin account** depends on the host's decision. The reports contain individual listeners' scores and must not be published.

## Security

- Passwords and keys live only in environment variables in the terminal, never in code or in the repository.
- Reports and traces in `history/` are git-ignored, because they contain scores.

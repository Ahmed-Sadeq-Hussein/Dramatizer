# What has been done (as of 14 Nov 2026)

This file lists what the project has accomplished, who did it, and what is
still open. Work by Ash is listed first, then the team's work, including the
changes pulled and merged on 12 Nov 2026 and how they were fitted into the
rest. The latest work is not committed yet (see "Open").

## At a glance

| Milestone | State |
|---|---|
| 1. Reproducible baseline (data, split, plain CosyVoice 3 renders, checks) | Done |
| 2. Random genome search against the baseline | Run (random-v2); the listener verdict still waits on blind pairs |
| Listening study `mos-diverse-v1` (24 voices, 3 listeners) | Done: 264 scores |
| Evidence milestone (questions answered from the records) | Done: steps 01-06, cited answers, safe absence, access control |
| Objective and selection rule (the reviewer's critique) | Written: `backend/evaluation/OBJECTIVE.md` |
| Website, API, logins, chatbot | Running; shared through a Cloudflare tunnel |

Tests: `python main_backend.py test` runs 159 tests in 13 files, and all pass.
The MCP graph test is skipped where the `mcp` package is not installed.

### This week (9 to 14 Nov 2026)

| Date | Accomplishment | Milestone it serves | By |
|---|---|---|---|
| 11 Nov | Final data filtering, new website, MOS rating, logins (`888ba07`, `9da1ad7`) | Listening study (M2 verdict) | Ash |
| 11 Nov | Listening batch `mos-diverse-v1`: 24 voices, 264 scores | Listening study; evidence milestone (its data) | Ash, Julius, Rama (listeners) |
| 12 Nov | Logins with roles, audit log, UI fixes, data views (`9374cd4`, `228cba5`, `255a43a`, `2b8d1ad`) | Website / study infrastructure | Ash |
| 12 Nov | Preference-guided genome search and the MLflow MOS import (`e6cdaf7`) | M2 follow-up (guided search) | Baesar |
| 12 Nov | Genome report with a reproducibility manifest (`5d4a437`, `fea248b`) | Evidence milestone (the six questions in text) | Rama Muharam |
| 12 Nov | Merged pull request #5 (`feature/milestone-2`) into `main` (`c488e93`) | M2 | Ahmed Hussein |
| 13 Nov | Cloudflare hosting alternative (`87c92b4`) | Sharing the site with listeners | Ash |
| 13 Nov | Chatbot (Gemma 4) with 15 API tools; eval 24/25 right tool, 25/25 right answer | Evidence milestone (questions answered) | Ash |
| 13 Nov | The six genome questions as charts (`evidence-charts`) | Evidence milestone | Ash |
| 12-14 Nov | Evidence steps 01-06: questions, answer key, BM25, MiniLM/hybrid, comparison, refresh | **Evidence milestone: done** | Ash |
| 14 Nov | Cited answers, safe absence, access control (26/26 on the check) | **Evidence milestone: done** | Ash |
| 14 Nov | `OBJECTIVE.md`: answer to the reviewer's critique | **Critique: done** | Ash |
| 14 Nov | Team work fitted in: report command, test accounts out of every MOS, guided search without disabled genes, first preference model (chance level), Baesar's genome distance and diverse shortlist | M2 and evidence milestone | Ash (integrating Rama's and Baesar's work) |
| 13-14 Nov | Hissing and vocalized noise disabled; speed 0.85-1.15; natural slow speech (A/B rendered) | Audio quality for the next listening round | Ash |
| 14 Nov | "Ask the Dramatizer" and About on Home; data questions; declines off-topic | Evidence milestone (in the product) | Ash |
| 14 Nov | Private listening batch rendered and published (168 clips, 146 passed) | Next listening round | Ash |
| 14 Nov | `ACCOMPLISHMENTS.md`, `MILESTONE.md`, `Compute.txt` | Hand-in documentation | Ash |

### Since then (15 to 16 Nov 2026)

| Date | Accomplishment | Milestone it serves | By |
|---|---|---|---|
| 15 Nov | Chatbot re-scored with all tools: 24/25 right tool, 25/25 right answer | Evidence milestone (in the product) | Ash |
| 15-16 Nov | Words-limit calibration (1,048 plain pairs): no stricter rule passes, limit kept | M2 follow-up (the guard) | Ash |
| 16 Nov | MCP and login API calls planned as stubs (`backend/mcp_server/api_calls.py`) | MCP access with logins | Ash |
| 16 Nov | `Improvement angles.txt`: the 16 features in 7 groups, 16 ways to better speech | Next listening round | Ash |
| 16 Nov | Registration on the website with a required email; host approves (`account approve`) | Website / study infrastructure | Ash |
| 16 Nov | Accounts, sessions and ratings moved into one SQLite database (`backend/storage.py`), imported and checked; `Best practices.txt` | Website / study infrastructure | Ash |

### The evidence milestone, part by part

| # | Part | Done | What we did |
|---|---|---|---|
| 01 | Questions | 100 % | 10 questions about the listening study (8 answerable, 2 that nothing answers) in `backend/rag/questions.json` |
| 02 | Answer key | 100 % | Written rules on the records' fields decide which records answer each question (`labels.py`), not hand-picked results |
| 03 | Baseline retriever | 100 % | BM25 over 1,110 records (264 clip scores, 24 run rankings, 822 sanity checks), each with an ID and a revision |
| 04 | Variant retriever | 100 % | MiniLM dense search and a hybrid (rank fusion), run locally |
| 05 | Comparison | 100 % | hit@k, MRR, P@10, nDCG@10 against the key; BM25 stays (MRR 0.518), hybrid best on summaries; `COMPARISON.md` |
| 06 | Refresh | 100 % | Rebuild after new ratings, report what changed, flag labels that may be stale (`refresh.py`) |
| 07 | Cited answers and safe absence | 100 % | Three routes (summary counts, genome lookup, search); every answer lists record IDs and revisions; "no record answers this" below cut-off 6; 26 of 26 on the check (24 on the first run, both fixes reported) |
| 08 | Access control | 100 % | An answer uses only records the asker may see, by the API's own rule; tested as admin and as a listener |
| 09 | In the product | 100 % | `/evidence/answer`, the chatbot's `answer_evidence` tool, and "Ask the Dramatizer" on the website's Home page |
| 10 | The reviewer's critique | 100 % | `OBJECTIVE.md`: task, metric directions, selection rule, thresholds justified, checks with failing inputs, learned model spec, baseline with intervals |
| 11 | Data hygiene | 100 % | Test accounts left out of every MOS; record IDs name the listener |
| 12 | Team work fitted in | 100 % | Rama's report, Baesar's preference model, import and genome branch integrated and tested (see below) |
| - | Chatbot re-scored with the new tools | 100 % | 15 Nov: 24/25 right tool, 25/25 right answer (`Compute.txt` item 2) |

Required parts: 12 of 12 done, and the chatbot is re-scored.

---

## Done by Ash

### Data and baseline (Milestone 1), 28 Oct to 8 Nov
- **Manifest:** 106,171 clips from VCTK, CREMA-D and LibriTTS-R, each with speaker, script, emotion where known, and licence.
- **Filters:** narrowed the clips to one consistent set of voices (`DATA_CONTRACT.md`).
- **Split by voice:** each voice uses its longest neutral clip, split 176 validation / 44 test. The split is locked in `voice-split-v2.json` and the test voices are never used.
- **Baselines v1 to v4:** plain CosyVoice 3 renders of fixed lines.
  - The final one is 12 voices × 6 lines.
  - Measured with WER, speaker similarity, PESQ, STOI and SI-SDR, with 95 % intervals over voices.
- Measurement cache, local host version and iOS integration, back end / front end split, folder layout.

### Genome search (Milestone 2), 8 Nov
- The random genome search, and a sanity check against the plain render on three things: words, speaker and pace.
- Thresholds calibrated on 400 pairs of plain renders, with the reasons written next to them in `reward.py`.
- **Run `random-v2`:** 72 searches and 720 genomes.
  - 67 % of the genomes passed.
  - The words limit turned out too loose for a guard. This is stated in `MILESTONE_2.md` and was not quietly changed.
- The sanity check of every gene type on its own (`sanity-all-genes`).

### Listening study and website, 11 to 13 Nov
- **Rating:** MOS rating on the website, with artifact answers and listener piles.
- **Accounts:** login accounts with roles (listener, researcher, admin), a per-pile assignment, and an audit log.
- **Genome sets:** far-apart genomes for the voice-map voices (`explore`), voice similarity and the voice map.
- **The batch `mos-diverse-v1`:** 24 voices in 3 piles, rated by Ash, Julius and Rama (264 scores).
- **Website:**
  - a Home page;
  - modes (Rate, Generate, Research);
  - the Studio, for a line and a genome;
  - a genome library with "My genomes";
  - a Renders tab;
  - a plain-language status for each step.
- **Sharing:** the API and website are shared through a Cloudflare tunnel, with playit as an alternative (`SHARING.md`).
- **The six genome questions as charts:** `evidence-charts`, for any date range.

### Chatbot and API tools, 13 Nov
- **`API calls/API_REQUEST.py`:** a client for the API that checks the certificate's fingerprint, plus 15 tools a chat model can call.
- **`API calls/chatbot.py`:** a local Gemma 4 model (4-bit) that answers questions about the study by calling those tools.
- **`API calls/chatbot_eval.py`:** 25 questions whose right answers are computed from the data when the test runs.
  - Latest result (Gemma 4 E4B, 13 Nov): right tool on 24 of 25 questions, right numbers or names in 25 of 25 answers, 19 s per answer.
  - Re-run on 15 Nov with the new tools (`answer_evidence`, `about_dramatizer`, voice totals) and sharpened tool descriptions: right tool on 24 of 25, right answer on 25 of 25, 18 s per answer.

### Ask the Dramatizer and About, 14 Nov
- **Questions on the website:** the Home page has a question box, "Ask the Dramatizer".
  - The chatbot (Gemma 4) answers. It runs as a helper process started by the API, and every look-up it makes uses the asker's own login, so a listener only gets their own data.
  - While the model loads, or if it fails, the answer comes from the records, with their IDs, and the page says so.
  - Tested end to end: loading takes about 2 minutes, then answers take 12 to 27 s.
  - The model loads when the API starts. Questions about the data ("how many voices are used?") are answered by the model and by the records (220 voices: 176 validation, 44 locked test).
  - It answers only questions about the Dramatizer. Anything else (general knowledge, jokes, code) gets a fixed reply that points to example questions. Checked live with test_user_1.
- **About on Home:** what the Dramatizer is, how to use it (Rate, Generate, Listen, Ask, Research), and the contributors: Ahmed Hussein, Julius Norén, Rama Muharam and Tran Hieu Minh.
  - The text lives in one file (`backend/api/about.py`), shared by the page and the chatbot's `about_dramatizer` tool.
- **`Compute.txt`:** the list of GPU and long-running jobs, each with how to run it, why, and its status.

### Evidence milestone, 12 to 14 Nov (`backend/rag/`, write-up: `MILESTONE.md`)
- **Step 01:** 10 questions, 8 answerable and 2 not.
- **Step 02:** an answer key made by written rules on the records' fields.
- **Step 03:** a BM25 baseline over 1,110 records (clips, runs and sanity checks), each with an ID and a revision.
- **Step 04:** a MiniLM dense search and a hybrid search, run locally.
- **Step 05:** a comparison by hit@k, MRR, P@10 and nDCG@10.
  - BM25 stays best overall (MRR 0.518).
  - Hybrid is best on the summary questions.
  - Write-up: `COMPARISON.md`.
- **Step 06:** refresh after new ratings, with stale labels flagged.
- **Answers:** each answer cites its record IDs, or says "no record answers this" when nothing does (safe absence, cut-off 6).
  - An answer uses only the records the asker may see.
  - The same answers are served by `/evidence/answer` and the chatbot's `answer_evidence` tool.
  - Checked on the 10 questions and 16 reworded versions of them: 26 of 26 now. The first run got 24, and both fixes are reported in `MILESTONE.md`.
- **Data fixes:**
  - test accounts' scores are left out of the evidence;
  - record IDs name the listener, because two listeners had rated the same pile.

### The reviewer's critique (`backend/evaluation/OBJECTIVE.md`)
- The task in one sentence, and which way each metric is better.
- **The selection rule:** the sanity check is a pass-or-fail constraint, then the objective is the MOS gain over the same listener's plain line. PESQ, STOI and SI-SDR are reported only.
- Every threshold is justified from the calibration measurements, including the known weakness of the words limit.
- Each data check and sanity check is described with an input that fails it.
- The learned model is written out exactly: inputs, target, loss and split.
- The baseline is recorded with intervals.

### Audio and genome fixes, 13 to 14 Nov
- **Hissing and vocalized noise are disabled** for new genomes.
  - Their old scores stay in the evidence.
  - The Studio marks them as disabled, and an old genome that uses them shows that it can't be rendered as it is.
- **Speed is limited to 0.85-1.15** in the genome factory, the editor and the guided search.
- **Natural slow speech:** before, CosyVoice stretched the whole spectrogram, which sounded like an edited recording.
  - Now the reference clip is time-scaled (WSOLA), the line is rendered at normal speed, and only the pauses between words are scaled (`backend/speech/pacing.py`).
  - Set by `SPEED_METHOD`, natural by default.
  - An A/B render on two voices showed the model follows the slower pace partly, depending on the voice. The pace check measures what was actually reached.

---

## Done by the team (not by Ash)

### Pulled and merged on 12 Nov 2026
Merged into `main` through Ahmed Hussein's merge of pull request #5 (`feature/milestone-2`).

| Who | What | Commit |
|---|---|---|
| **Baesar** | Preference-guided genome search: `backend/genome/preference.py` (a conditional-logit ranker trained on listeners' choices, shared and voice-conditioned), `backend/genome/descriptors.py` (a genome as a fixed vector), guided methods in `search.py`, and website options for them | `e6cdaf7` |
| **Baesar** | `backend/evaluation/import_mos.py`: logs a listening batch to MLflow as a paired comparison (genome against the same listener's plain line) | `e6cdaf7` |
| **Rama Muharam** | `backend/genome/report.py`: the six genome questions answered in text from the ratings, with a CSV and a manifest (hashes of the ratings, code commit, settings) for reproducing them | `5d4a437` |
| **Rama Muharam** | "Changed the listener": the report's minimum listeners per genome from 2 to 1. Both commits were later reverted on `feature/evaluation`; on `main` the report and the change are kept | `fea248b`, reverts `c9346b7`, `cc100e4` |
| **Ahmed Hussein** | Merged pull request #5 | `c488e93` |

**How they were fitted into the rest (14 Nov):**
- **Report:** `report.py` can now be run as `python main_backend.py genome-report`. Its output prints correctly on Windows. It marks genomes that use a gene type disabled since, which can no longer be rendered.
- **Test accounts:** their scores no longer count in any MOS (`ratings.is_test_rater`). That one rule now covers the report, the genome library, the preference training and the MLflow import. The import also reports how many runs it left out.
- **Guided search:** it never proposes a disabled gene or an old speed, even when it starts from an old genome that has one.
- **The preference model was trained for the first time** on the mos-diverse-v1 ratings: 19 choices for training, 5 held-out speakers.
  - On those 5 it picks the winner 1 time in 5, which is chance.
  - With one listener per genome, the ratings don't yet show a learnable preference. This is written up in `OBJECTIVE.md`.
- **Baesar's `genome` branch** (2 to 5 Nov, never merged: `genome.py`, `genome_distance.py`, `genome_search.py` in the old root layout). Its genome structure, mutation and descriptor vector were already carried into `main` by `e6cdaf7` (`descriptors.py`, `GenomeFactory.mutate`). The two parts still missing were moved onto the current schema:
  - `descriptors.genome_distance(a, b)`: the distance between two genomes.
  - `search.diverse_shortlist(run)`: the run's fine genomes that differ most from each other (his `select_diverse`), printed after a run.
  - The branch's old root files and its `.env` changes were not brought back: they would duplicate `backend/genome/`.
- **Rama's reverts on `feature/evaluation`** (`c9346b7`, `cc100e4`) remove the report. They were not applied to `main`, where the report is kept and in use.
- **New tests:** `backend/genome/test_guidance_and_report.py` (7 tests) covers training, saving and loading the model, the guided search's gene rule, the report's marks, the import leaving test accounts out, and the genome distance and diverse shortlist.

### Earlier team work, from the git history
- **Rama Muharam:**
  - the evaluation setup: manifest, speaker split, deterministic checks, MLflow tracking;
  - the limitations, listening protocol, scope draft and known-answer tests;
  - one speaker split for evaluation;
  - the MCP server's knowledge graph rebuilt on demand, with pinned requirements.
- **Baesar:** the genome structure (`genome.py`), `genome_search.py` and `genome_distance.py`, and an update to the search algorithm.
- **Julius Norén:** merged pull requests #2 (evaluation) and #3 (MCP graph). Also a listener in `mos-diverse-v1`.
- **Ahmed Hussein:** the repository and the `.env` setup.
- **Julius and Rama:** listeners in the study, 88 scores each.

---

## Open

- **Commit and push** the latest work (28 changed files and 16 new ones, this file included). Ash does this.
- **Pick the branch to hand in.** All the work is on `main`. `feature/milestone-2` is at `fea248b` and behind it.
- **Milestone 2's verdict** waits on the blind listening pairs in `output/milestone2/random-v2/listening/`.
- **One listener per genome:** most comparisons rest on one person's score, and no listener has answered the artifact question yet.
- **The words limit:** calibrated again on 15-16 Nov (1,048 plain pairs). The stricter rule tested passed only 93.5 % of plain renders, under the 95 % bar, so the limit stays as it is (`OBJECTIVE.md`). The next candidate, comparing with several plain renders, needs a decision.
- **Natural pacing** reaches the asked speed only partly on some voices. A combined method (natural first, then a small stretch for the rest) is possible if wanted.
- **Other machines** with their own `history/` run `python main_backend.py storage migrate` once; until then logging in says so. The named genomes (`named.json`), API keys (`keys.json`) and render notes are still JSON files.
- **The MCP graph test** needs an environment with the `mcp` package.

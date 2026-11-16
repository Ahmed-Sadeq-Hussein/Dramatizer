# The questions, and how the chatbot answers each one through the API

The chatbot (Gemma 4 E4B, `chatbot.py`) knows no numbers itself. For every question it picks a tool from
`API_REQUEST.py`, the tool calls the Dramatizer API, and the model answers only from what came back.

## The path of one question

1. **Asking.** Someone types a question in **Ask the Dramatizer** on the website's Home page. The page sends `POST /api/v1/assistant/ask`.
2. **Passing it on.** The API server hands the question, with the asker's own session token, to the chatbot. The chatbot runs as a helper process (`chatbot.py --serve`), started by `backend/api/assistant.py`.
3. **Choosing a tool.** Gemma replies with a tool call, e.g. `{"tool": "query_scores", "arguments": {"listener": "Rama", "sort": "score"}}`.
4. **Calling the API.** `API_REQUEST.py` turns the call into a `GET` request to the API, made with **the asker's token**.
   - A listener gets only their own scores.
   - An admin gets everyone's.
   - A refusal (403) is passed back to the model, which says so.
5. **Answering.** Gemma reads the result and replies with the answer. It may call another tool first if the first result is not enough (at most a few calls).
6. **Showing it.** The page shows the answer and which tools were used.

**When the model can't answer.** While the model loads (about 2 minutes after the API starts), or if it fails, the server answers from the records itself (`backend/rag/answer.py`), with the record IDs. A question about something other than the Dramatizer gets a fixed reply.

## The tools and the API routes they call

| Tool | API route(s) | Answers |
|---|---|---|
| `query_scores` | `GET /evidence/scores` | Who rated what, highest or lowest, averages, by listener, voice, dataset, gene type or number of genes |
| `evidence_question` | `GET /evidence/questions/<1-6>` | The six research questions, with the same numbers as their charts |
| `answer_evidence` | `GET /evidence/answer` | Any other question about the evidence, with record IDs, or "nothing answers this" |
| `search_evidence` | `GET /evidence/search` | The records that best match some words |
| `overview` | `GET /me`, `/datasets`, `/genome/searches`, `/genome/named`, `/renders`, `/results` | How much there is: voices in the project, searches, piles, renders, results |
| `voices` | `GET /datasets`, `/voices` | How many voices and datasets, and which voices |
| `list_searches`, `search_details`, `top_genomes` | `GET /genome/searches`, `/genome/searches/<id>`, `/genome/library` | Genome searches and their genomes |
| `named_genomes`, `listening_piles`, `pile` | `GET /genome/named`, `/genome/listening` | Named genomes and listening piles |
| `renders` | `GET /renders` | Lines people rendered on the website |
| `baseline_results`, `milestone2_results` | `GET /results` | Baseline metrics and Milestone 2 pass rates |
| `about_dramatizer` | `GET /about` | What the Dramatizer is, how to use it, the contributors |

## 1. The six research questions

| # | Question | How it is answered |
|---|---|---|
| 1 | Which genomes work best for each voice? | `evidence_question` number 1 → `GET /evidence/questions/1` |
| 2 | Which genomes produce artifacts? | `evidence_question` 2 |
| 3 | Which genome configurations give bad MOS? | `evidence_question` 3 |
| 4 | Which genome configurations consistently give good genomes? | `evidence_question` 4 |
| 5 | How often does a genome beat the plain line, for each number of active features? | `evidence_question` 5. In the 13 Nov eval Gemma used `query_scores` grouped by number of genes instead: right answer, other tool |
| 6 | Which genomes can we exclude? | `evidence_question` 6 |

These need the right to see everyone's scores (researcher or admin). A listener gets a refusal, and the chatbot says so.

## 2. The evidence milestone's questions (`backend/rag/questions.json`)

| ID | Question | How it is answered |
|---|---|---|
| q01 | Which genomes work best for each voice? | `evidence_question` 1, or `answer_evidence` (summary "best") |
| q02 | Which genomes produce artifacts? | `evidence_question` 2, or `answer_evidence` (summary "artifacts") |
| q03 | Which genome configurations give bad MOS? | `evidence_question` 3, or `answer_evidence` (summary "bad") |
| q04 | Which genomes were rated better than the plain line of the same voice? | `answer_evidence` (summary "better"), citing the clip and run records |
| q05 | How often does a genome beat its plain line, for each number of active features? | `evidence_question` 5, or `answer_evidence` (summary "by features") |
| q06 | Which genomes can we exclude? | `evidence_question` 6, or `answer_evidence` (summary "exclude") |
| q07 | What did genome 5 of voice CREMA-D/1014 do, and how was it rated? | `query_scores` with voice and genome, or `answer_evidence` (the genome lookup) |
| q08 | Which genomes of voice VCTK/p250 scored better than its plain line? | `answer_evidence` (summary "better", limited to that voice) |
| q09 | Which genome suits an angry character? | Nothing in the records answers it: `answer_evidence` says "No record in the evidence answers this", and the chatbot repeats that instead of guessing |
| q10 | How do the 44 test voices score? | Nothing answers it: the test voices are locked and were never rated |

The same answers are checked without the model: `python main_backend.py rag answer --check` (26 of 26, with 16 reworded questions).

## 3. The chatbot evaluation's questions (`chatbot_questions.json`)

Measured by `chatbot_eval.py`. The right answer is computed from the data when the eval runs. Last run: 13 Nov 2026, Gemma 4 E4B, right tool 24 of 25, right answer 25 of 25, about 19 s per answer. The arguments below are the ones Gemma chose.

| ID | Question | Tool and arguments |
|---|---|---|
| s01 | Which genomes did Rama rate highest? | `query_scores` listener Rama, sort score |
| s02 | What is the lowest score Ash gave a genome, and to which voice? | `query_scores` listener Ash, sort score, order asc |
| s03 | Which listener gives the highest scores on average? | `query_scores` group_by listener, sort mean |
| s04 | What is Julius's average score for genomes? | `query_scores` listener Julius, group_by listener |
| s05 | Which gene type beats the plain line most often? | `query_scores` group_by gene_type, sort beat_plain |
| s06 | Which gene type has the worst average score? | `query_scores` group_by gene_type, sort mean, order asc |
| s07 | How many genome scores has Rama given? | `query_scores` listener Rama, group_by listener (count) |
| s08 | Which dataset's genomes got the best average score? | `query_scores` group_by dataset, sort mean |
| s09 | What is the average score of genomes with two genes? | `query_scores` genes 2, group_by genes |
| s10 | Which voice's genomes got the best average score? | `query_scores` group_by voice, sort mean |
| s11 | What did Ash score the plain line of CREMA-D/1014? | `query_scores` listener Ash, voice CREMA-D/1014, clip plain |
| s12 | Which genome with a sigh got the highest score? | `query_scores` gene_type sigh, sort score |
| s13 | Who has rated genomes so far? | `query_scores` group_by listener |
| s14 | What gene types does genome 5 of voice CREMA-D/1014 have? | `query_scores` voice CREMA-D/1014, genome 5 |
| s15 | How often do genomes from the VCTK dataset beat the plain line? | `query_scores` dataset VCTK, group_by dataset (beat_plain) |
| s16 | Which listener is the strictest, giving the lowest average score? | `query_scores` group_by listener, sort mean, order asc |
| r01 | What is the mean word error rate of baseline v4? | `baseline_results` name v4 → `GET /results` |
| r02 | In Milestone 2, what share of the random-search genomes passed the checks? | `milestone2_results` name random-v2 → `GET /results` |
| r03 | How many genome searches are there? | `overview` |
| q1-q6 | The six research questions | `evidence_question` 1-6 (see section 1) |

Re-run on 15 Nov with all the tools and the off-topic rule. The first try scored 22/25 right tool and 23/25 right answer: it counted searches with `list_searches`, counted plain lines as genome scores and used `top_genomes` for an average. The tool descriptions were sharpened; the second try scored **24/25 right tool, 25/25 right answer, 18 s a question**. The miss is still q5 (right answer through `query_scores`).

## 4. Questions about the Dramatizer and its data (checked live on 14 Nov, as a test listener)

| Question | Tool | Answer (short) |
|---|---|---|
| How many voices are used? | `overview` (or `voices`) | 220 voices: 176 validation, 44 locked test; VCTK 87, CREMA-D 27, LibriTTS-R 62 |
| How many speakers are there per dataset? | `voices` | VCTK 87, CREMA-D 27, LibriTTS-R 62 |
| How do I rate clips? | `about_dramatizer` | Open Rate, pick a pile, score each clip 0 to 5, and the plain line too |
| What is a genome? | `about_dramatizer` | A set of edits (sigh, pause, laugh, stress, speed) given to a line |
| Who made the Dramatizer? | `about_dramatizer` | Ahmed Hussein, Julius Norén, Rama Muharam and Tran Hieu Minh |

## 5. Questions it does not answer

General knowledge, jokes, code, advice, the weather and so on are not about the Dramatizer. The chatbot calls no tool and replies:

> I only answer questions about the Dramatizer: its voices, genomes, listening scores, results, the website and the team.

The reply ends with two example questions. An answer made without any tool, at the start of a conversation, is replaced by that reply too. Every fact must come from the API.

Checked live: "What is the capital of France?", "Write Python code to sort a list", "Tell me a joke", "What's the weather like today?" were all declined.

If the model wrongly declines a Dramatizer question, the server answers it from the About text or the records instead.

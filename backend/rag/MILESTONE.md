# The evidence milestone: questions answered from the project's own records

**Goal.** Answer questions about the genome listening study (which genomes work,
which to drop, how one genome was rated) from the project's records, so that
every answer is traceable: a question leads to evidence the asker is allowed to
see, and the answer names the records it rests on. When nothing in the records
answers a question, the system says so instead of guessing.

Everything here runs locally: no model is called over the network, and the
corpus holds listeners' scores, so it is written to the git-ignored `history/rag/`.

## The steps

| Step | What | Where | Command |
|---|---|---|---|
| 01 | 10 questions, 8 answerable and 2 not | `questions.json` | |
| 02 | The answer key: which records answer each question, by written rules on the records' fields | `labels.py`, `labels.json` | `rag label` |
| 03 | The baseline retriever: BM25 | `baseline.py` | `rag ask "..."`, `rag run` |
| 04 | The variant: MiniLM dense search, and hybrid (BM25 + MiniLM by rank fusion) | `dense.py` | `rag ask "..." --retriever hybrid` |
| 05 | The comparison: hit@k, MRR, P@10, nDCG@10 against the key | `compare.py`, `COMPARISON.md` | `rag compare` |
| 06 | Refresh: rebuild after new ratings, report what changed, flag labels that may be stale | `refresh.py` | `rag refresh [--check]` |
| + | Answers that cite their records, safe absence, and access control | `answer.py` | `rag answer "..." [--as USER]`, `rag answer --check` |

All commands are `python main_backend.py rag ...`. Tests: `backend/rag/test_rag.py`
(28 tests) and the API's `/evidence/answer` test in `backend/api/test_accounts.py`.

## The evidence

The corpus (`corpus.py`) has one record per thing that can be cited, each with a
source ID and a revision (when its score was given):

| Kind | Count | ID | Example |
|---|---|---|---|
| clip | 264 | `clip:<run>/<clip>@<listener>` | one listener's score of one clip |
| run | 24 | `run:<run>@<listener>` | one listener's ranking of a voice's genomes |
| sanity | 822 | `sanity:<search>/<voice>/<line>/<genome>` | one automatic sanity check |

Built 13 Nov 2026 from `who_rated_what.csv` and the search folders. Scores from
test accounts (`test_user_N`) are left out: they tried the website, they did not
listen. The ID names the listener because two listeners can rate the same pile.

## Step 05 in short (full write-up: `COMPARISON.md`)

| 8 answerable questions | MRR | nDCG@10 |
|---|---|---|
| **BM25 (baseline)** | **0.518** | **0.460** |
| dense (MiniLM) | 0.312 | 0.220 |
| hybrid | 0.500 | 0.350 |

- The variant does not beat the baseline overall, so BM25 stays the retriever.
- BM25 is perfect on the two lookups (MRR 1.0). Hybrid is best on the six summary questions (MRR 0.500 against 0.357).
- No retriever finds the answer to three summary questions (q02, q03, q06) in its top 10: they have 139 to 277 relevant records each, and their answer is a count, not a passage.

That last finding shaped the answer step.

## Answers, citations and safe absence (`answer.py`)

A question takes one of three routes:

1. **summary**: questions that need counting over many records (best per voice,
   artifacts, bad MOS, better than the plain line, by number of features, which to
   exclude, consistently good). They are counted from the records' fields, limited
   to one voice if the question names one.
2. **genome**: "genome N of voice V". That genome's clip records, its plain line and
   its run are looked up directly.
3. **search**: anything else goes to BM25. If the best record scores below the
   cut-off (6), the answer is **"No record in the evidence answers this"**, not
   the nearest record (safe absence).

**Every answer lists its citations:** each record ID with its revision, best
grade first. A long list shows the first 8 and gives the total.

**Access.** An answer is made only from records the asker may see, by the same
rule as the API. An admin sees everything. A listener or researcher sees only
their own clip scores and the sanity checks: no rankings and no one else's
scores. The website and chatbot get the same answers through
`GET /api/v1/evidence/answer?q=` and the chatbot's `answer_evidence` tool.

Example:

```
$ python main_backend.py rag answer "Which genomes of voice VCTK/p250 scored better than its plain line?"
[summary:better]
6 of 10 rated genomes for VCTK/p250 scored better than their plain line: VCTK/p250 genome 3 (Ash: 5, +2.5 vs plain); ...
Evidence (7 records):
  clip:mos-diverse-v1_p1_VCTK_p250/genome_01@ash  (revision 2026-11-11 11:47:07)
  ...
  run:mos-diverse-v1_p1_VCTK_p250@ash  (revision 2026-11-11 11:48:49)

$ python main_backend.py rag answer "Which genome suits an angry character?"
[search]
No record in the evidence answers this (the closest scores 0.0, below the cut-off 6).
```

Asked as Julius (a listener who never heard VCTK/p250), the first question gets
"No record in the evidence answers this … you see only your own scores".

### The check (`rag answer --check`, saved to `history/rag/answer_check.json`)

For each of the 10 questions, the check tests two things:
- it is declined exactly when the key expects nothing;
- every record it cites is in the key's relevant set.

It also runs 16 **paraphrases** (`paraphrases.json`): questions worded differently,
each with the route it should take, including 5 that nothing answers.

| | Result |
|---|---|
| 10 questions | 10 as expected: 8 answered with every citation in the key, q09 and q10 declined |
| 16 paraphrases | 16 as expected |
| Safe-absence margin | highest declined search score 1.6, lowest answered 17.0, cut-off 6 |

**Fixes after the first run, stated as they happened.** The first run got 24 of 26:

- q04 cited every run, while the key counts only runs where at least one genome
  beat the plain line. The citation was changed to match the key's written rule.
- One paraphrase ("For **every voice**, which genome did the listener score
  highest?") went to the "consistently good" route, because "every voice" was one
  of its words. That word was removed from the rule.

The paraphrases were written before the first run, but the second fix was made
after seeing one of them fail. So 16 of 16 is not a clean held-out score; 15 of
16 is.

## Limits

- **The summary answers use the same rules as the answer key.** Their check shows
  that a question reaches the right rule, not that the rule is right. The rules
  are written out in `labels.py`, one sentence each, for a person to judge.
- **Routing is by keywords.** "What is the best recipe for chocolate cake?" is
  sent to "best genome per voice", because of the word "best". It is kept in the
  paraphrases on purpose, to show this weakness.
- **The cut-off rests on few questions:** two unanswerable and two answerable
  search questions in the key, plus the paraphrases. It was not tuned; it sits
  between 1.6 and 17.
- **One listener per genome.** A score difference of 0.5 is one step of the
  scale, from one person. The answers say so where it matters.
- **Artifacts:** no listener had answered the artifact question yet when the
  corpus was built, so "which genomes produce artifacts" is answered from failed
  sanity checks, and the answer says that.

## Reproduce

```bash
python main_backend.py rag refresh
python main_backend.py rag compare --show 5
python main_backend.py rag answer --check
python main_backend.py test
```

`rag refresh` rebuilds the corpus and says if any label may be stale. `labels.json`
holds the corpus fingerprint it was made from, and `rag compare` and
`rag answer --check` refuse to run against a different corpus.

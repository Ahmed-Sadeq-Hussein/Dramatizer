# Steps 4 and 5: the variant, and how it compares with the baseline

## Step 4: the variant

The baseline (step 3) is BM25: a keyword search (`baseline.py`). The variant
(`dense.py`) adds a search by meaning, in two forms:

- **dense:** MiniLM (`sentence-transformers/all-MiniLM-L6-v2`, run locally from
  `LLM_LIBRARY_DIR`) turns each record and the question into a vector, and
  records are ranked by cosine similarity. It reads at most 256 word pieces of
  a record.
- **hybrid:** BM25 and dense each rank the records, and the two lists are
  combined by reciprocal rank fusion. A record gets 1 / (60 + rank) from each
  list it is in, from each list's top 100.

Every setting was fixed before the comparison: the model, the 256-token cut,
the fusion constant 60 (the usual value) and the pool of 100. None was tuned on
the answer key.

```bash
python main_backend.py rag ask "Which genomes produce artifacts?" --retriever hybrid
```

## Step 5: the comparison

```bash
python main_backend.py rag compare --show 5
```

**What was compared:**
- **Data:** all three searches ran on the same corpus of 1,110 records (264
  clips, 24 runs, 822 sanity checks), built 13 Nov 2026. Test accounts'
  scores are left out.
- **Answer key:** each search was scored against the step 2 key
  (`labels.json`), which written rules made from the records' fields, never from
  a search's output.
- **Questions:** only the 8 questions with an answer are scored. q09 and q10
  have none.

**The measures, each over a search's top 10 records:**

| Measure | Meaning |
|---|---|
| hit@k | 1 if a relevant record is in the top k, else 0 |
| MRR | 1 / the rank of the first relevant record (0 if none) |
| P@10 | the share of the top 10 that is relevant |
| nDCG@10 | the ranking weighted by grade (2 = answers the question, 1 = supports it) and position. 1 is ideal |

### Results (13 Nov 2026)

| 8 questions | hit@1 | hit@3 | hit@10 | MRR | P@10 | nDCG@10 |
|---|---|---|---|---|---|---|
| **bm25 (baseline)** | 0.500 | 0.500 | 0.625 | **0.518** | **0.425** | **0.460** |
| dense | 0.125 | 0.500 | 0.500 | 0.312 | 0.300 | 0.220 |
| hybrid | 0.500 | 0.500 | 0.500 | 0.500 | 0.388 | 0.350 |

| By kind of question | bm25 MRR | dense MRR | hybrid MRR | bm25 nDCG | hybrid nDCG |
|---|---|---|---|---|---|
| Lookup (q07, q08) | **1.000** | 0.250 | 0.500 | **0.796** | 0.289 |
| Summary (q01–q06) | 0.357 | 0.333 | **0.500** | 0.348 | **0.371** |

**MRR on each question, against BM25:**
- Hybrid: better on 1 (q04), the same on 6, worse on 1 (q07).
- Dense: better on 1, the same on 4, worse on 3.

### What the results show

- **The variant does not beat the baseline overall,** so BM25 stays the default.
  Hybrid's MRR is 0.500 against 0.518, and it ties on most questions.
  - With 8 questions, these differences show a direction and prove nothing.
- **Dense search misses exact names and numbers.** On "What did genome 5 of
  voice CREMA-D/1014 do?" (q07), it returns sanity checks of other CREMA-D
  voices.
  - Hybrid inherits the miss. A record ranked fairly high by both searches beats
    one ranked first by BM25 alone. So the right clip drops out of the top 10
    (MRR 1.0 → 0).
- **Meaning helps where the words differ.** On "Which genomes were rated better
  than the plain line?" (q04), the records say "better than the plain line" in
  the middle of a long text.
  - BM25 finds the first relevant record at rank 7.
  - Dense and hybrid find it at ranks 2 and 1.
  - Across the six summary questions, hybrid has the best MRR (0.500) and nDCG
    (0.371).
- **Three questions (q02, q03, q06) get no relevant record in any search's
  top 10.** They need counting across hundreds of records:
  - q02 has 251 relevant records;
  - q03, 139;
  - q06, 277.

  Their right answer is a count over the fields: the evidence charts and the
  chatbot's `evidence_question` tool. Finding a few matching texts doesn't
  answer them. That is a result for the design: summary questions go to a
  structured query, and lookups go to BM25.
- **Safe absence needs a cut-off, and only BM25's scores separate well.**

  | Question | Has an answer? | BM25's top score |
  |---|---|---|
  | q09 | no | 0.001 |
  | q10 | no | 1.6 |
  | q07, q08 | yes | 21.2 to 24.5 |

  Dense's top scores are all between 0.48 and 0.82, answerable or not.
  - Before test accounts were left out, q10 ("How do the 44 test voices
    score?") got a BM25 top score of 8.0: the word "test" matched the account
    named "Test user 3".
  - Now any cut-off between 1.6 and 21 separates these questions. With only
    two unanswerable questions, where to put it is still open.

### Changes made along the way

- **Record IDs now name the listener:** `clip:<run>/<clip>@<listener>` and
  `run:<run>@<listener>`. A test account had rated a pile Ash had already rated,
  so two listeners' clips shared one ID, and their run records had been merged
  into one.
- **Test accounts are not evidence.** Scores from logins named `test_user_N`
  are left out of the corpus (`corpus.TEST_ACCOUNT`), so also out of the charts
  and the API's evidence routes. They stay in `who_rated_what.csv`.
- **The answer key was regenerated** with `rag refresh --relabel` after both
  changes. Its rules are unchanged.

Full results, with each question's top records for inspection:
`history/rag/compare_results.json`. It is git-ignored, because record titles
carry listeners' names and scores.

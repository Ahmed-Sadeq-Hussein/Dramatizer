# Response to the evidence-milestone feedback (16 Nov 2026)

What the feedback asked for, what was changed, and what is still open. Run it yourself with:

```
python main_backend.py genome-report --csv "D:/Dramatizer 2 renders/report/report.csv"
python main_backend.py artifact-check
python main_backend.py test
```

## The two parts are now separate

| Part | File | Its job |
|---|---|---|
| Analysis | `backend/genome/evidence_snapshot.py` | Turns ratings into conclusions, once. Writes `history/evidence/evidence-<snapshot_id>.json` (and `evidence.json` beside the report CSV). |
| Assistant | chatbot (`API calls/chatbot.py`), `GET /evidence/snapshot`, `GET /evidence/questions/<n>` | Fetches the snapshot and explains it. It may name a winner only where the snapshot says `"status": "winner"`. |

The question flows through the system like this:

1. **Question:** "Which genome should I use for VCTK/p250?"
2. **Tool call:** the chatbot calls `evidence_status(voice="VCTK/p250")`, which goes to `GET /evidence/snapshot?voice=VCTK/p250`.
3. **Retrieval:** the route returns the snapshot: the `snapshot_id`, `ratings_up_to`, the rule, the coverage, the voice's status and reason, and its candidates with MOS and number of listeners.
4. **Sourced answer:** "Not enough evidence yet: the highest-scored safe genome, 3, has MOS 5.0 from 1 listener (Ash). Snapshot 7b8cbf8dbfe4529f, ratings up to 2026-11-13 09:14."

## Point by point

**1. Reconcile the 227 report rows with the 240 genomes.**
- The 227 came from a report made while rating was still going on. The rating log shows the 227th genome was rated on 11 Nov at 18:40, and the last 13 were rated later, up to 13 Nov 09:14. Today the report has 240 rows for 240 genomes.
- A report can no longer disagree with the runs silently. Every report and snapshot now opens with a coverage check: the genomes in the rated runs against the genomes with a score. Any genome without a score is listed by name, and the report is marked INCOMPLETE.
- The snapshot also records `ratings_up_to`, so a report says how far the ratings went.
- The 264 ratings break down as 240 genome scores plus 24 plain-line scores. A further 11 scores from a test account are left out and counted.

**2. State which ratings support an answer, the aggregation rule, and the run or version.**
- Every snapshot carries a `provenance` block:
  - `snapshot_id`: a hash of the contents, so the same ratings and code give the same ID;
  - `created`, `code_commit`, `ratings_up_to`, and `rating_log_sha256`;
  - the run IDs and batch;
  - the `rule` and `winner_rule` in words, and every setting.
- Every genome row lists its listeners, n, MOS, interval, and the result of each check.
- The six-question route returns the same provenance. The chatbot must quote n and the snapshot ID.

**3. Do not present a winner when the evidence is weak.**
- **The rule:** a voice has a winner only when its top safe genome and the runner-up were each scored by at least 2 listeners, and the top genome's 95% interval lies above both the plain line and the runner-up. Otherwise the status is `not_enough_evidence` or `no_clear_winner`, with the reason and the counts.
- **The result today:** all 24 voices are `not_enough_evidence`, because every genome was heard by exactly 1 listener.
- **Also changed:**
  - the question-1 chart's claim now says so;
  - the record answerer's "best" answer adds the same caution;
  - question 5 shows how many genomes and scores each share rests on.

**4. Keep the analysis apart from the assistant.**
- See the table above. The chatbot's instructions now say "You explain the analysis; you never decide".
- It has a new tool, `evidence_status`, for "which should I use / how sure are we / which ratings".

**5. Check for artifacts before using MOS to recommend.** A genome is recommendable only if it passes all of these, and has at least 2 listeners:

| Check | What it catches | Result on the 240 rated genomes |
|---|---|---|
| Sanity check (`search.py`) | words dropped, voice drift, broken pace | 240 pass: only passing genomes were rated |
| Audio check (`artifact_check.py`, new) | clipping, bursts (more than 6 dB above the plain line's loudest moment), gaps over 2.5 s, near-empty audio | 3 fail (bursts) |
| Listeners' artifact answers | what a person heard | none answered: the question was added after this round |
| Still renderable | genes disabled since (hissing, vocalized noise), or speed outside 0.85–1.15 | 115 blocked |

In total, 118 rated genomes are blocked and **0 are recommendable** today. The audio limits are working choices, written into every result. A loudness test against the clip's own median was tried first and dropped, because the plain lines alone reach 18 dB on it.

## What the data itself still lacks (not fixable in code)

- **One listener per genome.** No winner can be named until at least 2 listeners score the top candidates.
  - **Built 16 Nov:** `python main_backend.py second-listen` made the batch `second-listener-v1`. It has 3 piles and 95 clips: the plain line and the top 3 safe genomes of each of the 24 voices (VCTK/p343 has 2). The audio is copied, so nothing is rendered again.
  - Each pile goes to a listener who didn't hear those voices: pile 1 to Julius, pile 2 to Rama, pile 3 to Ash.
  - The runs are `blind`: the Rate page shows no one else's scores, even to an admin.
  - Their scores are added to the original genomes in the snapshot, which then has n = 2 for them. A listener who already rated the original is not counted twice.
  - The host gives each pile to its listener with `python main_backend.py account assign <user> second-listener-v1 <pile>`.
- **Listeners use the scale differently.** Mean genome scores are Ash 3.02, Julius 2.16 and Rama 1.48; Rama gave 0 to 39 of 80 genomes. Each voice had a different listener, so comparing voices also compares listeners. A second listener per genome would also allow scores to be normalised per listener.
- **No listener artifact answers.** The next rating round asks the artifact question for every clip.

## Files changed

- New: `backend/genome/evidence_snapshot.py`, `backend/genome/artifact_check.py`, `backend/genome/test_evidence_snapshot.py` (9 tests), and this file.
- Changed:
  - `backend/genome/report.py`, now built on the snapshot. Its CSV gains `snapshot_id`, `listeners`, the checks, `blocked_by` and `recommendable`, and its manifest gains coverage and the rules.
  - `backend/genome/evidence_charts.py`: the question-1 claim.
  - `backend/rag/answer.py`: the "best" caution.
  - `backend/api/server.py`: `GET /evidence/snapshot`, and provenance in `/evidence/questions/<n>`.
  - `API calls/API_REQUEST.py`: the `evidence_status` tool.
  - `API calls/chatbot.py`: the rules.
  - `main_backend.py`: the `artifact-check` command and the new tests.

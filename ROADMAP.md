# Dramatizer 2: what is left, in stages

Written 16 Nov 2026, from `PLAN.md`, `ACCOMPLISHMENTS.md` (Open), `Improvement angles.txt`, `Best practices.txt`,
`OBJECTIVE.md` and `LIMITATIONS.md`. Each stage ends with something we can show, and a check that says it is done.
Do the stages in order; inside a stage the items can be shared out. Owners are left blank for the group to fill in.

Size: **S** a few hours, **M** a day or two, **L** most of a week. **GPU** means it goes on `Compute.txt` first.

**Where we stand:**
- Milestone 1 (the baseline) is done.
- Milestone 2's search has run, but its verdict is still open.
- The evidence milestone is done.
- The website, API, logins, registration, chatbot and database all work.
- Genomes are still rated below the plain line: they score 2.22 on average against 3.25, and only 31 % beat their plain line.

---

## Stage 0: Make the work safe and shared (this week)

The groundwork the rest builds on. No GPU.

| # | Task | Size | Done when | Owner |
|---|---|---|---|---|
| 0.1 | Commit and push the latest work, and pick the branch to hand in | S | `main` holds everything; the others pull it | |
| 0.2 | Run `python main_backend.py storage migrate` on every computer with its own `history/` | S | `storage check` shows the accounts and ratings on each | |
| 0.3 | Delete the plain-text password files once the passwords are handed over (`passwords_to_hand_out.txt`, `test_users.txt`) | S | No password in a text file | |
| 0.4 | GitHub Actions runs `python main_backend.py test` on every push; ruff + pre-commit | M | A green check on each pull request | |
| 0.5 | A daily `storage backup` (Windows Task Scheduler), and one restore tried | S | A backup from today, and a restore that worked | |
| 0.6 | Milestones and tasks from this file as GitHub Issues on a Projects board | S | Every row here is a card with an owner | |

## Stage 1: Close Milestone 2

The question still open: does the random search beat the plain line?

| # | Task | Size | Done when | Owner |
|---|---|---|---|---|
| 1.1 | Rate the 10 blind pairs in `output/milestone2/random-v2/listening/` with `LISTENING_PROTOCOL.md`, before opening `key.json` | S | Each pair has a choice from at least 2 listeners | |
| 1.2 | Write the verdict into `MILESTONE_2.md`: the genome is preferred in more than half → it beats the baseline | S | The verdict, its count and its limits are written | |
| 1.3 | Measure agreement between listeners (Krippendorff's alpha) on `mos-diverse-v1` | S | The number, and what it means for the evidence | |

## Stage 2: Judges we can trust

The automatic checks decide which genomes are kept, so they must be fair before the search is improved.

| # | Task | Size | Done when | Owner |
|---|---|---|---|---|
| 2.1 | Pace check over speech time only, left out the inserted sounds (angle 13); calibrate on plain pairs | M, GPU | ≥ 95 % of plain pairs pass; fewer genomes with sounds and no speed gene fail | |
| 2.2 | Words rule against several plain renders (angle 14); the item 7 renders can be reused | M, GPU | ≥ 95 % of plain pairs pass, and genomes that break a word are stopped | |
| 2.3 | Separate judges for reporting from the ones the search uses (`PLAN.md` phase 4): another recogniser, another speaker encoder, a quality predictor such as UTMOS | M | Every reported number comes from a judge the search did not use | |
| 2.4 | Check the judges against people: how well each automatic score agrees with the listeners' scores | S | A correlation per judge, in `OBJECTIVE.md` | |

## Stage 3: Better genomes

Make the genes sound right before searching harder (`Improvement angles.txt`).

| # | Task | Size | Done when | Owner |
|---|---|---|---|---|
| 3.1 | Listen to every sound token on CosyVoice 3 (`PLAN.md` phase 1, angle 1): 10 clips each, two listeners | M, GPU | Each feature marked works, weak or absent; `VERIFIED_FEATURES` filled; absent ones off | |
| 3.2 | Put genes where people would: breaths and pauses at commas, laughter at phrase ends, stress on content words (angle 7) | M | Rule-placed genomes beat random placement in a blind A/B | |
| 3.3 | At most 3 genes, and fewer sounds on short lines (angle 8) | S | More genomes pass on short lines, scores not lower | |
| 3.4 | Pauses with a set length instead of "..." (angle 2) | S | Pause genes placed and timed as asked | |
| 3.5 | Re-rate speed with the natural method (angle 3); narrow or switch off if it still scores under 2.5 | S | A decision on speed, with scores behind it | |
| 3.6 | Even out the loudness of inserted sounds (angle 4) | S | No worse quality scores, fewer artifacts heard | |

## Stage 4: A better search

The method the paper is about (`PLAN.md` section 3).

| # | Task | Size | Done when | Owner |
|---|---|---|---|---|
| 4.1 | Novelty with a real archive of earlier genomes (PLAN idea 2) | S | Novelty compares against the archive; a test shows it | |
| 4.2 | Quality-diversity search, MAP-Elites over the style measures (PLAN idea 1, pyribs) | L, GPU | A map of best genomes per style region, for the baseline voices | |
| 4.3 | Guided search from the ratings (angle 11); the first preference model was at chance level, so more ratings first | M | A guided round beats a random round of the same size | |
| 4.4 | Transfer: seed a new voice from the nearest earlier voice's winners | M, GPU | Steps to a set score, nearest vs random vs far seed | |

## Stage 5: The main listening study

The human evidence the project and the paper stand on.

| # | Task | Size | Done when | Owner |
|---|---|---|---|---|
| 5.1 | Ethics: the university's rules, a consent text on the registration page, a short `PRIVACY.md` (emails are personal data) | S | Approved, and consent asked before rating | |
| 5.2 | Recruit 20 or more listeners through the website's registration; approve them | M | ≥ 20 approved listeners | |
| 5.3 | Blind A/B pairs and 0-5 ratings with the artifact question and attention checks, on the **locked test voices** (used once, at the end) | M, GPU | The piles rendered and published; test voices untouched until now | |
| 5.4 | Collect at least 100 scores per feature and per condition | L | The counts reached; agreement measured again | |

## Stage 6: Experiments and statistics

What reviewers ask for (`PLAN.md` section 4).

| # | Task | Size | Done when | Owner |
|---|---|---|---|---|
| 6.1 | Baselines at the same render budget: plain line, random genomes, an LLM that writes the markup, the search | L, GPU | One table, same voices and lines | |
| 6.2 | Ablations: no novelty term, no guards, no archive, no transfer, MAP-Elites vs the plain loop | L, GPU | One row per ablation, several seeds | |
| 6.3 | Statistics: bootstrap intervals, Wilcoxon on paired results, held-out speakers and lines | M | Every claim has an interval and a test | |
| 6.4 | Failures reported: voices that clone badly, features that never work | S | A section that lists them | |

## Stage 7: Finish the product

Not needed for the result, but for a complete hand-in and demo.

| # | Task | Size | Done when | Owner |
|---|---|---|---|---|
| 7.1 | Implement the MCP and login calls (`backend/mcp_server/api_calls.py`) | M | Its tests call the API for real; an MCP client logs in and asks | Done in pull request #6 (Rama): `read_calls.py` logs in and makes the 13 read calls; the `api_calls.py` stubs were removed |
| 7.2 | Move API keys, named genomes and render notes into the database too | M | No shared state left in JSON files | |
| 7.3 | Times in UTC, one naming standard for voices and runs (`Best practices.txt`, 2C) | M | Written in `DATA_CONTRACT.md` and used everywhere | |
| 7.4 | A Docker image for the API, and a clean-clone check of the README | M | A new computer runs it from the README alone | |
| 7.5 | Website polish from listener feedback; the confirmation email for registration if a mail service is allowed | M | Feedback items closed | |

## Stage 8: Write-up and hand-in

| # | Task | Size | Done when | Owner |
|---|---|---|---|---|
| 8.1 | Pick the venue and check its real deadline (Interspeech 2027 is the main target in `PLAN.md`; verify) | S | A date on the board | |
| 8.2 | The report or paper: method, experiments, results, related work, limitations, ethics | L | A full draft read by the whole group | |
| 8.3 | Reproducibility: every result tied to its commit, data manifest and settings (MLflow); a clean clone reproduces one table | M | One table reproduced by someone else | |
| 8.4 | Final presentation and demo (website, chatbot, a before/after of genomes) | M | Rehearsed once | |
| 8.5 | Hand in: the branch, `ACCOMPLISHMENTS.md` final, and what is not released (cloned audio of real people) | S | Submitted | |

---

## The order, and what can run side by side

```
Stage 0 ──► Stage 1 ──► Stage 2 ──► Stage 3 ──► Stage 4 ──► Stage 5 ──► Stage 6 ──► Stage 8
                          │            │                       ▲
                          └─ 2.3, 2.4 ─┘   Stage 7 at any time  │
                                           5.1 and 5.2 can start now (ethics and recruiting take time)
```

- **Start now, alongside Stage 0:** 5.1 ethics and 5.2 recruiting, because approval and finding listeners take weeks.
- **The GPU items** (2.1, 2.2, 3.1, 4.2, 4.4, 5.3, 6.1, 6.2) go on `Compute.txt` with their cost before they run.
- **The single biggest risk** is the human study (Stage 5): too few listeners, or listeners who disagree, and nothing later can be claimed.
- **Track it weekly** with the metrics in `Best practices.txt` part 4. The headline number is the share of genomes rated above their plain line, now 31 %, and the target is over 50 %.

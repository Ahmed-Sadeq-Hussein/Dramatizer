# The objective: what a genome is chosen for, and how it is judged

This answers the reviewer's critique of the first report point by point. Every
number below comes from a file in the repository or the render folder, named
where it is used, so it can be checked and reproduced.

## 1. The task in one sentence

Given a speaker's reference clip and a line of text, choose a **genome** (a set
of delivery edits: sound tags such as a sigh or a breath, pauses, word emphasis,
and speed) so that **listeners prefer the CosyVoice 3 render with the genome to
the plain render of the same line**, while the render stays intelligible and
keeps the same speaker.

The baseline is that plain render: CosyVoice 3 with its default settings, the
same reference clip, the same line and the same render seed, and no genome.

## 2. The metrics, and which way is better

| Metric | Measured with | Better is | Range | Role |
|---|---|---|---|---|
| WER (word error rate) | Whisper base | **lower** | 0 and up | guard (constraint) |
| Speaker similarity | Resemblyzer, cosine | **higher** | about 0 to 1 | guard (constraint) |
| Pace against the asked rate | speaking time, pauses removed | **higher**, at least 0.70 | about 0.5 to 1.5 | guard (constraint) |
| PESQ | SQUIM's estimate, no reference recording | **higher** | about 1 to 4.5 | reported only |
| STOI | SQUIM | **higher** | 0 to 1 | reported only |
| SI-SDR | SQUIM | **higher** | dB | reported only |
| MOS gain over the plain line | listeners, 0 to 5 scale | **higher** | -5 to +5 | **the objective** |

Our PESQ is SQUIM's estimate of PESQ made without a clean reference recording,
not PESQ itself; it is used to compare renders with each other, not as an
absolute quality figure.

## 3. The selection rule

A candidate genome is judged in two steps. Every candidate, from any search
method (random, preference-guided, or a hand-made genome), gets the same rule.

**Step 1, constraints (pass or fail): the sanity check.** The genome's render is
compared with the plain render of the same voice, line and seed
(`backend/metrics/reward.py`, `sanity_check`). It passes only if all three hold:

1. **Words:** at most 10 % of the words are heard differently (`sanity_wer_limit
   = 0.10`), and never fewer than one word allowed (`sanity_wer_words = 1`).
   Fillers ("uh", "ha", "hmm") are left out first, because a laughter or "mn"
   gene is often transcribed as one.
2. **Speaker:** speaker similarity to the plain render of at least 0.90
   (`sanity_similarity`), or 0.80 when the plain render has under 6 s of speech
   (`sanity_similarity_short`, `sanity_long_seconds = 6.0`).
3. **Pace:** at least 0.70 of the pace that was asked for (`sanity_pace_min`).
   Pace is the plain render's speaking time divided by the genome's, divided by
   the speed the genome asked for, so a slow genome is not failed for being
   slow, only for being slower than it asked.

**Step 2, the objective: MOS gain over the plain line.** Among the genomes that
pass, the better genome is the one with the higher **MOS gain**: a listener's
score for the genome minus the **same listener's** score for the plain line of
the same voice. Taking the difference within one listener removes each
listener's own use of the scale (one listener's 3 is another's 4).

**WER, similarity and pace are constraints, not part of a weighted score.**
PESQ, STOI and SI-SDR are reported and never optimised. A genome cannot win by
sounding cleaner to a measure while sounding worse to people, and no weighting
between automatic measures and listeners has to be invented.

**Whose scores count.** Scores from accounts made to try the website (logins
`test_user_N`, shown as "Test user N") are not evidence: they are kept in the
log but left out of every MOS, report, model and import
(`backend/genome/ratings.py`, `is_test_rater`).

## 4. Why these thresholds

**Words (0.10, at least one word).** Calibrated on 6 Nov 2026 with
`calibrate_sanity.py`: 10 validation voices (4 VCTK, 3 CREMA-D, 3 LibriTTS-R) ×
2 lines × 5 seeds, no genes, 400 pairs of plain renders compared with each
other. On the 8-word line, 89 % were heard identically and 99 % were within one
word; on the 22-word line, 96 % and 100 %. So the rule passes what two plain
renders differ by on their own.

Milestone 2 then showed this limit is **too loose for a guard**
(`MILESTONE_2.md`, run `random-v2`): the genomes that passed had a WER of 0.018,
above the baseline's 95 % interval (0 to 0.013), and 120 of the 483 that passed
had at least one word wrong. As decided before the run, the limit was not
changed after seeing this; a stricter limit needs a new calibration first. This
is a known weakness, stated rather than hidden.

**The new calibration (15-16 Nov 2026, `backend/metrics/calibrate_words.py`).** A
stricter rule was fixed before rendering: a clip may not get a word wrong against
the script that its own plain render gets right ("no_new"), kept only if at least
95 % of plain pairs pass it. 216 new plain renders (9 new validation voices ×
Milestone 2's 6 lines × 4 seeds) plus the 6 Nov pairs gave 1,048 plain pairs:

| Rule | Plain pairs passed (95 % interval) | Milestone 2 genomes passed | ... with a word wrong |
|---|---|---|---|
| Current (10 %, at least 1 word) | 95.6 % (94.2 to 96.7) | 483 of 720 | 119 |
| At most 1 word | 94.7 % (93.1 to 95.9) | 467 | 103 |
| No difference | 86.5 % (84.2 to 88.4) | 363 | 0 |
| no_new | 93.5 % (91.9 to 94.8) | 371 | 3 (all also wrong in the plain render) |

no_new would make the guard clean on Milestone 2, but it rejects 6.5 % of plain
renders for chance alone, which misses the 95 % bar. **So the limit is not
changed.** Most chance errors come from a few voices (VCTK/p247 and
CREMA-D/1060) and the longer lines. The next candidate compares a genome with
several plain renders of its voice instead of one. It needs its own calibration.

**Speaker (0.90, or 0.80 under 6 s).** From the same calibration, by the plain
render's speech time:

| Speech time | Same voice, plain vs plain |
|---|---|
| 6 s or more | 98 % at least 0.90 (lowest 0.851) |
| 4.5 to 6 s | 79 % at least 0.90, 100 % at least 0.80 |
| under 4.5 s | as low as 0.751; 0.80 lets 87 to 100 % through |

Different voices on the same line (5 Nov 2026, 132 pairs, 3.8 to 7.7 s): highest
0.889; 6 % at least 0.80, under 1 % at least 0.88. So 0.90 separates the same
voice from a different one on long lines; on short lines the encoder is
unsteady and the check is weaker, which is why searches that depend on it use
lines of about 20 words.

**Pace (0.70).** Plain renders alone vary a lot (same calibration): the lowest
was 0.63 and 5 % fell below 0.75. 0.70 lets 98 % of plain renders through (96 %
on the short line), while stopping a clip that buys clear words by dragging
them out.

**The reference-recording WER of 0.10** in the baseline is a different
threshold: it checks that Whisper can be trusted on the reference recordings
(clean read speech), not that a clone is right.

## 5. Each check in practical terms

**Data checks** (`backend/evaluation/checks.py`, run by `python main_backend.py
checks`). Each line says what it protects against and an input that fails it.

| Check | Protects against | Fails when, for example |
|---|---|---|
| The manifest has exactly the contract's columns | a changed export silently dropping a field | a `mic` column is renamed `microphone` |
| Every clip comes from a known dataset and carries its licence | using audio under the wrong terms | a CREMA-D clip is listed with `CC BY 4.0` instead of `ODbL` |
| No clip is listed twice | one recording counted as two | the same `VCTK/p250_003` appears in two rows |
| Every clip has a speaker and a non-empty script | clips that cannot be split by speaker or transcribed | a LibriTTS-R row with an empty `text` |
| Labels are allowed values | invented or misspelled emotions | a CREMA-D clip with emotion `angry` instead of `anger` |
| Validation and test share no speaker | a test speaker leaking into tuning | `VCTK/p250` is in both lists |
| Every clip's speaker is validation, test, or listed as unused | a speaker falling outside the split | a new dataset speaker added to the manifest only |
| The locked split is unchanged | quietly re-splitting after seeing results | `voice-split-v2.json` no longer matches its recorded hash |
| Each split voice's clip is its own, neutral and 3 to 25 s long | a bad reference clip | a voice whose chosen clip is 1.8 s, or "happy" |

**Sanity check** (per genome, section 3):

| Part | Fails when, for example |
|---|---|
| Words | a cough tag makes CosyVoice drop words, and Whisper hears "you really this time" |
| Speaker | vocalized noise pulls the voice towards someone else (similarity 0.78 on a long line) |
| Pace | three inserted sighs add two seconds, so the clip runs at 0.6 of the plain render's pace |

## 6. The learned model: inputs, target, loss

The preference-guided search (`backend/genome/preference.py`, used by
`search.draw_guided`) learns which way to move a genome. Written down exactly:

- **Unit of data:** one listener's scores on one run (one voice, one line): a
  shortlist of rendered clips and which of them the listener scored highest
  (all the top-scored ones share the win), or an explicit choice recorded with
  `preference choose`.
- **Inputs:** the genome as a fixed vector (`backend/genome/descriptors.py`:
  each gene type by where it falls in the line, pause kinds, and the speed),
  and for the voice-conditioned model also the speaker's Resemblyzer embedding
  (256 numbers) reduced by PCA to 8 numbers, combined with the starting genome.
- **Target:** which clip of the shortlist won.
- **Model and loss:** a conditional logit: each clip gets a linear score, and
  the loss is the negative log-probability of the winners under a softmax over
  the shortlist, plus L2 regularisation (penalty 0.01). A shared version
  ignores the voice; the conditioned version lets the weights depend on it.
- **Split:** by speaker. Validation speakers are held out of training, and only
  validation-split voices are used at all; the locked test speakers are never
  read.
- **Constraint:** the search only proposes genomes the genome factory allows
  (section 7), and every proposal is still judged by section 3.

**Its result so far** (trained 13 Nov 2026 on the mos-diverse-v1 ratings,
`python main_backend.py preference train`): 19 training choices (12 of them
ties) from 19 speakers, and 5 validation choices from 5 held-out speakers. On
those 5, every model picked the winner 1 time in 5, the same as choosing at
random (uniform log-loss 2.40, shared 2.35). With one listener per genome
and this few choices, the ratings do not yet show a learnable preference; more
blind ratings are needed before guidance can be expected to beat random search.

## 7. Genome rules since the first report

- **Disabled gene types:** hissing and vocalized noise are no longer used in new
  genomes (`DISABLED_FEATURES` in `backend/genome/genome_factory.py`). The
  evidence gave them a lower chance: of the rated genomes with them, 75 % (of
  40) and 66 % (of 41) scored below the plain line, against 58 % for all
  genomes, and vocalized noise failed the sanity check more often than any other
  gene type in random-v2 (42 % of 294 renders). Their past scores stay in the
  evidence.
- **Speed limited to 0.85-1.15** (`SPEED_BOUNDS`): 79 % of the 61 rated genomes
  with a speed gene scored below the plain line.
- **Speed is rendered naturally** (`SPEED_METHOD=natural`,
  `backend/speech/pacing.py`): the reference clip is time-scaled so the model
  speaks the words at the new pace, and only the pauses between words are
  scaled. Before, CosyVoice stretched the whole spectrogram, which sounded like
  an edited recording. A test on two voices (14 Nov 2026) showed the model
  follows the paced reference partly: asked for 0.85, one voice reached 0.87
  and the other 0.95. The pace check (section 3) measures what was reached.

## 8. The baseline, recorded

Plain CosyVoice 3, baseline v4 (`output/baseline/v4/summary.json`): 12
validation voices × 6 fixed lines = 72 renders, render seed 0. Means with 95 %
intervals over voices:

| Metric | Mean | 95 % interval |
|---|---|---|
| WER | 0.028 | 0.007 to 0.058 |
| Similarity to the reference | 0.869 | 0.828 to 0.904 |
| PESQ (SQUIM) | 3.59 | 3.21 to 3.93 |
| STOI | 0.991 | 0.985 to 0.996 |
| SI-SDR | 23.2 dB | 21.0 to 25.3 |

78 % of the 72 renders had a WER of zero.

Listeners (mos-diverse-v1, 24 voices, three listeners, one per voice): the
plain line averaged 3.25 out of 5. The 240 rated genomes averaged 2.22, a mean
gain of -1.03; 75 beat their plain line, 26 tied and 139 were below. Random
genomes are, on average, worse than no genome, which is the case for a
guided search, and also why that search has to be judged by the rule above
rather than assumed to help.

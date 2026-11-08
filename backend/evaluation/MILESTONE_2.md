# Milestone 2: does a random genome search beat the baseline?

The next update, written before it is run, so that the comparison is decided in
advance and not after seeing the numbers.

## The question

Milestone 1 measured the baseline: plain CosyVoice 3 renders of five fixed lines
in twelve validation voices (`python main_backend.py baseline`, run `v1`). The next
step is the simplest search: does a **random search of genomes**, kept honest by
the sanity check, find deliveries that are more expressive without losing the
words or the voice?

This is baseline 3 in PLAN.md section 7 ("random markup genomes, same guards"),
and the floor that every smarter search must beat later.

## What changes, and what stays the same

| Stays the same as baseline v1 | Changes |
|---|---|
| The 12 validation voices and 6 lines (`output/baseline/v2/plan.json`; line 5 is the shared comparison sentence) | Each line gets 10 random genomes (3 genes each) |
| Render seed 0, the same renderer (`search.render_plan`) | Each genome is judged by the sanity check against the plain render |
| The judges: Whisper base, Resemblyzer, SQUIM | |
| The data contract, the split, the test speakers locked | |

Already built: `backend/genome/` (the genome, the factory, the search, the run
files) and the Find a genome panel in the V4 UI. The sanity limits were
calibrated on 6 Nov 2026 (`calibrate_sanity.py`, numbers in `reward.py`).

## What to run

```
python main_backend.py compare-search --run      # all searches, then the comparison (resumes if stopped)
python main_backend.py compare-search            # the comparison again, rendering nothing
```

`backend/evaluation/compare_search.py` runs one random search
(`backend/genome/search.py`) per voice and line of baseline v2: render seed 0,
10 genomes, 3 genes, genome seed 0. That is 12 x 6 x 10 = 720 renders. The
baseline's own render is copied in as each search's plain clip (the same
renderer, empty plan and seed made it), so it is not rendered again. The same
genome seed gives the same 10 genomes on a line in every voice. It then
measures every genome like the baseline, writes the numbers below to
`output/milestone2/random-v2/summary.json`, draws the blind listening pairs,
and logs the comparison in MLflow as `search-random-v2`.

## What is measured

Per line and voice, against the plain render:

- **Share of genomes judged fine** by the sanity check (words, pace, similarity).
- **Why the others failed**: words, pace or similarity. A genome that is always
  rejected for one reason tells us which genes break the model.
- **WER, similarity and quality** of the fine genomes, against the baseline's,
  with a 95 % bootstrap interval over voices and a Wilcoxon signed-rank test on
  the paired clips.
- **Per gene type**: how often a genome with that gene was fine. This shows
  which tags CosyVoice 3 handles well.

Expressiveness itself has no automatic judge yet. For this milestone it is
judged by listening: for 10 voice-and-line pairs, chosen at random before
listening, the best fine genome and the plain render are rated blind with
LISTENING_PROTOCOL.md. "Best" is fixed before listening: fewest word errors
against the line, then most like the reference. The pairs are in
`output/milestone2/random-v2/listening/` as A.wav and B.wav in random order;
`key.json` says which is which and is opened only after rating.

## What counts as a result

Decided now:

- The search **works as a guard** if at least half of the genomes are fine and
  the fine ones keep WER and similarity within the baseline's interval.
- It **beats the baseline** only if listeners prefer the genome in more than
  half of the blind pairs. Automatic scores alone cannot show this.
- If most genomes fail, that is a finding about the genes, not a reason to
  loosen the sanity limits. The limits change only after a new calibration.

## Risks and limits

- Short lines have a weak similarity check (limit 0.80 under 6 s of speech):
  lines 0 and 1 are reported apart.
- One render seed: a genome's effect and the seed's effect are mixed. A second
  seed for the fine genomes is the first follow-up if the results are close.
- The test speakers stay unused. Only after the method is fixed is the final
  comparison run once on them.

## Results (8 Nov 2026, run `random-v2`)

`python main_backend.py compare-search --run`: 72 searches (12 voices x 6 lines
of baseline v2), 720 genomes, none failed to render. Logged in MLflow as
`search-random-v2`; everything is in `output/milestone2/random-v2/`.

**Share of genomes judged fine: 67 %** (483 of 720).

| | Fine | Failed on words | Failed on similarity | Failed on pace |
|---|---|---|---|---|
| All | 67 % | 32 % | 19 % | 65 % |
| Short lines (0, 1) | 42 % | 20 % | 20 % | 78 % |
| Longer lines (2-5) | 80 % | 48 % | 18 % | 45 % |
| VCTK | 58 % | 25 % | 16 % | 78 % |
| CREMA-D | 61 % | 28 % | 30 % | 57 % |
| LibriTTS-R | 82 % | 56 % | 5 % | 49 % |

The failure shares are of the rejected genomes; a genome can fail for more than
one reason. By line: 51, 33, 77, 69, 91 and 82 % fine. The comparison sentence
(line 5) has 82 % fine. 4 of the 72 voice-and-line pairs had no fine genome.

**Fine genomes against the baseline** (mean per voice and line, minus the plain
render; 95 % interval over voices; Wilcoxon signed-rank on 68 pairs):

| | Difference | 95 % interval | p |
|---|---|---|---|
| WER | +0.017 | +0.012 to +0.022 | < 0.001 |
| Similarity to the reference | -0.008 | -0.014 to -0.002 | 0.017 |
| PESQ (SQUIM) | -0.06 | -0.09 to -0.02 | 0.013 |
| STOI | -0.001 | -0.003 to +0.001 | 0.05 |
| SI-SDR | -0.18 dB | -0.78 to +0.39 | 0.33 |

**The verdicts, by the rules above:**

- **Works as a guard: no.** At least half the genomes are fine (67 %), and their
  similarity (0.884) stays inside the baseline's interval (0.855 to 0.903). But
  their WER (0.018) is above the baseline's interval (0 to 0.013). The sanity
  check lets a genome change up to 10 % of the words compared with the plain
  render. That is looser than the baseline's own spread, so 120 of the 483 fine
  genomes have at least one word wrong (at most 1 in 7). As decided in advance,
  the limits are not loosened or tightened after seeing this; a stricter words
  limit needs a new calibration first.
- **Beats the baseline: not decided yet.** It needs the blind listening pairs
  (below). The automatic scores only show the costs: the fine genomes are
  slightly worse on words, similarity and quality. That is expected, because
  the judges reward a plain, clean voice.

**Findings for the next step:**

- **Pace is the main reason a genome is rejected.** 125 of the 237 rejected
  genomes failed on pace alone, and 106 of the 153 pace failures had no speed
  gene. Inserted sounds (sighs, breaths, laughter) lengthen the clip, so the
  words per second drop below 0.70 of the plain render, even though the words
  themselves are not slower. The pace check should probably count speech time
  only, not the inserted sounds. That is a change to the judge, so it needs its
  own calibration, not an edit after the fact.
- **Short lines are much harder** (42 % fine against 80 %), mainly through pace:
  one inserted sound is a large share of a short clip.
- **No gene type is clearly broken.** Every type is fine in 58 to 79 % of the
  genomes that have it; `vocalized_noise` is the lowest (58 %), and
  `laughing_speech`, `breath`, `mn` and `emphasis` the highest (78-79 %).

**Listening pairs.** 10 pairs were drawn (seed 0) among the pairs with a fine
genome, in `output/milestone2/random-v2/listening/pair_01` ... `pair_10`, each
with A.wav, B.wav and line.txt. Rate them with LISTENING_PROTOCOL.md before
opening `key.json`. The genome is preferred in more than half → the search
beats the baseline.

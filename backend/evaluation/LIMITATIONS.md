# Limitations

What the first measurement shows, and what it does not.

## Data

The data comes from three public speech datasets: VCTK (CC BY 4.0), CREMA-D (ODbL) and LibriTTS-R (CC BY 4.0, as stated on OpenSLR SLR141). Every clip is listed in a manifest, and the manifest's hash is recorded with every result. The fields, the allowed labels and what is known when are written down in DATA_CONTRACT.md, and `checks.py` enforces them.

The datasets do not contain recording dates that could order the clips in time. A time-based split, as in the lecture's match example, is therefore not possible. Instead, all clips from the same speaker are kept in the same split. This is the group-aware safeguard for this data: the same voice never appears on both sides.

## Split

Each dataset is split on its own, by speaker, with seed 42 into 80 % validation and 20 % test. The result is 223 validation speakers (VCTK 87, CREMA-D 73, LibriTTS-R 63) and 56 test speakers (VCTK 22, CREMA-D 18, LibriTTS-R 16). The split is made once from the full library and saved to a file. It refuses to overwrite itself, and the checks confirm that no speaker is in both sets.

Validation is used for all development and for the baseline. The test speakers are locked and unused, so no result in this milestone has been measured on them.

## Measurements

WER uses Whisper base, similarity uses Resemblyzer, and quality uses SQUIM, which estimates PESQ, STOI and SI-SDR without a clean reference. These are automatic estimates with their own errors. Whisper can mishear clear speech, Resemblyzer can confuse similar voices, and PESQ was designed for telephone speech, not for synthetic voices; SQUIM mostly measures how clean a recording is, not how natural it sounds. No listening test has been done, so the numbers do not show whether a listener would find the output natural or the voice convincing.

Results are reported per dataset as well as overall, because one overall score can hide a weaker group. A gap between datasets is a reason to look further, not a finding in itself.

The same kind of judge is also used to choose outputs in the later search. A search that optimises for a judge will find ways to please that judge, so the final reporting will need judges that the search did not use.

## The baseline

The baseline is the voice model CosyVoice 3 (Fun-CosyVoice3-0.5B-2512) with its default settings and no genome search. It is the comparison point for any later change. It is not a tuned system, and a later change counts as an improvement only if it beats the baseline on the same validation data.

The sample on validation is small, so a difference of a few hundredths is not meaningful. Timing is measured on one machine (an RTX 3080), and it will differ on another. Reference recordings are measured with WER as well, so the recogniser's own error is visible.

### Baseline v1 (7 Nov 2026)

`python main_backend.py baseline`: 12 validation voices (4 per dataset) x 5 fixed lines of 8 to 22 words, render seed 0, 60 clips. Mean, with a 95 % bootstrap interval that resamples voices:

| | WER | Heard exactly | Similarity to the reference | PESQ (SQUIM estimate) |
|---|---|---|---|---|
| All | 0.003 (0.000-0.009) | 97 % | 0.880 (0.853-0.902) | 3.38 (2.89-3.84) |
| VCTK | 0.009 (0.000-0.027) | 90 % | 0.904 (0.893-0.914) | 3.86 (3.73-3.99) |
| CREMA-D | 0.000 | 100 % | 0.826 (0.796-0.856) | 2.18 (1.91-2.42) |
| LibriTTS-R | 0.000 | 100 % | 0.910 (0.902-0.918) | 4.08 (3.95-4.21) |

- Only 2 of 60 clips were not heard exactly, both from one VCTK voice (p302), each missing one word.
- CREMA-D clones are less similar to their reference and score lower on quality. Its references are short acted recordings of lower quality, and the clones speak about 24 % faster than the reference. This is a gap to look into, not a finding.
- The two 8-word lines give lower similarity (0.851 and 0.864) than the longer ones (0.894 to 0.896): the speaker encoder needs more speech.
- A clip takes about 7 s to render.

Known-answer checks, all passed: a reference compared with itself has similarity 1; Whisper hears the reference recordings with a median WER of 0 (mean 0.027); 59 of 60 renders are most like their own voice of the 12; and rendering the same voice, line and seed again gives the same samples.

What it shows: plain CosyVoice 3 keeps the words almost perfectly, so WER leaves little room to improve and is mainly a guard for later changes. Similarity and quality differ by dataset, so later results must be reported per dataset as well.

### Baseline v2 (8 Nov 2026): the current baseline

The same 12 voices and render seed, with a sixth line added: the shared comparison sentence "Well, you really did it this time, and I honestly didn't think you had it in you, so I'll be waiting." (21 words), which reads as praise, sarcasm or a threat depending on the delivery. 72 clips:

| | WER | Heard exactly | Similarity to the reference | PESQ (SQUIM estimate) |
|---|---|---|---|---|
| All | 0.005 (0.000-0.013) | 96 % | 0.882 (0.855-0.903) | 3.40 (2.92-3.85) |
| VCTK | 0.008 (0.000-0.023) | 92 % | 0.903 (0.892-0.914) | 3.88 (3.75-4.01) |
| CREMA-D | 0.008 (0.000-0.024) | 96 % | 0.829 (0.802-0.856) | 2.24 (1.98-2.47) |
| LibriTTS-R | 0.000 | 100 % | 0.914 (0.906-0.923) | 4.07 (3.95-4.20) |

- **It reproduces v1.** The 60 clips that v1 and v2 share give exactly the same WER, similarity and PESQ. This was checked clip by clip, and the means in `summary.json` were recomputed from `results.json` by hand.
- **The new line** has similarity 0.891, in line with the other long lines. One clip was misheard (CREMA-D 1089: "had it in you" heard as "had a denue"), so CREMA-D now has one error too.
- **Known-answer checks: 4 of 4 passed.** 70 of 72 renders are most like their own voice, and a repeated render gives the same samples.
- **Logged in MLflow** as `baseline-v2`. It was run after MLflow upgraded protobuf to 7.x, so rendering is confirmed to work with it.

### Sanity check limits

The genome search judges each genome against a plain render with a sanity check. Its limits were calibrated on 6 Nov 2026 from 400 pairs of plain renders of validation voices (`calibrate_sanity.py`); the numbers are in `reward.py`. Two plain renders of a short line differ by chance more than those of a long line, so on lines under about 6 s of speech the similarity limit is lower (0.80 instead of 0.90), and a different voice can then pass about 6 % of the time.

## Leakage risks

Three routes are checked: a voice on both sides of the split, a check or a statistic computed on the whole library before the split, and a test set used to make choices. The first is prevented by the speaker split. The second is avoided because no normalisation is computed from the full library. The third is handled by keeping the test set locked and reporting only on validation until the final evaluation. The baseline, the calibrations and the genome finder take validation voices only; the UI can list test speakers to show the split, but nothing measured on them is used.

## Run record

Each run records the manifest hash, the split file hash, the seed, the device and the library versions. Where a known-answer check exists, its result is recorded too. The baseline also records the git commit (and whether there were uncommitted changes), the CosyVoice weights and code commit, the judges, the render and pick seeds, the CUDA version and the non-secret settings, in `output/baseline/<name>/record.json`, and logs them in MLflow (`baseline-v1`, `baseline-v2`). Both baselines ran on uncommitted code (`git_dirty` is true in their records), so the commit in the record is the last commit, not the exact code. Once the code is committed, run `python main_backend.py baseline --name v3` to render it again from the commit. Because v2 already reproduces v1 exactly, v3 should give the same numbers, and its record will then point at code anyone can check out.

## Ethics and licences

Only the public datasets are used, under their licences. Generated audio imitates real people's voices, so it stays in the project and is not shared outside the group. No voice of a person who has not agreed to be used is cloned, and any future data must come with consent.

## What is not claimed

This milestone shows that the setup can be repeated and that the first numbers were measured in a documented way. It does not show that the system is good enough for users, that the search improves anything, or that the chosen judges reflect what a listener hears.

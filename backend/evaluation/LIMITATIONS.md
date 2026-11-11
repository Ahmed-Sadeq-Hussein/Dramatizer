# Limitations

What the first measurement shows, and what it does not.

## Data

The data comes from three public speech datasets: VCTK (CC BY 4.0), CREMA-D (ODbL) and LibriTTS-R (CC BY 4.0, as stated on OpenSLR SLR141). Every clip is listed in a manifest, and the manifest's hash is recorded with every result. The fields, the allowed labels and what is known when are written down in DATA_CONTRACT.md, and `checks.py` enforces them.

The datasets do not contain recording dates that could order the clips in time. A time-based split, as in the lecture's match example, is therefore not possible. Instead, all clips from the same speaker are kept in the same split. This is the group-aware safeguard for this data: the same voice never appears on both sides.

## Split

The split is made over **voices**: speakers with at least one neutral clip of 3-25 s of speech. Each voice uses one clip, its longest, because a short reference gives a clone that is less like the speaker and is heard less reliably. Each dataset is split on its own with seed 42 into 80 % validation and 20 % test. All of it is in the voice manifest, `backend/data/manifests/voice-split-v2.json`: every voice, its clip and its split. It refuses to overwrite itself, and the checks confirm that no voice is in both sets and that each voice's clip is a usable one.

| | Validation voices | Test voices | Unused speakers | Mean clip length |
|---|---|---|---|---|
| VCTK | 87 | 22 | 0 | 10.2 s |
| CREMA-D | 27 | 7 | 57 | 3.4 s |
| LibriTTS-R | 62 | 15 | 2 | 20.0 s |
| **All** | **176** | **44** | **59** | |

- **CREMA-D is the weak dataset.** 57 of its 91 actors have no neutral clip of 3 s or more (its sentences are short), and even its longest clips are only 3.0-4.8 s. Its clones are expected to be less like the speaker, whatever is changed later.
- **The clip lengths differ by dataset** (CREMA-D about 3 s, LibriTTS-R about 20 s). A gap between datasets can come from the reference length, not only from the dataset.
- **The first split was replaced on 8 Nov 2026.** It split every speaker, usable or not (223 / 56), so CREMA-D had only 5 usable test voices, and each run picked a voice's clip at random. The test set had never been used, so it could still be redrawn. Baselines v1 to v3 were made with the first split and are kept as history; v4 is the first baseline on the voice split.

Validation is used for all development and for the baseline. The test speakers are locked and unused, so no result in this milestone has been measured on them.

### The voice map (9 Nov 2026): which voices sound alike

`python main_backend.py voice-map --pick 18` (`backend/voice_data/voice_map.py`; output in `OUTPUT_DIR/voice_map/`). It maps the 176 validation voices by how alike they sound, to pick a varied set of voices for the genome search.

**Our similarity judge mostly measures the recording, not the voice.** With Resemblyzer on each voice's reference clip, 86 % of a voice's 5 nearest voices come from its own dataset (39 % by chance), and a classifier tells the dataset from the embedding 93 % of the time. CosyVoice's own speaker model (CAM++, `campplus.onnx`) does the same (82 %). Equalising the audio (same band, same loudness, silence cut) or removing noise (DeepFilterNet) leaves it at 79-86 %. Part of this is real (CREMA-D's American actors, VCTK's British accents, read against acted speech), but most is the recording setup. This also matters for every similarity score: a CREMA-D render's lower similarity (0.81 in script-v1) is partly its recording.

**What fixed it**, chosen among about 70 configurations scored on a reference clip and up to 4 other neutral clips per voice (the scripts were run outside the repository):

- Each voice is the mean of its clips' embeddings (the reference and up to 4 more).
- Each dataset's mean embedding is removed.
- Each similarity is z-normalised within its dataset pair (s-norm). The order of voices within a dataset does not change; only scores across datasets become comparable.

| Measured on held-out clips | Before | Way 1 (CAM++) | Way 2 (CAM++ + Resemblyzer) |
|---|---|---|---|
| Nearest 5 from own dataset (chance 39 %) | 86 % | 39 % | 43 % |
| A clip not used for the map finds its own voice first, of 176 | 88 % | 93 % | 95 % |
| ... within its own dataset | 89 % | 99 % | 100 % |

Resemblyzer and CAM++ are independent models; after the fix, 44 % of their 5 nearest voices from other datasets are the same (4.5 % by chance), and removing the dataset means makes them agree more within a dataset (r 0.79 to 0.85), so the fix removes noise rather than voice. LDA trained on the extra clips, dataset whitening and removing dataset directions (NAP) were worse.

**Voices form a continuum, not types.** HDBSCAN finds only two natural groups. The stable structure is two families (stability 0.98 on 80 % subsets), split by pitch (median 192 Hz against 115 Hz, most likely female and male; the datasets have no gender labels here). Below that, Way 1's 4 groups (0.85) and 8 neighbourhoods (0.67) and Way 2's 3 communities (0.88) are browsing aids. No dataset dominates any cluster (dataset NMI 0.00-0.05), except that one of the 8 neighbourhoods is 15 of 17 VCTK.

**Not shown:** no listener has judged whether neighbours on the map sound alike; pitch is the only check by ear (nearest voices are 1.4-1.6 semitones apart, random pairs 4.8). S-norm assumes every dataset's voices vary as much as the others'. Removing the dataset mean also removes accent differences between datasets.

**Fewer, more different voices** (`voice-map --dedupe`): one voice per tight cluster plus every loner, in the Way 2 similarity, because the two models agree best on which pairs are close (54 % of their 10 % most alike pairs are shared; chance 10 %). Two voices are close when they are among the 10 % most alike pairs. Loners are the 5 % of voices whose closest voice is least alike; they are kept first, and an isolated pair of loners keeps one. Then the voice with the most close voices still uncovered is kept, and those voices are dropped, until every voice is kept or close to a kept one. On 9 Nov 2026 this kept 24 of 176 voices (6 loners): LibriTTS-R 12, VCTK 10, CREMA-D 2, with 15 high-pitched and 9 low-pitched voices. No two picks are close; their most alike pair scores 1.41 standard deviations, against 2.64 for 24 random voices. The picks cover both families, all 4 groups and 7 of the 8 neighbourhoods.

Limits of this pick:

- "Tight" is relative, not "duplicate". Two halves of one voice's clips score a median of 4.0, and the most alike pair of different voices scores 3.4, so no two voices are interchangeable to the models.
- Loners are the least certain part. The two models share only 7 of their 18 most alone voices (chance about 2).
- The count follows from the two shares (10 % and 5 %), which are choices, not measurements.
- CREMA-D, the only emotional dataset, gets 2 voices, because most of its voices are close to a voice from another dataset.

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

### The script baseline, script-v1 (9 Nov 2026): the baseline for every later test

The test script, `backend/data/scripts/context-script-v1.json`, is **one line of two sentences**: "So you finally came back. I kept everything exactly the way you left it." (14 words). It names no emotion and no relationship, so the same words fit very different characters: a parent welcoming a child home, a widow at a grave, a villain, a stalker, a hurt partner, a sarcastic roommate, a relieved comrade, a ghost. The file lists 9 such characters with the delivery each one calls for. The genome's job in later tests is to bring out one character's reading from these fixed words.

The line is rendered plainly (no genome, render seed 0) in **every validation voice** of the voice manifest, each with its longest clip: 176 renders. They live outside the project, in `BASELINE_AUDIO_DIR/script-v1/` (default: the folder "Dramatizer 2 baselines" next to the project), so the repository stays small. Each voice's folder is named after its reference clip (`<dataset>_<clip id>`), and `baseline_manifest.json` lists every audio with its line and the reference clip it was cloned from (dataset, speaker, clip id, transcript, length, checksum, split). It is logged in MLflow as `baseline-script-v1`.

| | WER | Heard exactly | Similarity to the reference | PESQ (SQUIM estimate) |
|---|---|---|---|---|
| All (176 voices) | 0.009 (0.004-0.014) | 93 % | 0.906 (0.899-0.913) | 3.77 (3.69-3.85) |
| VCTK (87) | 0.011 (0.004-0.020) | 91 % | 0.919 (0.915-0.923) | 3.94 (3.90-3.99) |
| CREMA-D (27) | 0.008 (0.000-0.021) | 93 % | 0.810 (0.797-0.824) | 2.56 (2.41-2.70) |
| LibriTTS-R (62) | 0.006 (0.000-0.013) | 95 % | 0.930 (0.925-0.934) | 4.06 (4.02-4.11) |

- **All 4 known-answer checks passed.** 171 of 176 renders (97 %) are most like their own voice out of all 176, a much harder test than among 12. The reference recordings are heard with a median WER of 0.03.
- **13 of 176 renders were misheard, almost all at the start.** "finally" became "find the" or "find and" in five voices, and "So you" became "Sue" or "Are you". The second sentence is nearly always heard exactly. This start is the weakest part of the line, and a genome that slows or stresses "finally" may change its WER either way.
- **The long-reference leak happened again in LibriTTS-R 7127** ("be damned" before the line), and one other LibriTTS-R voice (3729) adds "Aw." at the start. These are 2 of 62 LibriTTS-R voices.
- **CREMA-D stays the least similar** (0.810), as in every baseline: its references are only about 3 s.
- **The intervals are narrow** because the sample is 176 voices, so a later change of a few hundredths in similarity can now be told apart from chance.
- **The final test voices are not rendered.** They stay locked until the final evaluation, where the same script is rendered in them.
- **The references are the longest clips,** so the long-reference leak seen in v4 (LibriTTS-R 7127) is expected in some LibriTTS-R voices. It is part of the plain model's behaviour that later tests are compared with.

### Baseline v4 (8 Nov 2026): the voice split, longest clips

The same 6 lines and render seed 0, on 12 validation voices of the new voice split, each with its longest neutral clip (VCTK 8-10 s, CREMA-D 3.0-3.9 s, LibriTTS-R 17-24 s). 72 clips. All 4 known-answer checks passed (70 of 72 renders are most like their own voice). Logged in MLflow as `baseline-v4`. It ran on uncommitted code (`git_dirty: true`).

| | WER | Heard exactly | Similarity to the reference | PESQ (SQUIM estimate) |
|---|---|---|---|---|
| All | 0.028 (0.007-0.058) | 78 % | 0.869 (0.828-0.904) | 3.59 (3.21-3.93) |
| VCTK | 0.022 (0.012-0.034) | 71 % | 0.900 (0.887-0.914) | 4.02 (3.89-4.15) |
| CREMA-D | 0.023 (0.000-0.062) | 88 % | 0.781 (0.752-0.826) | 2.70 (2.40-2.88) |
| LibriTTS-R | 0.039 (0.000-0.116) | 75 % | 0.925 (0.915-0.941) | 4.04 (3.96-4.12) |

- **The longest clip did not lower the WER: it rose from 0.005 (v3) to 0.028.** The voices are different from v3's, so this is not a controlled comparison, but the errors have a clear cause.
- **A very long reference leaks into the render.** LibriTTS-R 7127's reference (23 s) ends on "bedimmed", and all 6 of its renders begin with "be dimmed" or "be damned" before the line. CosyVoice warns that the line is much shorter than the reference text. Without this voice, WER is 0.016 and 85 % of clips are heard exactly.
- **VCTK drops small words** ("had", "to", "on"). All four VCTK voices use one of the long "rainbow passage" sentences, the longest a VCTK speaker reads.
- **Similarity rose for LibriTTS-R** (0.925, against 0.914 in v2) and stayed the same for VCTK (0.900 against 0.903). CREMA-D fell (0.781 against 0.829), but its longest clips are still only about 3 s and the actors are different, so this says more about the voices than about the rule.
- **What follows:** the longest clip is not automatically the best reference. A length cap (for example the longest clip up to about 12-15 s), tested on the same voices, is the next thing to check.

### Baseline v3 (8 Nov 2026): the reference baseline on the first split

The same run as v2 (12 voices x 6 lines, render seed 0), rendered again from the committed code, commit `2f99315`, with no uncommitted changes. It gives **exactly the same numbers as v2 on all 72 clips** (WER, similarity and PESQ, checked clip by clip), so the table below is v3's too. All 4 known-answer checks passed. It is logged in MLflow as `baseline-v3`.

It was the reference baseline of the first split (see Split above), and the Milestone 2 search was compared with v2 on that split. Its record points at code anyone can check out:

| Run record (`output/baseline/v3/record.json`) | |
|---|---|
| Code | git `2f99315`, clean (`git_dirty: false`) |
| Data | manifest SHA-256 `4873e073…`, split SHA-256 `3be1b163…` |
| Voice model | CosyVoice 3, weights Fun-CosyVoice3-0.5B, CosyVoice code `074ca6d` |
| Judges | Whisper base (WER), Resemblyzer (similarity), SQUIM (quality) |
| Seeds | render 0, voice pick 0 |
| Machine | NVIDIA GeForce RTX 3080, CUDA 12.1 |
| Versions | torch 2.3.1, torchaudio 2.3.1, transformers 4.51.3, openai-whisper 20231117, resemblyzer 0.1.4, jiwer 4.0.0, numpy 1.26.4, mlflow 3.18.0 |

### Baseline v2 (8 Nov 2026)

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

Each run records the manifest hash, the split file hash, the seed, the device and the library versions. Where a known-answer check exists, its result is recorded too. The baseline also records the git commit (and whether there were uncommitted changes), the CosyVoice weights and code commit, the judges, the render and pick seeds, the CUDA version and the non-secret settings, in `output/baseline/<name>/record.json`, and logs them in MLflow (`baseline-v1` to `baseline-v4`). v1 and v2 ran on uncommitted code (`git_dirty` is true in their records), so the commit in their records is the last commit, not the exact code. v3 ran from commit `2f99315` with no uncommitted changes and gives the same numbers as v2 on every clip, so the results can be repeated from code anyone can check out: `git checkout 2f99315`, then `python main_backend.py baseline --name <new name>`.

What the record cannot pin down: the voice library itself is not in git (it is identified by the manifest's hash), the model weights are identified by name only, and the timings depend on the machine.

## Ethics and licences

Only the public datasets are used, under their licences. Generated audio imitates real people's voices, so it stays in the project and is not shared outside the group. No voice of a person who has not agreed to be used is cloned, and any future data must come with consent.

## What is not claimed

This milestone shows that the setup can be repeated and that the first numbers were measured in a documented way. It does not show that the system is good enough for users, that the search improves anything, or that the chosen judges reflect what a listener hears.

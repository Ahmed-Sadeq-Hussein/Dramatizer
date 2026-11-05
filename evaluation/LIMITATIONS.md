# Limitations

What the first measurement shows, and what it does not.

## Data

The data comes from three public speech datasets: VCTK (CC BY 4.0), CREMA-D (ODbL) and LibriTTS-R (CC BY 4.0, as stated on OpenSLR SLR141). Every clip is listed in a manifest, and the manifest's hash is recorded with every result.

The datasets do not contain recording dates that could order the clips in time. A time-based split, as in the lecture's match example, is therefore not possible. Instead, all clips from the same speaker are kept in the same split. This is the group-aware safeguard for this data: the same voice never appears on both sides.

## Split

Each dataset is split on its own, by speaker, with seed 42 into 80 % validation and 20 % test. The result is 223 validation speakers (VCTK 87, CREMA-D 73, LibriTTS-R 63) and 56 test speakers (VCTK 22, CREMA-D 18, LibriTTS-R 16). The split is made once from the full library and saved to a file. It refuses to overwrite itself, and the checks confirm that no speaker is in both sets.

Validation is used for all development and for the baseline. The test speakers are locked and unused, so no result in this milestone has been measured on them.

## Measurements

WER uses Whisper base, similarity uses Resemblyzer, and quality uses PESQ. These are automatic estimates with their own errors. Whisper can mishear clear speech, Resemblyzer can confuse similar voices, and PESQ was designed for telephone speech, not for synthetic voices. No listening test has been done, so the numbers do not show whether a listener would find the output natural or the voice convincing.

Results are reported per dataset as well as overall, because one overall score can hide a weaker group. A gap between datasets is a reason to look further, not a finding in itself.

The same kind of judge is also used to choose outputs in the later search. A search that optimises for a judge will find ways to please that judge, so the final reporting will need judges that the search did not use.

## The baseline

The baseline is the voice model CosyVoice 3 (Fun-CosyVoice3-0.5B-2512) with its default settings and no genome search. It is the comparison point for any later change. It is not a tuned system, and a later change counts as an improvement only if it beats the baseline on the same validation data.

The sample on validation is small, so a difference of a few hundredths is not meaningful. Timing and memory are measured on one CPU and one machine, and they will differ on another machine. Reference recordings are measured with WER as well, so the recogniser's own error is visible.

## Leakage risks

Three routes are checked: a voice on both sides of the split, a check or a statistic computed on the whole library before the split, and a test set used to make choices. The first is prevented by the speaker split. The second is avoided because no normalisation is computed from the full library. The third is handled by keeping the test set locked and reporting only on validation until the final evaluation.

## Run record

Each run records the manifest hash, the split file hash, the seed, the device and the library versions. Where a known-answer check exists, its result is recorded too.

## Ethics and licences

Only the public datasets are used, under their licences. Generated audio imitates real people's voices, so it stays in the project and is not shared outside the group. No voice of a person who has not agreed to be used is cloned, and any future data must come with consent.

## What is not claimed

This milestone shows that the setup can be repeated and that the first numbers were measured in a documented way. It does not show that the system is good enough for users, that the search improves anything, or that the chosen judges reflect what a listener hears.

# Limitations

What the Milestone 1 baseline shows, and what it does not.

## Data

The data comes from two public datasets: VCTK (109 speakers, 88,156 clips, CC BY 4.0) and CREMA-D (91 actors, 7,442 clips, ODbL). The manifest records every clip with its speaker, script, emotion and licence, and its hash is saved with each run. CREMA-D uses twelve short sentences, and VCTK has no emotion labels, so the two sets differ in style and recording quality. CREMA-D is crowd-sourced, which explains its low quality scores.

The split is by speaker: 160 dev speakers and 40 test speakers, with seed 42. The test set is locked and is not used for any result in this milestone.

## Measurements

WER uses Whisper base, similarity uses Resemblyzer, and quality uses PESQ. These are automatic estimates and they have their own errors. Whisper can mishear clear speech, Resemblyzer can confuse similar voices, and PESQ is designed for telephone-style speech, not for synthetic voices. No listening test has been done, so the numbers do not tell us whether a listener would find the output natural or the voice convincing.

The same kind of judge is also used to choose outputs in the later search. A search that optimises for a judge will find ways to please that judge, so the final reporting will need judges that the search did not use.

## Baseline

The baseline uses five dev speakers and one target sentence per speaker. That is far too few to say how well the model performs in general. Results will vary between speakers, and a difference of a few hundredths is not meaningful at this sample size. The baseline measures default settings only, so it is a reference point and not a tuned system.

Timing and memory are measured on one CPU, on one machine, with one set of library versions. The real-time factor will be different on another machine, and memory is read after each generation, so it is a sample and not a guaranteed peak.

## Ethics and licences

Only the public datasets are used, under their licences. Generated audio imitates real people's voices, so it stays in the project and is not shared outside the group. No voice of a person who has not agreed to be used is cloned, and any future data added must come with consent.

## What is not claimed

This milestone shows that the setup can be repeated, and that the first numbers were measured in a documented way. It does not show that the system is good enough for users, that the search improves anything, or that the chosen judges reflect what a listener hears.

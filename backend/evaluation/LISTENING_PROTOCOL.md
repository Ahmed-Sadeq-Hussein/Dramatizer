# Listening protocol

A repeatable way for people to rate the generated speech. It is kept apart from the automatic measurements: the automatic numbers say what can be measured, and this protocol says what a listener perceives.

## Material

The material is a small evaluation set, taken from the dev speakers in `history/eval/baseline.json`. For each speaker, use the reference clip and the generated clip from the baseline run, with the clip IDs written down. Record the evaluation set as a list of file names with its own hash, so that everyone rates the same files.

Include the reference clips in the same session as the generated clips. Listeners should not know which is which.

## Listeners

At least three people rate each clip. Use members of the group and, if possible, people outside the project. Listeners must not be the person who generated the audio. Record each listener's name or code, and whether they use headphones.

## Setup

Listeners hear the clips in a random order, and the order is saved with the results. All clips are played at the same loudness. Use a quiet room and headphones, and allow listeners to replay a clip.

## Questions

For each clip, the listener answers three questions on a scale from 1 to 5.

1. Naturalness: how much does it sound like a person speaking? (1 = robotic, 5 = natural)
2. Same voice: does this sound like the same person as the reference clip? (1 = clearly different, 5 = clearly the same)
3. Clarity: how easy is it to understand? (1 = hard, 5 = easy)

The listener also writes down the words they heard. This gives a second intelligibility measure that does not depend on Whisper.

## Recording

Save each rating as one row: clip ID, listener, the three scores, what was heard, and the playback order number. Save the file as `history/eval/listening.csv`. Do not change ratings after they are saved.

## Reporting

Report the mean and the spread for each question. Report the agreement between listeners, for example by showing the difference between the highest and lowest score on each clip. A large spread means the question is unclear or the listeners disagree, and that is itself a finding.

Keep these results apart from the automatic measurements in MLflow. Subjective preference is a separate outcome, and it should never be merged into a single score with WER or PESQ.

# Minimum scope (proposal for group decision)

This is a draft. It becomes binding only when the group agrees on it. Each item says what we propose and why.

## Voice model

One model: CosyVoice 3 (Fun-CosyVoice3-0.5B-2512), zero-shot, with default settings. The baseline depends on this, so no other model is added in this milestone.

## Reference audio

One path: a neutral clip, a few seconds long, with its exact transcript. The reference is used as it is, with no trimming or cleaning, so that the baseline stays simple. Cleaning audio can be added later as a separate step with its own measurement.

## Delivery representation

One format: one speaker and one sentence per run, written as a wav file in the output folder with a json record beside it. The record holds the target text, the reference, the time taken and the date. This is the smallest representation that the later search can build on.

## Human selection

One workflow: a listener rates the generated clips with the listening protocol, and the best clip is chosen from those ratings. The search will later propose candidates, but a person makes the final choice.

## Genome

Not implemented in this milestone. The representation is the sentence-level gene idea in PLAN.md, and it will be defined and tested in the search milestone, against the baseline.

## Ethics and licences

Only the public datasets are used, under their licences. Voices are cloned for measurement only. Generated audio is kept in the project and is not published or shared outside the group. Any future voice must come from a person who has agreed to it.

## Changes

A change to this scope, such as a second model or a second reference path, must be agreed by the group and recorded here with the date. Changes are compared against the baseline, and a change that is not measured is not added.

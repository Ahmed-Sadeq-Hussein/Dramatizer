# Data contract

What the data looks like, what each field means, and when each piece of
information is known. `checks.py` enforces the rules marked **(checked)**; run
it with `python main_backend.py checks`.

## Sources and terms

| Dataset | Licence | Clips | Speakers | What it is |
|---|---|---|---|---|
| VCTK | CC BY 4.0 | 88,156 | 109 | Read English sentences, two microphones per clip |
| CREMA-D | ODbL | 7,442 | 91 | Actors saying 12 fixed sentences in six emotions |
| LibriTTS-R | CC BY 4.0 (OpenSLR SLR141) | 10,573 | 79 | Audiobook speech, restored |

Every clip carries its dataset's licence in the manifest **(checked)**. The
audio stays on the local disk under `VOICE_LIBRARY_DIR`; it is never committed
or shared. Cloned audio imitates real people, so it stays in the project
(see LIMITATIONS.md, Ethics and licences).

## The manifest

`history/eval/manifest.csv`, written by `make_manifest.py`: one row per clip,
sorted by dataset, speaker and clip. Its SHA-256 is logged with every run, so
a result can be tied to the exact data it was measured on.

| Field | Type | Meaning | Allowed values |
|---|---|---|---|
| `dataset` | text | Which dataset the clip comes from | VCTK, CREMA-D, LibriTTS-R **(checked)** |
| `clip_id` | text | The dataset's own clip name | unique within its dataset **(checked)** |
| `speaker` | text | The dataset's speaker id; ids repeat across datasets, so a speaker is `dataset/speaker` | never empty **(checked)** |
| `text` | text | What is said in the clip (the transcript) | never empty **(checked)** |
| `emotion` | label | The acted emotion | CREMA-D: anger, disgust, fear, happy, neutral, sad; empty elsewhere **(checked)** |
| `intensity` | label | How strongly it is acted | CREMA-D: low, medium, high, unspecified; empty elsewhere **(checked)** |
| `mic` | label | Which microphone recorded it | VCTK: mic1, mic2; empty elsewhere **(checked)** |
| `licence` | text | The dataset's licence | as in the table above **(checked)** |
| `source` | path | Where the audio file is on this machine | the file exists **(checked)** |

The columns must be exactly these, in this order **(checked)**.

## How the labels are used

The labels are **not** predicted and **not** used as targets. They are used
only to choose reference clips:

- `emotion`: only neutral clips (or clips with no emotion label) are used as references.
- `mic`: VCTK references use one microphone only.
- `intensity`: not used.

There are no human ratings in this milestone. When listening ratings are added
(LISTENING_PROTOCOL.md), they will be a new, separate file with its own contract.

## Feature timing: what is known when

This is not time-series data, so the question is not "is a feature from the
future?" but "is anything measured *after* rendering used *before* it?"

| Known before a clip is rendered | Known only after |
|---|---|
| The reference clip and its transcript (`text`) | What Whisper hears (WER) |
| The line to be spoken | Speaker similarity to the reference |
| The speaker's split (validation or test) | Quality (PESQ, SQUIM), pace, speech time |
| The render seed, the model and its settings | The sanity check's verdict |
| The genome (in the search) | |

Rules that follow from this:

1. A render uses only the left column. Nothing from the right column of a clip
   decides how that same clip is rendered.
2. Limits and thresholds (the sanity limits, the similarity limits) are set from
   **validation** clips only, and written in the code with the date and the
   numbers they came from. They are never tuned on the test speakers.
3. No statistic is computed over the whole library before the split (for
   example, no normalisation over all speakers).
4. The test speakers are not searched on, measured or used for any choice
   until the final evaluation. The baseline, the calibrations and the genome
   finder take validation voices only. The UI's Random voices list can show
   final test speakers when asked (to see the split), but nothing heard or
   measured from them may be used to tune anything.

## The split

`backend/data/manifests/split-validation-test.json`: speaker keys
(`dataset/speaker`) in `validation` and `test`. It is made once by
`split_voice_data.py` (seed 42, 20 % test, per dataset) and refuses to be
overwritten. Rules: both sets present, no speaker in both, every clip's
speaker in one of them **(all checked)**.

## Changing the contract

A new dataset, field or label value means changing this file, `FIELDS`,
`LICENCE` and `ALLOWED` in `checks.py`, and the known-answer tests in
`test_checks.py`, in the same commit.

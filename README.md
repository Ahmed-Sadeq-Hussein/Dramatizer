# Dramatizer 2

Dramatizer finds a more expressive way to speak a line in a cloned voice. A
small "genome" of delivery edits (stress this word, pause here, laugh there,
speak a little faster) is turned into markup for a voice cloning model,
rendered, scored automatically, and shortlisted for a person to rate. The
winning delivery seeds the search for the next, similar voice.

The voice cloning model is CosyVoice 3 (Fun-CosyVoice3-0.5B-2512).

Dramatizer 2 is a rebuild of the first version with a modular architecture:
the voice cloning model, the markup rules, the search and the measurements
each live in their own part and can be swapped without touching the rest.

## Status

The prototype includes voice generation, a voice library browser and audio
measurements. Delivery genomes, local mutation search and a structural genome
distance can be used from Python. The human review loop is still to come.

## Delivery genomes

`genome.py` holds the representation and editing functions; `model.py`
exposes `Gene`, `Genome` and `prepare_delivery` without loading a voice model.
The genome module has no model dependencies.

```python
import json
import model

genome = model.Genome([
    model.Gene("breath", position=0.0),
    model.Gene("emphasis", position=0.5),
    model.Gene("pause", position=0.75, kind="long"),
    model.Gene("speed", rate=1.1),
])
delivery = model.prepare_delivery("We really did pull this off", genome)
print(delivery.markup)
# [breath] We really did <strong>pull</strong> this... off
print(delivery.speed)  # 1.1

# Editing returns a new genome; the original is unchanged.
child = genome.modify(1, position=1.0)
child = child.add(model.Gene("laughter", position=1.0))
child = child.remove(0)
child = child.replace(0, model.Gene("sigh", position=0.0))

# Store this dictionary alongside a run, or restore it later.
record = json.dumps(genome.to_dict())
restored = model.Genome.from_dict(json.loads(record))
```

To synthesize with the existing voice setup:

```python
from Voice_model import generate_speech

path = generate_speech(
    "path/to/reference.wav", "Exact reference transcript.",
    "We really did pull this off", genome=genome,
)
```

Positions range from 0 (first word) to 1 (last word), mapping to the nearest
word, so edits transfer across sentences. Events precede a word; emphasis
and laughing speech wrap one word. Speed is global, ranges from 0.5 to 2.0,
and uses CosyVoice's speed parameter. A genome allows up to six genes, with
one speed gene. Invalid parameters are rejected; they are never clamped.
An empty genome gives neutral delivery. Supply plain text when using genes.

Supported feature names are listed in `genome.FEATURES`. CosyVoice token
spellings follow its [upstream tokenizer](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/tokenizer/tokenizer.py).
Token availability does not guarantee a convincing audible effect: verify
features by listening with your installed CosyVoice 3 weights before search.
`pause` uses a comma (`short`) or ellipsis (`long`), not timed silence;
existing punctuation immediately after the word takes precedence.
For reward evaluation, keep the original target text as the transcript and
pass `genome.speed` and `genome.features` separately. A persistent genome
library is still to come. Normal use of `model.py` needs `python-dotenv`
from the project's requirements.

## Random-parent genome exploration

`genome_search.py` uses the population strategy from the first Dramatizer.
At each step it picks a random parent from the seed and all accepted children,
then adds, removes, alters or replaces one gene. A new child is accepted when
both raw measurements pass the inclusive limits:

- Word error rate <= 0.10.
- Speaker cosine similarity to the reference >= 0.80.

Passing children join the parent pool regardless of their parent's scores.
The initial genome is a starting parent, excluded from the accepted archive
and shortlist until a new child passes the gates. Exact duplicate genomes
are skipped before rendering, including repeats of rejected or failed attempts.
Gene-list order is part of identity because simultaneous vocal events render
in that order. Duplicate attempts consume a search step.

After all steps, greedy farthest-point sampling selects up to 10 accepted
candidates. It starts with the first accepted candidate, then repeatedly
chooses the candidate furthest from its nearest selected neighbour. Distances
use the genome vectors described below. If fewer than 10 pass, all are returned.
The shortlist measures diversity of delivery instructions. The previous
project used acoustic descriptors, which would need separate measurements.

Run actual audio exploration from Python in the configured voice environment:

```python
import json
from pathlib import Path
from genome import Genome
from genome_search import SearchConfig, explore_voice

result = explore_voice(
    reference_audio="path/to/reference.wav",
    reference_text="Exact words spoken in the reference.",
    text="We really did pull this off.",
    initial=Genome(),  # or a previous winning genome
    config=SearchConfig(steps=100, seed=42),
)
print(len(result.candidates), "accepted candidates")
for candidate in result.shortlist:
    print(candidate.step, candidate.evaluation.audio_path,
          candidate.evaluation.wer, candidate.evaluation.speaker_similarity)

Path("runs").mkdir(exist_ok=True)
Path("runs/search.json").write_text(
    json.dumps(result.to_dict(), indent=2, allow_nan=False), encoding="utf-8"
)
```

`explore_voice` uses `generate_speech`, `WER.word_error_rate` and
`Similarity.speaker_similarity`, so generation, Whisper and Resemblyzer must
be set up. WER is measured against the original script. These are raw
measurements rather than the transformed rewards from `reward.py`.
The generator writes each evaluated clip to its normal output folder.
Rejected/error audio may remain there, but the saved accepted genome archive
and shortlist contain only passing candidates. Duplicate genomes are never
rendered again within the same run.

For a cheap deterministic development check, the callback API can be used
without neural models:

```python
from genome_search import Evaluation, SearchConfig, explore

result = explore(
    lambda genome: Evaluation(wer=0.10, speaker_similarity=0.80),
    config=SearchConfig(steps=30, seed=42),
)
assert all(c.evaluation.wer <= 0.10 and
           c.evaluation.speaker_similarity >= 0.80 for c in result.candidates)
print(len(result.candidates), len(result.shortlist))
```

The callback returns `Evaluation(wer, speaker_similarity, audio_path=None)`.
`local_search` remains available as an entry point with this same callback
contract. Numeric reward callbacks must be changed to return measurements.
Exceptions are recorded with status `error` and the search continues. Missing,
nonfinite or invalid measurements cannot pass the gates. `result.history`
records each proposal's parent step, operation, status and available metrics.
`result.to_dict()` uses schema version 2 and stores each accepted genome once,
with shortlist references by step ID. It records the search configuration.
`result.evaluations` counts unique attempted callback calls, including errors,
and is at most `steps`.

The seed controls parent selection and mutation. Full audio reproducibility
also depends on rendering randomness. `features` restricts newly introduced
features to a subset verified by listening. Vector bins, speed weight and
shortlist size are configurable in `SearchConfig`. Defaults are five bins,
0.25 speed weight and 10 samples. Gates measure intelligibility and identity;
final preference among the diverse passing samples is left to a listener.

## Genome vectors and distance

`genome_distance.py` provides `genome_vector`, `vector_labels` and
`genome_distance`. By default, each local feature gets five position bins
(0, 0.25, 0.5, 0.75, 1). Short and long pauses have separate channels.
A gene is split linearly between its two nearest bins, so small position
changes give small vector changes. Contributions are divided by the six-gene
cap, retaining feature counts without depending on variable list length.
One extra coordinate represents global speed: `0.25 * log2(rate)`. Neutral
speed is zero; half and double speed map to -0.25 and +0.25.

This gives 76 coordinates: 15 channels x 5 bins + speed. Feature type is
represented by separate channels, not arbitrary integer feature IDs.
Euclidean distance is the square root of summed squared coordinate differences.

```python
from genome_distance import genome_vector, vector_labels, genome_distance

vector = genome_vector(genome)
named = dict(zip(vector_labels(), vector))
distance = genome_distance(genome, child)
```

`bins` and `speed_weight` are configurable design choices; use identical
settings across a dataset and store them with clustering results. They have
not been fitted to listener ratings. This is structural distance, not an
audio similarity or a mood label. It ignores gene-list order, can merge
different placements within a bin, and can distinguish genes that happen to
render the same words on a short line. Do not use zero distance as genome
identity. This representation can later be compared against a learned
embedding without changing the genome or the mutation search.

## Configuration

Settings that differ per machine (model paths, voice folder, GPU or CPU) live
in a `.env` file in this folder, never in the code. Each person keeps their
own `.env`, and it is never committed.

| Variable | Meaning |
|---|---|
| `TTS_BACKEND` | Voice cloning backend: `cosyvoice3`, or `demo` to run without a model |
| `VOICE_MODEL_DIR` | The voice cloning model's code: a checkout of [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) |
| `COSYVOICE_MODEL_DIR` | The CosyVoice 3 weights (`Fun-CosyVoice3-0.5B-2512`) |
| `DEVICE` | `cuda` or `cpu` |
| `VOICES_DIR` | One sub-folder per voice with `reference.wav` and `reference.txt` |
| `RUNS_DIR` | Where exploration runs are written |
| `HISTORY_PATH` | The stored winning genomes |
| `WHISPER_MODEL` | Speech recogniser size used for the word error rate |
| `DENOISE_MODEL` | Background noise remover for `Clean_Audio_Distortion.py`: `deepfilternet3` |
| `LOG_LEVEL` | `DEBUG`, `INFO`, `WARNING` or `ERROR` |

## Still to come

- Installation and the exact commands to run
- Project structure
- How to add a new voice cloning backend
- Tests
- Limitations and responsible use

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

## Local mutation search

`genome_search.py` adds, removes, alters or replaces one gene per step.
It starts from a supplied genome (neutral by default), scores the proposal,
and accepts it if its score is higher or equal. A worse proposal is recorded
but the next step starts from the last accepted genome. Ties allow movement
across flat scores; `accept_equal=False` makes acceptance strictly improving.
The best result is retained separately, so on a flat run it remains the seed.
This is a local hill climber and can get stuck; compare multiple seeds.

The callback defines what a higher score means. This cheap executable example
tests the search against a target genome; it does NOT evaluate speech:

```python
import json
from pathlib import Path
from genome import Gene, Genome
from genome_distance import genome_distance
from genome_search import SearchConfig, local_search

target = Genome([Gene("emphasis", position=0.6), Gene("speed", rate=1.1)])
result = local_search(
    lambda candidate: -genome_distance(candidate, target),
    config=SearchConfig(steps=100, seed=42, features=("emphasis", "speed")),
)
print(result.initial_score, result.best_score, result.evaluations)
print(result.best.to_dict())
# Save only when wanted; the search itself writes no files.
Path("runs").mkdir(exist_ok=True)
Path("runs/search.json").write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
```

For real audio, supply a callback that calls `generate_speech(...,
genome=candidate)` and returns a finite score from the existing measurements.
For example, with a fixed `reference`, `reference_text`, `text` and a neutral
render `neutral_audio`:

```python
from Voice_model import generate_speech
from reward import wer_reward, perq_reward, similarity_reward

def score_audio(candidate):
    audio = generate_speech(reference, reference_text, text, genome=candidate)
    return (
        wer_reward(audio, text, speed=candidate.speed,
                   features=candidate.features, neutral_audio=neutral_audio)
        * perq_reward(audio)
        * similarity_reward(reference, audio, neutral_audio=neutral_audio)
    )

# result = local_search(score_audio, initial=previous_winner,
#                       config=SearchConfig(steps=30, seed=42))
```

That product is a guard-only baseline. It does not reward expressiveness;
an expressive preference/objective still needs to be defined and evaluated.
`features` in SearchConfig restricts newly added/replacement features to ones
verified by listening. Seed genes outside that set can still be altered or
removed. Each run makes at most `steps + 1` callback calls, including the seed.
Exact repeat genomes reuse cached scores; proposal history records cached
and accepted status. A callback exception or invalid score stops the run.
For stochastic speech, control render seeds in the callback or score a fixed
aggregate of renders. The search seed controls mutations, not CosyVoice.
Scores must stay stable and independent of search history within a run;
an archive-dependent novelty score would need different caching logic.

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

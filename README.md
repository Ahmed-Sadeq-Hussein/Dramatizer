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
measurements. Delivery genomes can now be applied from Python. Search,
genome distance and the human review loop are still to come.

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
No automatic search, random mutation policy, distance metric or persistent
genome library is implemented yet. The editing methods provide building
blocks for those. For reward evaluation, keep the original target text as
the transcript and pass `genome.speed` and `genome.features` separately.

Run the genome and mocked generation tests from this folder:

```text
python -m unittest discover -s tests -v
```

These tests use the standard library and replace dotenv loading and audio
dependencies in the generation test. They need no GPU or model weights and
do not test acoustic quality. Normal use of `model.py` needs `python-dotenv`
from the project's requirements.

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

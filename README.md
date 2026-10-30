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

Early setup. Nothing runs yet. This README grows with the project.

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

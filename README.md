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

Milestone 1, a reproducible baseline, is in place:

- The data: three public datasets, one manifest, a data contract with checks
  ([`backend/evaluation/DATA_CONTRACT.md`](backend/evaluation/DATA_CONTRACT.md)).
- A locked split by voice: each usable voice uses its longest neutral clip, and the
  voices are split 80 % validation, 20 % final test (176 / 44), never touched. The
  voice manifest is `backend/data/manifests/voice-split-v2.json`.
- The baseline: plain CosyVoice 3 renders of fixed lines in validation voices,
  measured, with known-answer checks and a full run record in MLflow.
- What it does not show: [`backend/evaluation/LIMITATIONS.md`](backend/evaluation/LIMITATIONS.md).
- The next step, decided in advance: [`backend/evaluation/MILESTONE_2.md`](backend/evaluation/MILESTONE_2.md).
  It has been run: a random genome search keeps 67 % of genomes, but it is not yet a strict
  enough guard on the words, and whether it beats the baseline waits on blind listening.

## Running it

Use the project's Python environment. Every back-end command goes through
`main_backend.py`, the website through `main_frontend.py`:

```
python main_backend.py              # list every command
python main_backend.py checks       # the data contract and the split
python main_backend.py baseline     # the baseline (renders on the GPU)
python main_backend.py baseline --name script-v1 --script backend/data/scripts/context-script-v1.json --voices all
                                    # the script baseline: the test line in every validation voice
python main_backend.py voice-map --pick 18   # 3D map and clusters of the validation voices, 18 spread-out picks
python main_backend.py voice-map --dedupe     # or: one voice per tight cluster plus every loner
python main_backend.py explore --hours 6.5     # far-apart genomes for those voices, in 3 MOS piles (renders)
python main_backend.py compare-search --run   # Milestone 2: random genome search vs the baseline
python main_backend.py test         # the tests that need no GPU
python main_frontend.py             # the website (V4)
```

The code is in `backend/` (models, metrics, data, speech, genomes, evaluation, api)
and the website in `frontend/`.

To let the group use it from their own PCs: the back end runs as an API on the
host's PC (`python main_backend.py api`, tunnelled with playit.gg), and everyone
else runs the client (`python main_frontend.py connect https://ADDRESS:PORT`).
Step by step: [`SHARING.md`](SHARING.md).

## Configuration

Settings that differ per machine (model paths, voice folder, GPU or CPU) live
in a `.env` file in this folder, never in the code. Each person keeps their
own `.env`, and it is never committed.

| Variable | Meaning |
|---|---|
| `VOICE_MODEL_DIR` | The voice cloning model's code: a checkout of [CosyVoice](https://github.com/FunAudioLLM/CosyVoice) |
| `COSYVOICE_MODEL_DIR` | The CosyVoice 3 weights (`Fun-CosyVoice3-0.5B-2512`) |
| `DEVICE` | `cuda` or `cpu` |
| `VOICES_DIR` | One sub-folder per voice with `reference.wav` and `reference.txt` |
| `VOICE_LIBRARY_DIR` | The datasets, one sub-folder each (`CREMA_D_FOLDER`, `VCTK_FOLDER`, `LIBRITTS_R_FOLDER`) |
| `OUTPUT_DIR` | Renders, genome searches and baselines (default `D:/Dramatizer 2 renders`, off the project drive) |
| `WER_MODEL` | Speech recogniser for the word error rate: `whisper-base` (or tiny, small, medium) |
| `SIMILARITY_MODEL` | Speaker similarity: `resemblyzer` |
| `PERQ_MODEL` | Estimated quality: `squim` |
| `VOICE_SEED` | Seed of every render (default 0; `random` for a new one each time) |
| `DENOISE_MODEL` | Background noise remover for `Clean_Audio_Distortion.py`: `deepfilternet3` |

`.env.example` lists every setting with its default.

## Still to come

- Installation
- How to add a new voice cloning backend

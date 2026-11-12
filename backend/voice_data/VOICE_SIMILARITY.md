# Voice similarity: the heatmap, and the closest voice with a genome

`backend/voice_data/voice_similarity.py` compares every voice in the data manifest with every
other voice, using Resemblyzer, the project's speaker-similarity judge. It does all of the
processing on the GPU. With the result you can:

- see which voices sound alike (the heatmap);
- take a new voice, find the closest voices we know, and suggest the genome of the closest
  voice that has a rated genome.

Run everything from the project folder, with the project's Python:

```bash
E:/envs/dramatizer/python.exe main_backend.py voice-similarity build
```

## 1. Build the map (once, and again after the manifest or the genomes change)

```bash
python main_backend.py voice-similarity build
```

What it takes:

- Every clip in `backend/data/manifests/voices-v1.json` with **more than 3 s of speech**.
- **Not** the 44 locked test voices: `voice-split-v2.json` keeps them unused until the final
  evaluation.
- A voice is a dataset and a speaker, such as `VCTK/p250`. Its embedding is the mean of its
  clips' embeddings.

**How long:** about 14,000 clips (23 h of speech) took minutes on the RTX 3080.

**Time limit:** the build never runs for more than `--max-minutes` (default 120). If it
finds, after the first two minutes, that it won't fit:

1. It keeps fewer clips per voice: every voice stays in, each with an even share of its clips.
2. If time still runs out, it stops and builds the map from the clips done so far.

Options:

| Option | What it does |
|---|---|
| `--max-minutes 30` | A tighter time limit |
| `--min-seconds 5` | Only clips with more speech than this |
| `--max-clips-per-voice 20` | Fewer clips per voice from the start |
| `--limit 500` | A quick trial on 500 clips |
| `--fresh` | Embed every clip again |

**Stopping and continuing:** the clip embeddings are saved every two minutes. A build that
was stopped (or a later build after new clips were added) only embeds the clips it hasn't
seen.

**GPU and CPU:** these steps run on the GPU:
- resampling to 16 kHz;
- volume;
- cutting long silences;
- the mel spectrogram;
- the Resemblyzer network;
- the voice-against-voice matrix.

Two steps can't run on a GPU, so they run on CPU threads alongside it: decoding the audio
files, and Resemblyzer's speech detector (webrtcvad). The detector only marks which 30 ms
windows have speech; the GPU does the cutting.

**Is the GPU path the same as Resemblyzer?** Check it at any time:

```bash
python main_backend.py voice-similarity check
```

It compares the GPU embeddings with Resemblyzer's own code on 20 clips. On 12 Nov 2026 the
cosine between the two was 0.9998 on average and 0.998 at the lowest, so the GPU path gives
the same answers. The unit tests check the silence cutting, the 1.6 s windows and the mel
spectrogram step by step.

## 2. Where everything is

The data is in `history/voice_similarity/`. It holds embeddings of real people's voices, so
it is **git-ignored**, by `/history/` and by its own line in `.gitignore`.

| File | What |
|---|---|
| `similarity.npy` | The voices × voices matrix (cosine similarity, float32), in the order of `voices.json` |
| `voices.json` | Each voice: its dataset, its split (validation or unused), the clips and seconds used, and its 5 closest voices |
| `voice_embeddings.npy` | One 256-number embedding per voice, in the same order |
| `clip_embeddings.npz` | One embedding per clip (keys, voices, embeddings): what a later build reuses |
| `build.json` | When it was built, how long it took, the GPU, the scope, and whether it had to cut clips or stop early |

The heatmap image is `D:\Dramatizer 2 renders\voice similarity\heatmap.png` (OUTPUT_DIR):

- The voices are grouped by dataset, and alike voices sit together within a dataset.
- Darker means more alike.
- The red lines on the colour bar mark 0.70 and 0.80.

To draw it again without embedding anything:

```bash
python main_backend.py voice-similarity heatmap
```

### Reading a similarity

The scale was calibrated with `calibrate_similarity.py` (see `backend/metrics/reward.py`):

| Similarity | Means |
|---|---|
| 0.80 and above | Sounds like the same person |
| 0.70 to 0.80 | Alike: a close voice, but not the same person |
| below 0.70 | Different people (95 % of different-speaker pairs are below 0.70) |

## 3. A new voice: the closest voice, and the genome to suggest

```bash
python main_backend.py voice-similarity closest path/to/new_voice.wav
```

It embeds the recording the same way (on the GPU) and prints:

1. **The closest voices in the map** (`--top 5` by default), each with its similarity and
   whether it has a genome.
2. **The closest voice that has a genome.** A voice has a genome when a genome search was
   made for it and listeners rated it. Its genome is the best-rated one: by its gain over its
   plain line first, then its MOS.
3. **The suggestion:**
   - which genome (run and number, and its name if it was named);
   - its MOS and its gain over its plain line;
   - what it does, in words and as markup.

   It also says how close the voices are ("sounds like the same person", "alike", or "a
   different person: treat the suggestion as a weak start").

Add `--json` to get the same as JSON, for another program.

From Python:

```python
from backend.voice_data import voice_similarity as vs

result = vs.closest("new_voice.wav")        # {"closest": [...], "suggestion": {..., "genome": {...}}}
genome = result["suggestion"]["genome"]     # run, genome number, genes, markup, MOS, gain over the plain line
```

To render the suggested genome in the new voice, open it in the website's Genome studio (the Make genomes mode)
from its run and number, or pass its `genes` to `backend.genome.editor.render`.

**Keeping it current:**
- The map needs a rebuild only when the manifest changes.
- The genomes are read fresh on every `closest` call, so a new search that listeners have
  rated counts straight away.
- A voice with genomes that isn't in the map (none of its clips has more than 3 s of speech)
  is listed at the end of the output. Lower `--min-seconds` and rebuild to bring it in.

## 4. The same line in every voice (the rendered-line map)

The dataset clips differ in wording, length, emotion and recording. To remove the wording and
the recording, the second map compares every voice saying **the same line**, rendered plain
(no genome) by CosyVoice 3:

- **Line:** the listening batch's line, "Well, you really did it this time, …".
- **Seed:** 0.
- **Reference clip:** each voice's split clip. A voice outside the split uses its longest
  neutral clip of 3–25 s.

```bash
python main_backend.py voice-similarity render                # render the line in every voice (GPU, ~6 s each)
python main_backend.py voice-similarity render --reuse-only   # only copy the renders that already exist
python main_backend.py voice-similarity build --rendered      # the map from the renders, and how it compares
python main_backend.py voice-similarity heatmap --map rendered
python main_backend.py voice-similarity closest new.wav --map rendered
```

How `render` works:

- **Reuses renders:** the 24 voices of the listening batch already have this exact render
  (same line, seed and reference clip), so they are copied, not rendered again.
- **Resumes:** every other voice is rendered once, so a stopped run continues where it
  stopped.
- **Time limit:** `--max-minutes` (default 120).
- **Another line:** `--line "…"` with another `--out` folder.

Where the files are:

| What | Where |
|---|---|
| The renders | `D:\Dramatizer 2 renders\voice similarity\rendered_line\`, with `renders.json`: each voice's reference clip, its emotion, and whether it was rendered or copied |
| The map | `history/voice_similarity/rendered_line/` (git-ignored): the same files as the dataset map, plus `compare_with_dataset_clips.json` |
| The heatmap | `D:\Dramatizer 2 renders\voice similarity\heatmap_rendered_line.png` |

`compare_with_dataset_clips.json` says how the two maps agree:

- the rank correlation of the voice-pair similarities;
- how often a render is closest to its own voice's dataset clips;
- each render's similarity to its own voice, which shows how faithfully CosyVoice cloned it.

**Which map for `closest`?**
- A new recording of a real person: use the dataset map (the default). Real recordings
  against real recordings.
- A CosyVoice render: use `--map rendered`. Render against render.

## 5. Limits to keep in mind

- **Similar voices are not proof that a genome transfers.** "A genome that worked for the
  closest voice" is a starting point for a search, not a guarantee. Check it by listening, as
  for any genome.
- **Few voices have genomes yet:** 24, one listener each. The suggestion improves as more
  voices are searched and rated.
- **Emotion lowers similarity.** CREMA-D clips are acted in several emotions, and an emotional
  clip of the same actor scores about 0.08 lower. A voice's embedding averages all its clips
  over 3 s.
- **The test voices are left out** of the map on purpose. Don't add them before the final
  evaluation.

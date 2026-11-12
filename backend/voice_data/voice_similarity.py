"""voice_similarity.py | Every voice against every other voice (Resemblyzer, on the GPU), and the closest voice with a genome.

    python main_backend.py voice-similarity build                  # embed, compare, draw the heatmap (2 hours at most)
    python main_backend.py voice-similarity build --max-minutes 30 # a tighter time limit: fewer clips per voice
    python main_backend.py voice-similarity heatmap                # draw the heatmap again from the stored data
    python main_backend.py voice-similarity closest my_voice.wav   # the closest voices, and the genome to suggest
    python main_backend.py voice-similarity check                  # GPU path against Resemblyzer's own code
    python main_backend.py voice-similarity render                 # the same line rendered in every voice (CosyVoice 3)
    python main_backend.py voice-similarity build --rendered       # the map from those renders instead

The guide is VOICE_SIMILARITY.md, next to this file.

What it does:

    1. clips    every clip of the data manifest (data/manifests/voices-v1.json) with more than 3 s of speech,
                of every voice except the locked test voices (voice-split-v2.json: unused until the final
                evaluation). A voice is a dataset and a speaker: "VCTK/p250".
    2. embed    each clip to Resemblyzer's 256 numbers, the project's similarity judge (metrics/Similarity.py).
                The files are decoded and Resemblyzer's silence detector (webrtcvad, which has no GPU version)
                runs on CPU threads, alongside the GPU; everything else runs on the GPU: resampling to 16 kHz,
                the volume, trimming the silences, the mel spectrogram, the encoder, in large batches.
    3. voices   a voice's embedding is the mean of its clips' embeddings (unit length).
    4. compare  cosine similarity of every voice with every other voice, on the GPU.
    5. heatmap  the matrix as an image, voices grouped by dataset and, within a dataset, alike voices together.

Time limit: --max-minutes (default 120). After the first minutes it projects the total; if that is over the
limit, it keeps fewer clips per voice (the same voices, each with an evenly cut share of its clips). If time
still runs out, it stops and uses the clips done so far. Clip embeddings are saved as it goes, so a stopped
build continues where it stopped, and a later build only embeds clips it has not seen.

Stored in history/voice_similarity/ (git-ignored: embeddings of real people's voices):

    clip_embeddings.npz   one row per clip: its key (dataset/clip id), voice, embedding
    voices.json           the voices in matrix order: dataset, split, clips used, seconds, closest voices
    similarity.npy        the voices x voices matrix (float32)
    build.json            when, what scope, how long, on which GPU

The heatmap image goes to OUTPUT_DIR/voice similarity/heatmap.png.

The similarity scale (calibrated with calibrate_similarity.py, see metrics/reward.py): 0.80 and above sounds
like the same person; 0.70 to 0.80 alike; below 0.70 different people.
"""
import argparse
import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from backend import model, safe_files

SR = 16000
MIN_SECONDS = 3.0
MAX_MINUTES = 120
DATA_DIR = model.PROJECT_DIR / "history" / "voice_similarity"
IMAGE_DIR = model.OUTPUT_DIR / "voice similarity"
MANIFEST = model.DATA_DIR / "manifests" / "voices-v1.json"
CLIP_CACHE = DATA_DIR / "clip_embeddings.npz"
RENDER_DIR = IMAGE_DIR / "rendered_line"            # the renders (audio: on the render drive)
RENDERED_DATA = DATA_DIR / "rendered_line"          # their map (git-ignored, like the rest)
MAPS = {"dataset": DATA_DIR, "rendered": RENDERED_DATA}
BATCH_CLIPS = 64              # clips per GPU step
PARTIALS_PER_PASS = 4096      # 1.6 s windows through the encoder at once
LOADERS = 8                   # CPU threads decoding files and running the silence detector
SAVE_EVERY = 120              # seconds between saves of the clip embeddings
PILOT_SECONDS = 120           # measure this long before deciding whether the scope must shrink
SAME_PERSON, ALIKE = 0.80, 0.70


# ---- which clips -----------------------------------------------------------------------------

def voice_key(dataset, speaker):
    return f"{dataset}/{speaker}"


def clip_key(dataset, clip_id):
    return f"{dataset}/{clip_id}"


def test_voices():
    from backend.voice_data.split_voice_data import load_voice_manifest
    return {v["voice"] for v in load_voice_manifest()["voices"] if v.get("split") == "test"}


def splits():
    from backend.voice_data.split_voice_data import load_voice_manifest
    return {v["voice"]: v["split"] for v in load_voice_manifest()["voices"]}


def manifest_clips(min_seconds=MIN_SECONDS, path=MANIFEST, locked=None):
    """The manifest's clips with more than min_seconds of speech, outside the locked test voices."""
    locked = test_voices() if locked is None else locked
    clips = safe_files.read_json(path)["clips"]
    return [c for c in clips if c["speech_seconds"] > min_seconds and voice_key(c["dataset"], c["speaker"]) not in locked]


def cap_per_voice(clips, cap, keep=()):
    """At most `cap` clips per voice, always keeping clips in `keep` (already embedded); the same choice every time."""
    keep = set(keep)
    by_voice = defaultdict(list)
    for c in clips:
        by_voice[voice_key(c["dataset"], c["speaker"])].append(c)
    out = []
    for voice, items in by_voice.items():
        seed = int(hashlib.sha256(voice.encode()).hexdigest()[:8], 16)
        order = [items[i] for i in np.random.default_rng(seed).permutation(len(items))]
        done = [c for c in order if clip_key(c["dataset"], c["clip_id"]) in keep]
        rest = [c for c in order if clip_key(c["dataset"], c["clip_id"]) not in keep]
        out += done + rest[:max(0, cap - len(done))]
    return out


def attach_sources(clips):
    """Each clip with where its audio is (the wav file, or the parquet file, row group and row)."""
    from backend.voice_data.split_voice_data import read_library
    where = {clip_key(c.dataset, c.clip_id): c for c in read_library()}
    out, missing = [], []
    for c in clips:
        found = where.get(clip_key(c["dataset"], c["clip_id"]))
        if found is None:
            missing.append(clip_key(c["dataset"], c["clip_id"]))
        else:
            out.append({**c, "source": str(found.source), "row_group": found.row_group, "row": found.row})
    return out, missing


def work_units(clips):
    """Clips grouped so each file (or parquet row group) is read once: [[clip, ...], ...] in file order."""
    groups = defaultdict(list)
    for c in clips:
        groups[(c["source"], c["row_group"])].append(c)
    return [groups[k] for k in sorted(groups, key=lambda k: (k[0], -1 if k[1] is None else k[1]))]


# ---- reading audio (CPU threads) -------------------------------------------------------------

def decode_unit(unit):
    """[(clip, samples float32 mono, rate)] for one file or row group."""
    import io

    import soundfile as sf
    first = unit[0]
    out = []
    if first["row_group"] is not None:
        import pyarrow.parquet as pq
        column = pq.ParquetFile(first["source"]).read_row_group(first["row_group"], columns=["audio"]).column("audio")
        for c in unit:
            data, rate = sf.read(io.BytesIO(column[c["row"]].as_py()["bytes"]), dtype="float32")
            out.append((c, data.mean(axis=1) if data.ndim > 1 else data, rate))
    else:
        for c in unit:
            data, rate = sf.read(c["source"], dtype="float32")
            out.append((c, data.mean(axis=1) if data.ndim > 1 else data, rate))
    return out


def vad_flags(wav16):
    """webrtcvad's speech flag for every 30 ms window, as Resemblyzer's trim_long_silences computes it."""
    import webrtcvad
    window = 30 * SR // 1000
    wav16 = wav16[:len(wav16) - len(wav16) % window]
    pcm = np.round(wav16 * 32767).astype("<i2").tobytes()
    vad = webrtcvad.Vad(mode=3)
    return np.array([vad.is_speech(pcm[i * 2:(i + window) * 2], sample_rate=SR) for i in range(0, len(wav16), window)],
                    dtype=np.float32)


# ---- embedding (GPU) -------------------------------------------------------------------------

class GpuEmbedder:
    """Resemblyzer's preprocess_wav + embed_utterance, batched on the GPU (same model, same steps)."""

    def __init__(self, device=None):
        import torch
        import torchaudio
        self.torch = torch
        self.F = torchaudio.functional
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.device == "cpu":
            print("No GPU found: running on the CPU, which is much slower.")
        from resemblyzer import VoiceEncoder
        self.encoder = VoiceEncoder(device=self.device, verbose=False).eval()
        # librosa.feature.melspectrogram as Resemblyzer calls it: power 2, Slaney mel, centred, zero padding
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=400, hop_length=160, n_mels=40, power=2.0,
                                                        center=True, pad_mode="constant", norm="slaney",
                                                        mel_scale="slaney").to(self.device)
        self.loaders = ThreadPoolExecutor(LOADERS)

    def name(self):
        return self.torch.cuda.get_device_name(0) if self.device == "cuda" else "cpu"

    def _to16k(self, samples, rate):
        x = self.torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)).to(self.device)
        if rate != SR:
            x = self.F.resample(x, rate, SR)
        rms = self.torch.sqrt(self.torch.mean((x * 32767) ** 2))
        change = -30 - 20 * self.torch.log10(rms / 32767 + 1e-12)
        return x * 10 ** (change / 20) if change > 0 else x         # louder only, to -30 dBFS

    def _trim(self, x, flags):
        """Resemblyzer's trim_long_silences, with the window flags given: moving average, rounding, dilation."""
        torch = self.torch
        window = 30 * SR // 1000
        x = x[:len(x) - len(x) % window]
        if len(flags) == 0:
            return x
        f = torch.from_numpy(flags).to(self.device).view(1, 1, -1)
        width = 8
        padded = torch.nn.functional.pad(f, ((width - 1) // 2, width // 2))
        mask = torch.round(torch.nn.functional.avg_pool1d(padded, width, stride=1))      # round half to even, as numpy
        mask = torch.nn.functional.max_pool1d(mask, 7, stride=1, padding=3).view(-1) > 0  # binary_dilation, 7 wide
        trimmed = x[mask.repeat_interleave(window)]
        return trimmed if len(trimmed) >= SR // 10 else x

    def _partials(self, x):
        """The 160-frame windows of one clip's mel spectrogram, as compute_partial_slices picks them."""
        hop, n = 160, 160
        frames = int(np.ceil((len(x) + 1) / hop))
        step = int(np.round((SR / 1.3) / hop))
        starts = list(range(0, max(1, frames - n + step + 1), step))
        last = starts[-1] * hop
        if len(starts) > 1 and (len(x) - last) / (n * hop) < 0.75:
            starts = starts[:-1]
        need = (starts[-1] + n) * hop
        if need >= len(x):
            x = self.torch.nn.functional.pad(x, (0, need - len(x)))
        mel = self.mel(x).T                                          # frames x 40
        return self.torch.stack([mel[s:s + n] for s in starts])

    def embed_decoded(self, decoded):
        """[(samples, rate)] -> (n x 256) unit embeddings."""
        torch = self.torch
        with torch.no_grad():
            wavs = [self._to16k(s, r) for s, r in decoded]
            flags = list(self.loaders.map(vad_flags, [w.cpu().numpy() for w in wavs]))
            parts = [self._partials(self._trim(w, f)) for w, f in zip(wavs, flags)]
            counts = [len(p) for p in parts]
            allp = torch.cat(parts)
            out = torch.cat([self.encoder(allp[i:i + PARTIALS_PER_PASS]) for i in range(0, len(allp), PARTIALS_PER_PASS)])
            embeds = torch.stack([e.mean(0) for e in torch.split(out, counts)])
            return torch.nn.functional.normalize(embeds, dim=1).cpu().numpy()

    def embed_files(self, paths):
        import soundfile as sf
        decoded = []
        for p in paths:
            data, rate = sf.read(str(p), dtype="float32")
            decoded.append((data.mean(axis=1) if data.ndim > 1 else data, rate))
        return self.embed_decoded(decoded)


# ---- the build -------------------------------------------------------------------------------

def load_clip_cache(path=CLIP_CACHE):
    if not Path(path).is_file():
        return {}
    saved = np.load(path, allow_pickle=False)
    return {k: (v, e) for k, v, e in zip(saved["keys"].tolist(), saved["voices"].tolist(), saved["embeddings"])}


def save_clip_cache(cache, path=CLIP_CACHE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted(cache)
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez(temporary, keys=np.array(keys), voices=np.array([cache[k][0] for k in keys]),
             embeddings=np.vstack([cache[k][1] for k in keys]).astype(np.float32) if keys else np.zeros((0, 256), np.float32))
    temporary.replace(path)


def embed_clips(clips, embedder, cache, deadline, budget_seconds, started, log=print):
    """Embed every clip not in cache, in file order; shrink the scope if the projection is over the budget."""
    todo = [c for c in clips if clip_key(c["dataset"], c["clip_id"]) not in cache]
    total_audio = sum(c["speech_seconds"] for c in todo)
    log(f"To embed: {len(todo)} clips ({total_audio / 3600:.1f} h of speech); {len(cache)} already done.")
    if not todo:
        return {"cap": None, "stopped_early": False}
    units = work_units(todo)
    cap, allowed, decided, stopped, done_audio, last_save = None, None, False, False, 0.0, time.time()
    work_start = time.time()
    pending = []

    def flush():
        nonlocal pending, done_audio
        if not pending:
            return
        embeds = embedder.embed_decoded([(s, r) for _, s, r in pending])
        for (c, _, _), e in zip(pending, embeds):
            cache[clip_key(c["dataset"], c["clip_id"])] = (voice_key(c["dataset"], c["speaker"]), e)
            done_audio += c["speech_seconds"]
        pending = []

    with ThreadPoolExecutor(4) as readers:
        i, inflight = 0, []
        while i < len(units) or inflight:
            while i < len(units) and len(inflight) < 16:
                inflight.append(readers.submit(decode_unit, units[i]))
                i += 1
            for item in inflight.pop(0).result():
                c = item[0]
                if allowed is None or clip_key(c["dataset"], c["clip_id"]) in allowed:
                    pending.append(item)
                if len(pending) >= BATCH_CLIPS:
                    flush()
            now = time.time()
            if not decided and now - work_start > PILOT_SECONDS and done_audio:
                decided = True
                rate = done_audio / (now - work_start)                      # seconds of speech per second
                projected = (now - started) + (total_audio - done_audio) / rate
                log(f"  pilot: {rate:.0f} s of speech per second; projected total {projected / 60:.0f} min "
                    f"(limit {budget_seconds / 60:.0f} min)")
                if projected > budget_seconds * 0.9:
                    left = budget_seconds * 0.85 - (now - started)
                    share = max(0.0, left * rate / max(1.0, total_audio - done_audio))
                    per_voice = defaultdict(int)
                    for c in clips:
                        per_voice[voice_key(c["dataset"], c["speaker"])] += 1
                    cap = max(1, int(np.median(list(per_voice.values())) * share))
                    kept = cap_per_voice(clips, cap, keep=cache.keys())
                    allowed = {clip_key(c["dataset"], c["clip_id"]) for c in kept}
                    total_audio = done_audio + sum(c["speech_seconds"] for c in kept
                                                   if clip_key(c["dataset"], c["clip_id"]) not in cache)
                    log(f"  over the limit: keeping at most {cap} clips per voice ({len(kept)} clips)")
            if now - last_save > SAVE_EVERY:
                flush()
                save_clip_cache(cache)
                last_save = now
                log(f"  {done_audio / 3600:.2f} of {total_audio / 3600:.2f} h embedded, {(now - started) / 60:.0f} min")
            if now > deadline:
                log("  time limit reached: stopping and using the clips embedded so far")
                stopped = True
                for f in inflight:
                    f.cancel()
                break
        flush()
    save_clip_cache(cache)
    return {"cap": cap, "stopped_early": stopped}


def voice_matrix(cache, clips, embedder_device="cuda"):
    """(voices, voice embeddings, similarity matrix), from the clip embeddings of the clips in scope."""
    import torch
    wanted = {clip_key(c["dataset"], c["clip_id"]) for c in clips}
    by_voice = defaultdict(list)
    for k, (voice, e) in cache.items():
        if k in wanted:
            by_voice[voice].append(e)
    voices = sorted(by_voice)
    device = embedder_device if torch.cuda.is_available() else "cpu"
    with torch.no_grad():
        emb = torch.nn.functional.normalize(torch.stack([torch.from_numpy(np.mean(by_voice[v], axis=0)) for v in voices])
                                            .to(device), dim=1)
        sim = (emb @ emb.T).clamp(-1, 1)
    return voices, {v: len(by_voice[v]) for v in voices}, emb.cpu().numpy(), sim.cpu().numpy().astype(np.float32)


def order_voices(voices, sim):
    """By dataset, and alike voices together within a dataset (average-linkage clustering, optimal leaf order)."""
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    out = []
    for dataset in sorted({v.split("/")[0] for v in voices}):
        idx = [i for i, v in enumerate(voices) if v.split("/")[0] == dataset]
        if len(idx) > 2:
            d = np.clip(1 - sim[np.ix_(idx, idx)], 0, None)
            np.fill_diagonal(d, 0)
            leaves = leaves_list(linkage(squareform(d, checks=False), "average", optimal_ordering=True))
            idx = [idx[i] for i in leaves]
        out += idx
    return out


def build(max_minutes=MAX_MINUTES, min_seconds=MIN_SECONDS, cap=None, fresh=False, limit=None, log=print):
    started = time.time()
    budget = max_minutes * 60
    clips = manifest_clips(min_seconds)
    if cap:
        clips = cap_per_voice(clips, cap)
    if limit:
        clips = cap_per_voice(clips, 10 ** 9)[:limit]
    log(f"Scope: {len(clips)} clips over {min_seconds:g} s, {len({voice_key(c['dataset'], c['speaker']) for c in clips})} "
        "voices (test voices left out: they are locked)")
    clips, missing = attach_sources(clips)
    if missing:
        log(f"  {len(missing)} clips not found in the voice library, left out (first: {missing[0]})")
    cache = {} if fresh else load_clip_cache()
    embedder = GpuEmbedder()
    log(f"Embedding on {embedder.name()}, limit {max_minutes:g} min")
    run = embed_clips(clips, embedder, cache, started + budget, budget, started, log)
    info = {"limit_minutes": max_minutes, "device": embedder.name(), "min_seconds": min_seconds, "source": "dataset clips",
            "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(), "clips_in_scope": len(clips),
            "cap_per_voice": run["cap"] or cap, "stopped_early": run["stopped_early"], "test_voices": "left out (locked)",
            "missing_clips": len(missing)}
    return finish(cache, clips, DATA_DIR, IMAGE_DIR / "heatmap.png", info, started, log)


def finish(cache, clips, folder, image, info, started, log=print):
    """The voice matrix from the clip embeddings, stored in `folder`, and the heatmap drawn to `image`."""
    voices, used, emb, sim = voice_matrix(cache, clips)
    order = order_voices(voices, sim)
    voices = [voices[i] for i in order]
    emb, sim = emb[order], sim[np.ix_(order, order)]
    seconds = defaultdict(float)
    for c in clips:
        if clip_key(c["dataset"], c["clip_id"]) in cache:
            seconds[voice_key(c["dataset"], c["speaker"])] += c["speech_seconds"]
    split = splits()
    rows = []
    for i, v in enumerate(voices):
        near = [j for j in np.argsort(-sim[i]) if j != i][:5]
        rows.append({"voice": v, "dataset": v.split("/")[0], "split": split.get(v, "unused"), "clips": used[v],
                     "seconds": round(seconds[v], 1), "closest": [[voices[j], round(float(sim[i, j]), 4)] for j in near]})
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "similarity.npy", sim)
    np.save(folder / "voice_embeddings.npy", emb.astype(np.float32))
    safe_files.write_json(folder / "voices.json", {"voices": rows})
    info = {"built": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "minutes": round((time.time() - started) / 60, 1),
            "voices": len(voices), "clips": sum(used.values()), **info}
    safe_files.write_json(folder / "build.json", info)
    image = draw_heatmap(voices, sim, info, image)
    log(f"Done in {info['minutes']} min: {len(voices)} voices, {info['clips']} clips.\n  data: {folder}\n  heatmap: {image}")
    return info


# ---- the same line, rendered in every voice --------------------------------------------------

def reference_clip(clips, split_clip=None):
    """The clip a voice is cloned from: its split clip if it has one, else its longest neutral clip of 3-25 s
    (the split's own rule), else its longest clip of up to 25 s in any emotion."""
    if split_clip:
        found = [c for c in clips if c["clip_id"] == split_clip]
        if found:
            return found[0]
    fits = [c for c in clips if 3 <= c["speech_seconds"] <= 25]
    neutral = [c for c in fits if c.get("emotion") in (None, "", "neutral")]
    pool = neutral or fits or clips
    return sorted(pool, key=lambda c: (-c["speech_seconds"], c["clip_id"]))[0]


def render_line(max_minutes=MAX_MINUTES, line=None, seed=0, log=print, render=None, folder=RENDER_DIR, reuse_only=False):
    """Render the same line in every voice of the dataset map (CosyVoice 3, plain, no genome), skipping what exists.

    Voices whose plain render of the same line, seed and reference clip already exists (the listening batch,
    genome_explore/mos-diverse-v1) are copied instead of rendered again.
    """
    import shutil

    from backend.genome import explore, search
    from backend.speech.tts_formatter import process_tts_string
    from backend.voice_data.split_voice_data import load_voice_manifest, read_library
    from backend.voice_data.Voice_Importer import get_clip
    started = time.time()
    line = line or explore.LINE
    folder = Path(folder)
    voices = load(DATA_DIR)["voices"]
    by_voice = defaultdict(list)
    for c in safe_files.read_json(MANIFEST)["clips"]:
        by_voice[voice_key(c["dataset"], c["speaker"])].append(c)
    split_clip = {v["voice"]: v["clip_id"] for v in load_voice_manifest()["voices"]}
    record_path = folder / "renders.json"
    record = (safe_files.read_json(record_path) if record_path.is_file() else None) or {"line": line, "seed": seed, "voices": {}}
    if record["line"] != line or record["seed"] != seed:
        raise SystemExit(f"{folder} holds renders of another line or seed: choose another --out folder.")
    existing = {}
    plan_path = explore.batch_dir() / "plan.json"
    if plan_path.is_file():
        plan = safe_files.read_json(plan_path)
        if plan.get("line") == line and plan.get("render_seed") == seed:
            existing = {v["voice"]: (v["clip_id"], explore.batch_dir() / "voices" / explore.safe(v["voice"]) / "plain.wav")
                        for v in plan["voices"]}
    todo = [v for v in voices if not (folder / f"{explore.safe(v)}.wav").is_file()]
    log(f"Rendering one line in {len(voices)} voices: {len(voices) - len(todo)} done, {len(todo)} to go.\n  line: {line}")
    library = None
    for i, voice in enumerate(todo, 1):
        if time.time() - started > max_minutes * 60:
            log(f"Time limit reached after {i - 1} renders; run it again to continue.")
            break
        ref = reference_clip(by_voice[voice], split_clip.get(voice))
        path = folder / f"{explore.safe(voice)}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        how = "rendered"
        if voice in existing and existing[voice][0] == ref["clip_id"] and existing[voice][1].is_file():
            shutil.copyfile(existing[voice][1], path)
            how = "copied from genome_explore/mos-diverse-v1"
        elif reuse_only:
            continue
        else:
            if library is None:
                library = {clip_key(c.dataset, c.clip_id): c for c in read_library()}
            audio, text = get_clip(library[clip_key(ref["dataset"], ref["clip_id"])])
            t = time.time()
            (render or search.render_plan)(audio, text, process_tts_string(line, []), path, seed)
            how = f"rendered in {time.time() - t:.1f} s"
        record["voices"][voice] = {"file": path.name, "reference_clip": ref["clip_id"], "emotion": ref.get("emotion"),
                                   "reference_seconds": ref["speech_seconds"], "how": how}
        safe_files.write_json(record_path, record)
        log(f"  {i}/{len(todo)} {voice}: {how}")
    log(f"Renders: {folder}")
    return record


def build_renders(folder=RENDER_DIR, log=print):
    """The voice map from one rendered line per voice, stored in history/voice_similarity/rendered_line/."""
    import soundfile as sf
    started = time.time()
    folder = Path(folder)
    record = safe_files.read_json(folder / "renders.json") if (folder / "renders.json").is_file() else None
    if not record or not record["voices"]:
        raise SystemExit(f"No renders in {folder}: run  python main_backend.py voice-similarity render")
    clips = []
    for voice, r in sorted(record["voices"].items()):
        path = folder / r["file"]
        if path.is_file():
            dataset, speaker = voice.split("/", 1)
            clips.append({"dataset": dataset, "speaker": speaker, "clip_id": f"{speaker}__rendered_line",
                          "speech_seconds": sf.info(str(path)).duration, "source": str(path), "row_group": None, "row": None})
    embedder = GpuEmbedder()
    log(f"Embedding {len(clips)} renders on {embedder.name()}")
    cache = {}
    embeds = embedder.embed_files([c["source"] for c in clips])
    for c, e in zip(clips, embeds):
        cache[clip_key(c["dataset"], c["clip_id"])] = (voice_key(c["dataset"], c["speaker"]), e)
    info = {"device": embedder.name(), "source": "one rendered line per voice", "line": record["line"],
            "seed": record["seed"], "renders": str(folder), "test_voices": "left out (locked)"}
    info = finish(cache, clips, RENDERED_DATA, IMAGE_DIR / "heatmap_rendered_line.png", info, started, log)
    comparison = compare_maps()
    if comparison:
        safe_files.write_json(RENDERED_DATA / "compare_with_dataset_clips.json", comparison)
        log(f"Against the dataset-clip map ({comparison['voices']} voices): pair similarities correlate "
            f"{comparison['spearman_pairs']:.2f} (Spearman); a render is closest to its own voice in "
            f"{comparison['own_voice_first']} of {comparison['voices']}; render against its own voice: mean "
            f"{comparison['render_vs_own_mean']:.3f}, lowest {comparison['render_vs_own_lowest']:.3f} "
            f"({comparison['render_vs_own_lowest_voice']})")
    return info


def compare_maps(dataset_folder=DATA_DIR, rendered_folder=RENDERED_DATA):
    """How the rendered-line map agrees with the dataset-clip map, on the voices both have."""
    from scipy.stats import spearmanr
    if not (Path(dataset_folder) / "voices.json").is_file():
        return None
    a, b = load(dataset_folder), load(rendered_folder)
    shared = [v for v in b["voices"] if v in set(a["voices"])]
    ia = [a["voices"].index(v) for v in shared]
    ib = [b["voices"].index(v) for v in shared]
    upper = np.triu_indices(len(shared), 1)
    pairs_a = a["similarity"][np.ix_(ia, ia)][upper]
    pairs_b = b["similarity"][np.ix_(ib, ib)][upper]
    own = b["embeddings"][ib] @ a["embeddings"].T                     # each render against every dataset voice
    own_sim = own[np.arange(len(shared)), ia]
    first = int(sum(int(np.argmax(own[k])) == ia[k] for k in range(len(shared))))
    lowest = int(np.argmin(own_sim))
    return {"voices": len(shared), "spearman_pairs": round(float(spearmanr(pairs_a, pairs_b)[0]), 4) if len(shared) > 2 else None,
            "own_voice_first": first, "render_vs_own_mean": round(float(own_sim.mean()), 4),
            "render_vs_own_lowest": round(float(own_sim[lowest]), 4), "render_vs_own_lowest_voice": shared[lowest],
            "render_vs_own": {v: round(float(s), 4) for v, s in zip(shared, own_sim)}}


# ---- the heatmap -----------------------------------------------------------------------------

def draw_heatmap(voices=None, sim=None, info=None, path=None, folder=DATA_DIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if voices is None:
        data = load(folder)
        voices, sim, info = data["voices"], data["similarity"], safe_files.read_json(Path(folder) / "build.json")
        path = path or IMAGE_DIR / ("heatmap_rendered_line.png" if Path(folder) == RENDERED_DATA else "heatmap.png")
    n = len(voices)
    off = sim[~np.eye(n, dtype=bool)] if n > 1 else sim.ravel()
    low = float(np.floor(np.percentile(off, 2) * 20) / 20)
    size = min(30, max(8, 0.085 * n + 3))
    fig, ax = plt.subplots(figsize=(size + 1.6, size))
    im = ax.imshow(sim, cmap="Blues", vmin=low, vmax=1.0, interpolation="nearest")
    font = max(2.5, min(8, 520 / max(n, 1)))
    names = [v.split("/", 1)[1] for v in voices]
    ax.set_xticks(range(n), names, rotation=90, fontsize=font)
    ax.set_yticks(range(n), names, fontsize=font)
    ax.tick_params(length=0)
    datasets = [v.split("/")[0] for v in voices]
    start = 0
    for i in range(1, n + 1):
        if i == n or datasets[i] != datasets[start]:
            if i < n:
                ax.axhline(i - 0.5, color="#1d1f22", linewidth=1.2)
                ax.axvline(i - 0.5, color="#1d1f22", linewidth=1.2)
            ax.text((start + i - 1) / 2, -0.012 * n - 1, f"{datasets[start]} ({i - start})", ha="center", va="bottom",
                    fontsize=11, color="#1d1f22", fontweight="bold")
            start = i
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    bar.set_label("Resemblyzer cosine similarity (0.80+: sounds like the same person; under 0.70: different people)",
                  fontsize=10, color="#6b7078")
    for level in (ALIKE, SAME_PERSON):
        if level > low:
            bar.ax.axhline(level, color="#c0392b", linewidth=1.2)
    if info:
        what = (f"each saying the same rendered line (CosyVoice 3, plain, seed {info.get('seed')})"
                if info.get("source") == "one rendered line per voice"
                else f"{info['clips']} clips over {info.get('min_seconds', MIN_SECONDS):g} s of speech")
        ax.set_title(f"{n} voices, {what}, built {info['built']} (test voices left out)", fontsize=11, color="#6b7078",
                     pad=34, loc="left")
    path = Path(path or IMAGE_DIR / "heatmap.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---- using it: the closest voice with a genome -----------------------------------------------

def load(folder=DATA_DIR):
    folder = Path(folder)
    if not (folder / "voices.json").is_file():
        raise SystemExit(f"No voice similarity data yet in {folder}: run  python main_backend.py voice-similarity build")
    rows = safe_files.read_json(folder / "voices.json")["voices"]
    return {"voices": [r["voice"] for r in rows], "rows": rows,
            "embeddings": np.load(folder / "voice_embeddings.npy"), "similarity": np.load(folder / "similarity.npy")}


def genomes_by_voice(runs_dir=None):
    """{voice: the best genome found for it}: rated search runs (best by MOS) and genomes named from a search."""
    from backend.genome import ratings, search
    best = {}

    def better(a, b):
        key = lambda g: ((g.get("above_plain") or -99) > 0, g.get("above_plain") or -99, g.get("mos") or 0)  # noqa: E731
        return a if b is None or key(a) > key(b) else b

    for folder, run in search.list_runs(runs_dir):
        try:
            if not ratings.is_rated(folder):
                continue
            ranked = ratings.ranked(folder, run)
        except (OSError, ValueError, KeyError):
            continue
        rated = [g for g in ranked["genomes"] if g["mos"]]
        if not rated:
            continue
        g = rated[0]
        found = {"voice": run["voice"], "run": run["id"], "genome": g["number"], "name": g.get("picked"),
                 "mos": g["mos"]["mos"], "above_plain": g["above_baseline"], "described": g.get("described"),
                 "genes": g.get("genes"), "markup": g.get("markup"), "line": run.get("text"),
                 "raters": ranked["raters"], "rated_genomes": len(rated)}
        best[run["voice"]] = better(found, best.get(run["voice"]))
    return best


def closest(audio, top=5, folder=DATA_DIR, runs_dir=None, embedder=None):
    """The voices closest to a recording, and the genome of the closest voice that has one."""
    data = load(folder)
    embedder = embedder or GpuEmbedder()
    e = embedder.embed_files([audio])[0]
    sims = data["embeddings"] @ e
    order = np.argsort(-sims)
    genomes = genomes_by_voice(runs_dir)
    nearest = [{"voice": data["voices"][i], "similarity": round(float(sims[i]), 4),
                "has_genome": data["voices"][i] in genomes} for i in order[:top]]
    with_genome = next(({"voice": data["voices"][i], "similarity": round(float(sims[i]), 4)}
                        for i in order if data["voices"][i] in genomes), None)
    suggestion = None
    if with_genome:
        s = with_genome["similarity"]
        suggestion = {**with_genome, "how_close": "sounds like the same person" if s >= SAME_PERSON else
                      "alike" if s >= ALIKE else "a different person: treat the suggestion as a weak start",
                      "genome": genomes[with_genome["voice"]]}
    missing = sorted(set(genomes) - set(data["voices"]))
    return {"audio": str(audio), "closest": nearest, "suggestion": suggestion,
            "voices_with_genomes": len(genomes), "genome_voices_not_in_map": missing}


def show_closest(result):
    print(f"Closest voices to {result['audio']}:")
    for r in result["closest"]:
        print(f"  {r['similarity']:.3f}  {r['voice']}" + ("   (has a genome)" if r["has_genome"] else ""))
    s = result["suggestion"]
    if not s:
        print("No voice in the map has a rated genome yet.")
        return
    g = s["genome"]
    print(f"\nClosest voice with a genome: {s['voice']} ({s['similarity']:.3f}, {s['how_close']})")
    print(f"Suggested genome: genome {g['genome']} of run {g['run']}" + (f", named {g['name']!r}" if g["name"] else ""))
    gain = g["above_plain"]
    print(f"  MOS {g['mos']:.2f}" + (f", {gain:+.2f} against its plain line" if gain is not None else "")
          + f" ({g['rated_genomes']} genomes rated by {', '.join(g['raters'])})")
    if gain is not None and gain <= 0:
        print("  Note: no genome of that voice beat its plain line, so this is the least bad one.")
    for d in g.get("described") or []:
        print(f"  - {d}")
    if g.get("markup"):
        print(f"  markup: {g['markup']}")
    if result["genome_voices_not_in_map"]:
        print(f"\n{len(result['genome_voices_not_in_map'])} voices with genomes are not in the map (no clip over 3 s): "
              + ", ".join(result["genome_voices_not_in_map"]))


def check(clips=20):
    """The GPU path against Resemblyzer's own CPU code on a few clips: cosine of the two embeddings."""
    import librosa
    from resemblyzer import preprocess_wav
    sample, _ = attach_sources(cap_per_voice(manifest_clips(), 1)[:clips])
    embedder = GpuEmbedder()
    decoded = [item for unit in work_units(sample) for item in decode_unit(unit)]
    gpu = embedder.embed_decoded([(s, r) for _, s, r in decoded])
    cos = []
    for (c, s, r), g in zip(decoded, gpu):
        ref = embedder.encoder.embed_utterance(preprocess_wav(librosa.resample(s, orig_sr=r, target_sr=SR)
                                                              if r != SR else s))
        cos.append(float(np.dot(ref, g)))
    print(f"GPU against Resemblyzer's code on {len(cos)} clips: cosine mean {np.mean(cos):.5f}, lowest {min(cos):.5f}")
    return cos


def main(argv=None):
    parser = argparse.ArgumentParser(description="Every voice against every other (Resemblyzer, GPU), and the closest "
                                                 "voice with a genome. Guide: backend/voice_data/VOICE_SIMILARITY.md")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="embed the clips, compare every voice, draw the heatmap")
    b.add_argument("--max-minutes", type=float, default=MAX_MINUTES, help="time limit (default 120)")
    b.add_argument("--min-seconds", type=float, default=MIN_SECONDS, help="only clips with more speech than this (3)")
    b.add_argument("--max-clips-per-voice", type=int, help="fewer clips per voice from the start")
    b.add_argument("--fresh", action="store_true", help="embed every clip again (ignore the saved clip embeddings)")
    b.add_argument("--limit", type=int, help="only this many clips (a quick trial)")
    b.add_argument("--rendered", action="store_true", help="build the map from the rendered line instead (see render)")
    r = sub.add_parser("render", help="render the same line in every voice of the map (CosyVoice 3, GPU)")
    r.add_argument("--max-minutes", type=float, default=MAX_MINUTES, help="time limit (default 120); run again to continue")
    r.add_argument("--line", help="the line (default: the listening batch's line, so its 24 renders are reused)")
    r.add_argument("--out", default=str(RENDER_DIR), help=f"folder for the renders (default {RENDER_DIR})")
    r.add_argument("--reuse-only", action="store_true", help="only copy the renders that already exist; render nothing")
    h = sub.add_parser("heatmap", help="draw the heatmap again from the stored data")
    h.add_argument("--map", choices=MAPS, default="dataset", help="dataset clips (default) or the rendered line")
    c = sub.add_parser("closest", help="the closest voices to a recording, and the genome to suggest")
    c.add_argument("audio", help="a wav, flac or mp3 file of the voice")
    c.add_argument("--top", type=int, default=5)
    c.add_argument("--map", choices=MAPS, default="dataset",
                   help="compare with the dataset clips (default) or with the rendered line")
    c.add_argument("--json", action="store_true", help="print the result as JSON")
    k = sub.add_parser("check", help="compare the GPU path with Resemblyzer's own code")
    k.add_argument("--clips", type=int, default=20)
    args = parser.parse_args(argv)
    if args.command == "build":
        if args.rendered:
            build_renders()
        else:
            build(args.max_minutes, args.min_seconds, args.max_clips_per_voice, args.fresh, args.limit)
    elif args.command == "render":
        render_line(args.max_minutes, args.line, folder=args.out, reuse_only=args.reuse_only)
    elif args.command == "heatmap":
        print(draw_heatmap(folder=MAPS[args.map]))
    elif args.command == "closest":
        result = closest(args.audio, args.top, folder=MAPS[args.map])
        print(json.dumps(result, indent=2, ensure_ascii=False)) if args.json else show_closest(result)
    else:
        check(args.clips)


if __name__ == "__main__":
    main()

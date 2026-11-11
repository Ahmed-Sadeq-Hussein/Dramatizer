"""voice_map.py | Which validation voices sound alike, independent of the dataset they were recorded in?

    python main_backend.py voice-map                 # embed, cluster, draw (embeddings are cached)
    python main_backend.py voice-map --pick 18       # also suggest 18 voices that cover the map
    python main_backend.py voice-map --dedupe        # or: one voice per tight cluster plus every loner

Why not the plain similarity: Resemblyzer's similarity of one reference clip
groups voices by dataset, not by voice. 86 % of a voice's five nearest voices
come from its own dataset (39 % by chance): the recording (studio, actor booth,
audiobook) dominates. Cleaning or equalising the audio does not fix it, and
CosyVoice's own speaker model (CAM++) shows the same. The fix that works is
statistical, tested on 9 Nov 2026 against 70+ configurations (LIMITATIONS.md,
"The voice map"):

    every clip      a voice is the mean of the unit embeddings of its reference
                    clip and up to EXTRA_CLIPS other neutral clips (a steadier
                    voice than one clip)
    center          each dataset's mean embedding is removed
    s-norm          each similarity is z-normalised by the mean and spread of
                    its dataset pair (VCTK~VCTK, VCTK~CREMA-D, ...). The order of
                    voices WITHIN a dataset is unchanged; only scores across
                    datasets are made comparable

The two best ways, both shown on the page:

    Way 1  CosyVoice's ear: CAM++ (the speaker model CosyVoice clones from).
           Nested spectral clusters: 2 families -> 4 groups -> 8 neighbourhoods.
           3D layout: metric MDS (overall distances kept)
    Way 2  Two judges: CAM++ and Resemblyzer averaged (z-scored). Graph
           communities (Louvain on the 12-nearest-voice graph): the data picks
           how many. 3D layout: t-SNE (nearest voices kept)

What the scores mean (both ways, held-out clips): a voice's nearest voices come
from its own dataset about as often as chance; a clip not used for the map
finds its own voice first among all voices 93-95 % of the time, 99-100 % within
its dataset. Voices form a continuum: only the 2 families are near certain
(stability 0.98); groups and neighbourhoods are browsing aids, not types.

    --pick N    N voices that cover the map (Way 1 similarity): greedy, each
                step adds the voice that most raises everyone's similarity to
                their closest pick, with an equal share per dataset

    --dedupe    fewer, more different voices (Way 2 similarity, both judges, as
                they agree best on which pairs are close: 54 % of the 10 % most
                alike pairs are shared, chance 10 %): close = among the
                CLOSE_SHARE most alike pairs; loners = the LONER_SHARE of voices
                whose closest voice is least alike, always kept; then greedily
                the voice with the most close voices left is kept and those are
                dropped. The count follows from the data. No two voices are as
                alike as two halves of one voice's clips (split-half z median
                4.0; most alike pair of different voices 3.4), so "tight" means
                relatively alike, not duplicates

Test voices are never read: they stay locked until the final test.

Output (OUTPUT_DIR/voice_map/, never in the repository): embeddings.npz and
pitch.json (cached, remade when the voice manifest changes), voice_map.json and
voice_map.html and audio/ (the reference clips, so the page plays them; click a voice to hear it).
"""
import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from backend import model

OUT_DIR = model.OUTPUT_DIR / "voice_map"
SR = 16000
EXTRA_CLIPS = 4              # other neutral clips per voice, besides the reference
MIN_EXTRA_SECONDS = 1.5
NEIGHBOURS = 12              # the voice graph of Way 2
LOUVAIN_RESOLUTION = 0.8
LEVELS = ("family", "group", "neighbourhood")
SEED = 0
RANDOM_PICKS = 300
CLOSE_SHARE = 0.10           # --dedupe: two voices are "close" if among the 10 % most alike pairs
LONER_SHARE = 0.05           # --dedupe: the 5 % of voices whose closest voice is least alike are always kept
PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"


# --- The voices, their clips, embeddings and pitch ----------------------------

def _load(clip):
    """A clip as 16 kHz mono float32, read in memory (nothing is written next to the datasets)."""
    import librosa
    import pyarrow.parquet as pq
    import soundfile as sf

    if clip.row_group is not None:
        table = pq.ParquetFile(clip.source).read_row_group(clip.row_group, columns=["audio"])
        data, rate = sf.read(io.BytesIO(table.column("audio")[clip.row].as_py()["bytes"]), dtype="float32")
    else:
        data, rate = sf.read(str(clip.source), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return librosa.resample(data, orig_sr=rate, target_sr=SR) if rate != SR else data


def validation_voices():
    """[{voice, dataset, speaker, clip_id, audio, seconds, extra: [clip]}] for every validation voice, in manifest order."""
    from backend.voice_data.Voice_Importer import get_clip
    from backend.voice_data.split_voice_data import (group_by_speaker, load_voice_manifest, read_library,
                                                     usable_clips, voice_clips)

    seconds = {v["voice"]: v["speech_seconds"] for v in load_voice_manifest()["voices"]}
    groups = group_by_speaker(read_library())
    voices = []
    for voice, clip_id in voice_clips("validation").items():
        dataset = voice.split("/")[0]
        neutral = [c for c in usable_clips(dataset, groups[voice]) if c.details.get("emotion") in (None, "", "neutral")]
        ref = next(c for c in neutral if c.clip_id == clip_id)
        others = sorted((c for c in neutral if c.clip_id != clip_id), key=lambda c: c.clip_id)
        # The same extra clips every run: a fixed shuffle seeded by the voice's name
        seed = int(hashlib.sha256(voice.encode()).hexdigest()[:8], 16)
        others = [others[i] for i in np.random.default_rng(seed).permutation(len(others))]
        audio, _ = get_clip(ref)
        voices.append({"voice": voice, "dataset": dataset, "speaker": ref.speaker, "clip_id": clip_id,
                       "audio": str(audio), "seconds": seconds[voice], "_ref": ref, "_others": others})
    return voices


def manifest_sha256():
    from backend.voice_data.split_voice_data import SPLITS_FILE
    return hashlib.sha256(Path(SPLITS_FILE).read_bytes()).hexdigest()


_campplus = None


def campplus_embedding(wav):
    """CosyVoice 3's own speaker embedding (campplus.onnx, as cosyvoice/cli/frontend.py computes it)."""
    global _campplus
    import onnxruntime as ort
    import torch
    import torchaudio.compliance.kaldi as kaldi

    if _campplus is None:
        path = Path(model.setting("COSYVOICE_MODEL_DIR")) / "campplus.onnx"
        _campplus = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    feat = kaldi.fbank(torch.from_numpy(np.asarray(wav, np.float32)).unsqueeze(0), num_mel_bins=80, dither=0,
                       sample_frequency=SR)
    feat = feat - feat.mean(dim=0, keepdim=True)
    return _campplus.run(None, {_campplus.get_inputs()[0].name: feat.unsqueeze(0).numpy()})[0].flatten()


def resemblyzer_embedding(wav):
    """The project's similarity judge (backend/metrics/Similarity.py)."""
    from resemblyzer import preprocess_wav

    trimmed = preprocess_wav(np.asarray(wav, np.float32), source_sr=SR)
    return model.speaker_encoder().embed_utterance(trimmed if len(trimmed) >= SR // 10 else wav)


EMBEDDERS = {"campplus": campplus_embedding, "resemblyzer": resemblyzer_embedding}


def unit(x):
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def embeddings_of(voices, out_dir=OUT_DIR):
    """
    {"campplus": (voices x 192), "resemblyzer": (voices x 256), "reference_resemblyzer": ...}: each voice the mean
    of its clips' unit embeddings. Cached by manifest checksum and the clips used.
    """
    cache = Path(out_dir) / "embeddings.npz"
    key = hashlib.sha256((manifest_sha256() + json.dumps([v["voice"] for v in voices])).encode()).hexdigest()
    if cache.is_file():
        saved = np.load(cache, allow_pickle=False)
        if "key" in saved.files and str(saved["key"]) == key:
            return {k: saved[k] for k in saved.files if k not in ("key", "clips")}, json.loads(str(saved["clips"]))

    rows = {name: [] for name in EMBEDDERS}
    reference = []
    clips_used = {}
    for i, v in enumerate(voices, 1):
        wavs, ids = [_load(v["_ref"])], [v["clip_id"]]
        for clip in v["_others"]:
            if len(ids) > EXTRA_CLIPS:
                break
            wav = _load(clip)
            if len(wav) >= MIN_EXTRA_SECONDS * SR:
                wavs.append(wav)
                ids.append(clip.clip_id)
        clips_used[v["voice"]] = ids
        for name, embed in EMBEDDERS.items():
            each = [unit(embed(w)) for w in wavs]
            rows[name].append(unit(np.mean(each, axis=0)))
            if name == "resemblyzer":
                reference.append(each[0])
        print(f"  embedded {i}/{len(voices)} {v['voice']} ({len(wavs)} clips)", flush=True)
    result = {name: np.vstack(r) for name, r in rows.items()}
    result["reference_resemblyzer"] = np.vstack(reference)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, key=np.array(key), clips=np.array(json.dumps(clips_used)), **result)
    return result, clips_used


def median_pitch(path):
    """Median F0 (Hz) of the voiced, louder frames of a recording."""
    import librosa

    from backend.audio_io import load_mono

    wav = load_mono(path, SR)
    f0 = librosa.yin(wav, fmin=60, fmax=400, sr=SR, frame_length=1024)
    rms = librosa.feature.rms(y=wav, frame_length=1024, hop_length=256)[0][:len(f0)]
    voiced = f0[:len(rms)][rms > 0.3 * np.median(rms[rms > 0])]
    return float(np.median(voiced)) if len(voiced) else None


def pitches_of(voices, out_dir=OUT_DIR):
    """{voice: median F0 of its reference clip in Hz}, cached by manifest checksum."""
    cache = Path(out_dir) / "pitch.json"
    sha = manifest_sha256()
    saved = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else {}
    if saved.get("manifest_sha256") != sha:
        saved = {"manifest_sha256": sha, "pitch": {}}
    for v in voices:
        if v["voice"] not in saved["pitch"]:
            saved["pitch"][v["voice"]] = median_pitch(v["audio"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(saved, indent=1), encoding="utf-8")
    return saved["pitch"]


# --- Similarity without the recording (pure functions, tested in test_voice_map.py)

def similarity_matrix(emb):
    """Cosine similarity of every pair, as speaker_similarity() computes it."""
    u = unit(emb)
    return np.clip(u @ u.T, -1.0, 1.0)


def remove_recording(emb, datasets):
    """The embeddings with each dataset's mean removed, unit length again."""
    datasets = np.asarray(datasets)
    out = unit(emb).copy()
    for d in set(datasets):
        out[datasets == d] -= out[datasets == d].mean(axis=0)
    return unit(out)


def snorm(sim, datasets):
    """Each similarity as z-score within its dataset pair: order within a dataset kept, datasets made comparable."""
    datasets = np.asarray(datasets)
    out = np.array(sim, dtype=np.float64)
    names = sorted(set(datasets))
    for a in names:
        for b in names:
            block = sim[np.ix_(datasets == a, datasets == b)]
            values = block[~np.eye(*block.shape, dtype=bool)] if a == b else block.ravel()
            out[np.ix_(datasets == a, datasets == b)] = (block - values.mean()) / (values.std() + 1e-9)
    return (out + out.T) / 2


def zscore(sim):
    off = sim[~np.eye(len(sim), dtype=bool)]
    return (sim - off.mean()) / off.std()


def voice_similarity(emb, datasets):
    """Center per dataset, cosine, s-norm: the similarity both ways use."""
    return snorm(similarity_matrix(remove_recording(emb, datasets)), datasets)


def distances(sim):
    """Similarity scaled to distances from 0 (most alike pair) to 1 (least alike), 0 on the diagonal."""
    off = sim[~np.eye(len(sim), dtype=bool)]
    d = (off.max() - sim) / (off.max() - off.min())
    np.fill_diagonal(d, 0.0)
    return d


def neighbour_graph(sim, k=NEIGHBOURS):
    """Each voice linked to its k most similar voices, weighted by exp(z-scored similarity)."""
    z = zscore(sim)
    np.fill_diagonal(z, -np.inf)
    graph = np.zeros_like(z)
    for i in range(len(z)):
        nearest = np.argsort(-z[i])[:k]
        graph[i, nearest] = np.exp(z[i, nearest])
    return np.maximum(graph, graph.T)


def _spectral_two(sim):
    from sklearn.cluster import SpectralClustering

    if len(sim) < 8:
        return np.zeros(len(sim), int)
    k = min(NEIGHBOURS, len(sim) - 1)
    return SpectralClustering(n_clusters=2, affinity="precomputed", random_state=SEED).fit_predict(neighbour_graph(sim, k))


def nested_clusters(sim, levels=3):
    """Way 1: split every group in two, `levels` times (2 -> 4 -> 8). Returns one label array per level."""
    out, current = [], np.zeros(len(sim), int)
    for _ in range(levels):
        new, next_label = np.zeros(len(sim), int), 0
        for c in sorted(set(current)):
            members = np.flatnonzero(current == c)
            halves = _spectral_two(sim[np.ix_(members, members)])
            for h in sorted(set(halves)):
                new[members[halves == h]] = next_label
                next_label += 1
        out.append(_by_size(new))
        current = new
    return out


def communities(sim, resolution=LOUVAIN_RESOLUTION):
    """Way 2: Louvain communities of the nearest-voice graph; the data decides how many."""
    import networkx as nx

    graph = nx.from_numpy_array(neighbour_graph(sim))
    found = nx.community.louvain_communities(graph, weight="weight", resolution=resolution, seed=SEED)
    labels = np.zeros(len(sim), int)
    for c, members in enumerate(found):
        labels[list(members)] = c
    return _by_size(labels)


def _by_size(labels):
    order = sorted(set(labels), key=lambda c: (-np.sum(labels == c), c))
    renumber = {old: new for new, old in enumerate(order)}
    return np.array([renumber[c] for c in labels])


def stability(cluster_fn, sim, draws=15, share=0.8, seed=SEED):
    """Mean adjusted Rand index between the clusters of all voices and of random 80 % subsets."""
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)
    full = cluster_fn(sim)
    scores = []
    for _ in range(draws):
        sub = np.sort(rng.choice(len(sim), int(share * len(sim)), replace=False))
        scores.append(adjusted_rand_score(full[sub], cluster_fn(sim[np.ix_(sub, sub)])))
    return float(np.mean(scores))


def layout_3d(sim, method):
    """3D points: "mds" keeps overall distances, "tsne" keeps nearest voices. Returns (points, Pearson r, trustworthiness)."""
    from sklearn.manifold import MDS, TSNE, trustworthiness

    d = distances(sim)
    if method == "mds":
        points = MDS(n_components=3, dissimilarity="precomputed", random_state=SEED, n_init=4,
                     normalized_stress="auto").fit_transform(d)
    else:
        points = TSNE(n_components=3, metric="precomputed", init="random", perplexity=20,
                      random_state=SEED).fit_transform(d)
    upper = np.triu_indices(len(sim), 1)
    mapped = np.linalg.norm(points[:, None] - points[None], axis=2)[upper]
    return points, float(np.corrcoef(mapped, d[upper])[0, 1]), float(trustworthiness(d, points, n_neighbors=10,
                                                                                     metric="precomputed"))


def same_dataset_share(sim, datasets, k=5):
    """How often a voice's k nearest voices come from its own dataset, and the chance level."""
    datasets = np.asarray(datasets)
    s = np.array(sim, dtype=np.float64)
    np.fill_diagonal(s, -np.inf)
    nearest = np.argsort(-s, axis=1)[:, :k]
    n = len(datasets)
    chance = sum((datasets == d).mean() * ((datasets == d).sum() - 1) / (n - 1) for d in set(datasets))
    return float(np.mean(datasets[nearest] == datasets[:, None])), float(chance)


def medoids(sim, labels):
    """{cluster: index of its member with the highest mean similarity to the rest of the cluster}."""
    out = {}
    for c in sorted(set(labels)):
        members = np.flatnonzero(labels == c)
        inner = sim[np.ix_(members, members)]
        mean = (inner.sum(axis=1) - np.diag(inner)) / max(len(members) - 1, 1)
        out[int(c)] = int(members[np.argmax(mean)])
    return out


def quotas(datasets, n):
    """N split as evenly as possible over the datasets (never more than a dataset has); leftovers to the largest."""
    names, counts = np.unique(np.asarray(datasets), return_counts=True)
    size = dict(zip(names, counts))
    share = {d: 0 for d in names}
    while sum(share.values()) < min(n, len(datasets)):
        open_ = [d for d in names if share[d] < size[d]]
        d = min(open_, key=lambda d: (share[d], -size[d], d))
        share[d] += 1
    return share


def spread_pick(sim, datasets, n):
    """Greedy coverage: each step adds the voice that most raises the mean similarity to the closest pick, per dataset share."""
    datasets = np.asarray(datasets)
    share = quotas(datasets, n)
    picks, closest = [], np.full(len(sim), -np.inf)
    while len(picks) < sum(share.values()):
        taken = {d: sum(datasets[p] == d for p in picks) for d in share}
        best, best_gain = None, -np.inf
        for i in range(len(sim)):
            if i in picks or taken[datasets[i]] >= share[datasets[i]]:
                continue
            gain = np.maximum(closest, sim[:, i]).mean()
            if gain > best_gain:
                best, best_gain = i, gain
        picks.append(best)
        closest = np.maximum(closest, sim[:, best])
    return picks


def dedupe_pick(sim, close_share=CLOSE_SHARE, loner_share=LONER_SHARE):
    """
    One voice per tight cluster, plus every loner. Two voices are "close" when their similarity is among the
    `close_share` most alike pairs. Loners (the `loner_share` of voices whose closest voice is least alike) are
    kept first, most alone first, each dropping the few voices close to it (so an isolated pair keeps one).
    Then, greedily, the voice with the most close voices still uncovered is kept and those voices are dropped,
    until every voice is kept or close to a kept one.
    Returns (picks, loners, {pick: [voices it stands for, itself first]}).
    """
    sim = np.asarray(sim, dtype=np.float64)
    n = len(sim)
    off = ~np.eye(n, dtype=bool)
    close = (sim >= np.quantile(sim[off], 1 - close_share)) & off
    nearest = np.where(off, sim, -np.inf).max(axis=1)
    picks, left = [], set(range(n))
    for i in np.argsort(nearest)[:int(round(loner_share * n))]:     # most alone first
        if int(i) in left:                  # an isolated pair of loners keeps one of the two
            picks.append(int(i))
            left -= {int(i)} | set(np.flatnonzero(close[i]).tolist())
    loners = sorted(picks)
    while left:
        rest = sorted(left)
        best = max(rest, key=lambda i: (int(close[i, rest].sum()), sim[i, rest].mean(), -i))
        picks.append(best)
        left -= {best} | set(np.flatnonzero(close[best]).tolist())
    stands_for = {p: [p] for p in picks}
    for i in set(range(n)) - set(picks):
        stands_for[max(picks, key=lambda p: sim[i, p])].append(i)
    return picks, loners, stands_for


def coverage(sim, picks):
    """How alike every voice is to its closest pick: mean and worst."""
    closest = np.max(np.asarray(sim)[:, picks], axis=1)
    return {"mean": float(closest.mean()), "worst": float(closest.min())}


def random_coverage(sim, datasets, n, draws=RANDOM_PICKS, seed=SEED):
    """Mean coverage of random picks with the same per-dataset shares, for comparison."""
    datasets = np.asarray(datasets)
    rng = np.random.default_rng(seed)
    share = quotas(datasets, n)
    scores = []
    for _ in range(draws):
        picks = [i for d, k in share.items() for i in rng.choice(np.flatnonzero(datasets == d), k, replace=False)]
        scores.append(coverage(sim, picks)["mean"])
    return float(np.mean(scores))


# --- Report --------------------------------------------------------------------

def cluster_table(voices, sim, labels, pitch):
    meds = medoids(sim, labels)
    rows = []
    for c in sorted(set(labels)):
        members = np.flatnonzero(labels == c)
        mix = {}
        for i in members:
            mix[voices[i]["dataset"]] = mix.get(voices[i]["dataset"], 0) + 1
        hz = [pitch[voices[i]["voice"]] for i in members if pitch.get(voices[i]["voice"])]
        rows.append({"cluster": int(c) + 1, "size": int(len(members)), "datasets": mix,
                     "pitch_hz": {"median": round(float(np.median(hz))), "low": round(float(np.percentile(hz, 10))),
                                  "high": round(float(np.percentile(hz, 90)))} if hz else None,
                     "medoid": voices[meds[int(c)]]["voice"],
                     "members": [voices[i]["voice"] for i in sorted(members, key=lambda i: -sim[i, meds[int(c)]])]})
    return rows


def build(pick=None, dedupe=False, out_dir=OUT_DIR):
    voices = validation_voices()
    datasets = np.array([v["dataset"] for v in voices])
    emb, clips_used = embeddings_of(voices, out_dir)
    pitch = pitches_of(voices, out_dir)

    before = similarity_matrix(emb["reference_resemblyzer"])
    way1 = voice_similarity(emb["campplus"], datasets)
    way2 = (zscore(way1) + zscore(voice_similarity(emb["resemblyzer"], datasets))) / 2

    ways = {}
    for key, sim, layout_default, title in (
            ("way1", way1, "mds", "Way 1: CosyVoice's ear"), ("way2", way2, "tsne", "Way 2: Two judges agree")):
        if key == "way1":
            levels = nested_clusters(sim)
            stab = [stability(lambda s, lv=lv: nested_clusters(s)[lv], sim) for lv in range(3)]
            clusterings = {LEVELS[lv]: (levels[lv], stab[lv]) for lv in range(3)}
        else:
            labels = communities(sim)
            clusterings = {"community": (labels, stability(communities, sim))}
        layouts = {m: layout_3d(sim, m) for m in ("mds", "tsne")}
        share, chance = same_dataset_share(sim, datasets)
        ways[key] = {
            "title": title, "default_layout": layout_default,
            "same_dataset_share": round(share, 3), "chance": round(chance, 3),
            "layouts": {m: {"pearson_r": round(r, 3), "trustworthiness": round(t, 3),
                            "points": [[round(float(x), 4) for x in p] for p in pts]}
                        for m, (pts, r, t) in layouts.items()},
            "clusterings": {name: {"stability": round(st, 3), "labels": [int(x) + 1 for x in lab],
                                   "medoids": [voices[i]["voice"] for i in medoids(sim, lab).values()],
                                   "table": cluster_table(voices, sim, lab, pitch)}
                            for name, (lab, st) in clusterings.items()},
            "similarity": [[round(float(x), 2) for x in row] for row in zscore(sim)],
        }
    pts, r, t = layout_3d(before, "mds")
    share, chance = same_dataset_share(before, datasets)
    ways["before"] = {"title": "Before: Resemblyzer, one clip, as measured", "default_layout": "mds",
                      "same_dataset_share": round(share, 3), "chance": round(chance, 3),
                      "layouts": {"mds": {"pearson_r": round(r, 3), "trustworthiness": round(t, 3),
                                          "points": [[round(float(x), 4) for x in p] for p in pts]}},
                      "clusterings": {}}

    result = {
        "name": "voice-map-v2",
        "what": "Which validation voices sound alike independent of their dataset: every clip of a voice averaged, "
                "each dataset's mean removed, similarity z-normalised per dataset pair (s-norm). Way 1: CAM++ "
                "(CosyVoice's speaker model), nested clusters, MDS. Way 2: CAM++ + Resemblyzer, graph communities, t-SNE.",
        "voice_manifest_sha256": manifest_sha256(),
        "voices_count": len(voices),
        "clips_used": clips_used,
        "voices": [{k: v[k] for k in ("voice", "dataset", "speaker", "clip_id", "audio", "seconds")}
                   | {"pitch_hz": round(pitch[v["voice"]]) if pitch.get(v["voice"]) else None,
                      "clips": len(clips_used[v["voice"]])} for v in voices],
        "ways": ways,
        "pick": None,
    }
    for i, v in enumerate(result["voices"]):
        s = np.array(way1[i]); s[i] = -np.inf
        v["nearest"] = [voices[j]["voice"] for j in np.argsort(-s)[:3]]
    fam = ways["way1"]["clusterings"]
    if dedupe:
        judge = zscore(way2)
        chosen, loners, stands_for = dedupe_pick(judge)
        name = lambda i: voices[i]["voice"]
        result["pick"] = {
            "kind": "dedupe", "n": len(chosen),
            "shares": {d: int(sum(datasets[i] == d for i in chosen)) for d in sorted(set(datasets))},
            "rule": f"one voice per tight cluster plus every loner, in the Way 2 similarity (both judges): close = "
                    f"among the {CLOSE_SHARE:.0%} most alike pairs; loners = the {LONER_SHARE:.0%} of voices whose "
                    f"closest voice is least alike",
            "voices": [name(i) for i in chosen], "loners": [name(i) for i in loners],
            "stands_for": {name(p): [name(i) for i in sorted(m, key=lambda i: (i != p, -judge[i, p]))]
                           for p, m in stands_for.items()},
            "coverage_z": {a: round(b, 2) for a, b in coverage(judge, chosen).items()},
            "coverage_random_z": round(random_coverage(judge, datasets, len(chosen)), 2),
            "covered": {lv: sorted({fam[lv]["labels"][i] for i in chosen}) for lv in LEVELS}}
    elif pick:
        chosen = spread_pick(way1, datasets, pick)
        result["pick"] = {
            "kind": "spread", "n": pick, "shares": {d: int(k) for d, k in quotas(datasets, pick).items()},
            "rule": "greedy coverage in the Way 1 similarity, equal share per dataset",
            "voices": [voices[i]["voice"] for i in chosen], "loners": [], "stands_for": {},
            "coverage_z": {a: round(b, 2) for a, b in coverage(zscore(way1), chosen).items()},
            "coverage_random_z": round(random_coverage(zscore(way1), datasets, pick), 2),
            "covered": {lv: sorted({fam[lv]["labels"][i] for i in chosen}) for lv in LEVELS}}
    return result


def copy_audio(result, out_dir=OUT_DIR):
    """Each reference clip copied to audio/ beside the page, so the page plays it however it is opened."""
    import shutil

    folder = Path(out_dir) / "audio"
    folder.mkdir(parents=True, exist_ok=True)
    for v in result["voices"]:
        name = v["voice"].replace("/", "_") + Path(v["audio"]).suffix
        target = folder / name
        if not target.is_file() or target.stat().st_size != Path(v["audio"]).stat().st_size:
            shutil.copyfile(v["audio"], target)
        v["play"] = f"audio/{name}"


def save(result, out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_audio(result, out_dir)
    (out_dir / "voice_map.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    (out_dir / "voice_map.html").write_text(page(result), encoding="utf-8")
    return out_dir


def page(result):
    """One page (plotly.js from jsdelivr): the two ways and the old view, switchable, with the cluster tables."""
    data = json.dumps({k: result[k] for k in ("voices", "ways", "pick")}, separators=(",", ":"))
    p = result["pick"]
    pick_note = ""
    if p:
        how = ("one voice from each tight cluster plus every loner; ringed on the map" if p["kind"] == "dedupe"
               else "ringed")
        pick_note = (f"<p><b>Suggested pick of {p['n']}</b> ({', '.join(f'{d} {k}' for d, k in p['shares'].items())}; "
                     f"{how}): it covers {len(p['covered']['family'])} of 2 families, {len(p['covered']['group'])} of 4 "
                     f"groups and {len(p['covered']['neighbourhood'])} of 8 neighbourhoods. Every voice is on average "
                     f"{p['coverage_z']['mean']:.2f} standard deviations of similarity above average with its closest "
                     f"pick (random picks of the same size: {p['coverage_random_z']:.2f}).<br>"
                     f"<span class='muted'>Rule: {p['rule']}.</span></p>")
    if p and p["stands_for"]:
        lone = set(p["loners"])
        rows = "".join(
            f"<tr><td class='chips'><span data-v='{v}' style='font-weight:700'>{v}</span></td>"
            f"<td>{'loner' if v in lone else 'centre of a tight cluster'}</td><td class='num'>{len(m) - 1}</td>"
            f"<td class='chips'>{''.join(f'<span data-v={chr(39)}{o}{chr(39)}>{o}</span>' for o in m[1:])}</td></tr>"
            for v, m in sorted(p["stands_for"].items(), key=lambda kv: (kv[0] in lone, -len(kv[1]))))
        pick_note += (f"<div class='card wrap'><h2>The {p['n']} picks and the voices each stands for</h2>"
                      f"<p class='muted'>Each dropped voice is listed under its most alike pick, most alike first. "
                      f"Click a name to hear it.</p><table id='picks'><tr><th>Pick</th><th>Why</th>"
                      f"<th class='num'>Stands for</th><th>Dropped voices</th></tr>{rows}</table></div>")
    w1, w2, b = result["ways"]["way1"], result["ways"]["way2"], result["ways"]["before"]
    c1 = w1["clusterings"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Voice Map</title>
<script src="{PLOTLY}"></script>
<style>
:root{{--bg:#fafaf8;--fg:#1d1d1b;--muted:#6b6b66;--line:#e2e1dc;--card:#fff;--accent:#2E6FD8}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161615;--fg:#ecebe6;--muted:#a3a29c;--line:#33322f;--card:#1f1f1d;--accent:#6c9cf0}}}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1200px;margin:auto;padding:0 16px 32px}} h1{{font-size:22px;margin:16px 0 4px}} h2{{font-size:17px;margin:4px 0 8px}}
.muted{{color:var(--muted)}} .card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:12px 0}}
#map{{height:620px}} table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}}
.chips span{{display:inline-block;margin:2px 4px 2px 0;padding:1px 6px;border:1px solid var(--line);border-radius:6px;cursor:pointer;font-size:12px}}
.chips span:hover{{border-color:var(--accent)}}
.seg{{display:inline-flex;flex-wrap:wrap;border:1px solid var(--line);border-radius:8px;overflow:hidden;margin:0 10px 8px 0}}
.seg button{{background:none;border:0;padding:5px 10px;color:var(--fg);cursor:pointer;font:inherit;font-size:13px}}
.seg button.on{{background:var(--accent);color:#fff}} .seg button:disabled{{opacity:.35;cursor:default}}
#now{{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:8px 16px;z-index:2}}
.wrap{{overflow-x:auto}} .num{{text-align:right;font-variant-numeric:tabular-nums}}
</style></head><body>
<div id="now"><b id="nowname">Click a voice to hear its reference clip</b> <span class="muted" id="nowinfo"></span><br>
<audio id="player" controls style="width:100%;max-width:520px"></audio></div>
<main>
<h1>Voice map: {result['voices_count']} validation voices</h1>
<p class="muted">The plain similarity groups voices by the dataset they were recorded in, not by how they sound
({b['same_dataset_share']:.0%} of a voice's 5 nearest voices are from its own dataset; chance is {b['chance']:.0%}).
Both ways below average every clip of a voice, remove each dataset's average and compare similarities within
their dataset pair, so a voice's neighbours come from any dataset. Voices form a continuum: only the two families
are near certain, groups and neighbourhoods are for browsing.</p>
<div class="card wrap"><table>
<tr><th></th><th class="num">Nearest 5 from own dataset<br><span class="muted">chance {w1['chance']:.0%}</span></th><th class="num">Clusters and stability<br><span class="muted">1 = identical on 80 % subsets</span></th><th class="num">3D fits distances (r)</th><th class="num">3D keeps neighbours</th></tr>
<tr><td><b>{w1['title']}</b><br><span class="muted">CAM++, the speaker model CosyVoice clones from</span></td><td class="num">{w1['same_dataset_share']:.0%}</td>
<td class="num">2 families {c1['family']['stability']:.2f}<br>4 groups {c1['group']['stability']:.2f}<br>8 neighbourhoods {c1['neighbourhood']['stability']:.2f}</td>
<td class="num">{w1['layouts']['mds']['pearson_r']:.2f} (MDS)</td><td class="num">{w1['layouts']['mds']['trustworthiness']:.2f}</td></tr>
<tr><td><b>{w2['title']}</b><br><span class="muted">CAM++ and Resemblyzer averaged</span></td><td class="num">{w2['same_dataset_share']:.0%}</td>
<td class="num">{len(w2['clusterings']['community']['table'])} communities {w2['clusterings']['community']['stability']:.2f}</td>
<td class="num">{w2['layouts']['tsne']['pearson_r']:.2f} (t-SNE)</td><td class="num">{w2['layouts']['tsne']['trustworthiness']:.2f}</td></tr>
<tr><td><b>{b['title']}</b></td><td class="num">{b['same_dataset_share']:.0%}</td><td class="num">&mdash;</td><td class="num">{b['layouts']['mds']['pearson_r']:.2f}</td><td class="num">{b['layouts']['mds']['trustworthiness']:.2f}</td></tr>
</table></div>
{pick_note}
<div class="card">
<div class="seg" id="way"><button data-v="way1" class="on">Way 1</button><button data-v="way2">Way 2</button><button data-v="before">Before</button></div>
<div class="seg" id="level"></div>
<div class="seg" id="colour"><button data-v="cluster" class="on">Colour: cluster</button><button data-v="dataset">Dataset</button><button data-v="pitch">Pitch</button></div>
<div class="seg" id="layout"><button data-v="mds">MDS</button><button data-v="tsne">t-SNE</button></div>
<div class="muted" id="fit"></div>
<div id="map"></div></div>
<div class="card"><h2 id="tabletitle">Clusters</h2><p class="muted">Members are listed most typical first (bold = medoid); &#10003; = in the suggested pick. Click a name to hear it.</p><div class="wrap"><table id="clusters"></table></div></div>
</main>
<script>
const D = {data};
const CAT = ["#2E6FD8","#E0702B","#2A9D6F","#C2417E","#7B5BD6","#B88A1B","#1F9BB0","#D24D3E","#5E7D2A","#8C5A3C","#4C5BA8","#A23E9E"];
const DS = {{"VCTK": 0, "CREMA-D": 1, "LibriTTS-R": 2}}, SYM = {{"VCTK": "circle", "CREMA-D": "diamond", "LibriTTS-R": "square"}};
const picked = new Set(D.pick ? D.pick.voices : []), lone = new Set(D.pick ? D.pick.loners : []);
const dark = matchMedia("(prefers-color-scheme: dark)").matches, ink = dark ? "#ecebe6" : "#1d1d1b", grid = dark ? "#33322f" : "#e2e1dc";
const esc = s => String(s).replace(/[&<>"']/g, c => ({{"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}})[c]);
let way = "way1", level = "group", colour = "cluster", layout = "mds";
const W = () => D.ways[way];
const levels = () => Object.keys(W().clusterings);
function labelsOf() {{ const c = W().clusterings[level]; return c ? c.labels : null; }}
function play(name) {{
  const i = D.voices.findIndex(o => o.voice === name); if (i < 0) return; const x = D.voices[i], lab = labelsOf();
  document.getElementById("nowname").textContent = x.voice + (lab ? "  (" + level + " " + lab[i] + ")" : "");
  document.getElementById("nowinfo").textContent = x.clip_id + ", " + x.seconds.toFixed(1) + " s, " + (x.pitch_hz || "?") + " Hz; nearest voices (Way 1): " + x.nearest.join(", ");
  const a = document.getElementById("player"); a.src = x.play; a.play().catch(() => {{}});
}}
function groups() {{
  const lab = labelsOf();
  if (colour === "dataset" || !lab) return Object.keys(DS).map(d => [d, i => D.voices[i].dataset === d, CAT[DS[d]]]);
  if (colour === "pitch") return [["All voices", i => true, null]];
  return [...new Set(lab)].sort((a, b) => a - b).map(c => [level[0].toUpperCase() + level.slice(1) + " " + c, i => lab[i] === c, CAT[(c - 1) % CAT.length]]);
}}
function drawMap() {{
  const L = W().layouts[layout] || W().layouts[W().default_layout], P = L.points, lab = labelsOf();
  const meds = new Set(lab ? W().clusterings[level].medoids : []);
  const idx = D.voices.map((v, i) => i);
  const traces = groups().map(([name, keep, col]) => {{
    const ids = idx.filter(keep), vs = ids.map(i => D.voices[i]);
    return {{type: "scatter3d", mode: "markers", name: name + " (" + ids.length + ")", customdata: vs.map(v => v.voice),
      x: ids.map(i => P[i][0]), y: ids.map(i => P[i][1]), z: ids.map(i => P[i][2]),
      text: ids.map(i => "<b>" + D.voices[i].voice + "</b><br>" + (lab ? level + " " + lab[i] + ", " : "") + (D.voices[i].pitch_hz || "?") + " Hz" + (meds.has(D.voices[i].voice) ? "<br>medoid" : "") + (picked.has(D.voices[i].voice) ? (lone.has(D.voices[i].voice) ? "<br>picked (loner)" : "<br>picked") : "")),
      hovertemplate: "%{{text}}<extra></extra>",
      marker: {{size: vs.map(v => meds.has(v.voice) ? 9 : 5), symbol: vs.map(v => SYM[v.dataset]), opacity: 0.92,
        color: col || vs.map(v => v.pitch_hz || 0), colorscale: col ? undefined : "Viridis", showscale: !col,
        colorbar: col ? undefined : {{title: {{text: "Hz"}}, thickness: 12}},
        line: {{color: vs.map(v => picked.has(v.voice) ? ink : "rgba(0,0,0,0)"), width: vs.map(v => picked.has(v.voice) ? 4 : 0)}}}}}};
  }});
  const axis = {{showticklabels: false, title: "", gridcolor: grid, zerolinecolor: grid, showbackground: false}};
  Plotly.react("map", traces, {{paper_bgcolor: "rgba(0,0,0,0)", font: {{color: ink}}, margin: {{l: 0, r: 0, t: 0, b: 0}},
    scene: {{xaxis: axis, yaxis: axis, zaxis: axis}}, legend: {{itemclick: "toggleothers"}}, uirevision: way + layout}}, {{responsive: true}});
  document.getElementById("fit").textContent = W().title + ". " + (layout === "mds" ? "MDS keeps overall distances" : "t-SNE keeps each voice's nearest voices (overall distances are not to scale)") +
    ": distance fit r = " + L.pearson_r.toFixed(2) + ", neighbours kept " + L.trustworthiness.toFixed(2) + ". Shape = dataset (circle VCTK, diamond CREMA-D, square LibriTTS-R); large = medoid; ringed = picked. Drag to turn, click a point to hear it, click a legend entry to show it alone.";
}}
function drawTable() {{
  const c = W().clusterings[level], t = document.getElementById("clusters");
  document.getElementById("tabletitle").textContent = c ? "Clusters: " + W().title + ", " + level + " (stability " + c.stability.toFixed(2) + ")" : "No clusters in this view";
  if (!c) {{ t.innerHTML = ""; return; }}
  t.innerHTML = "<tr><th>" + esc(level) + "</th><th>Voices</th><th>Datasets</th><th>Pitch (Hz)</th><th>Members</th></tr>" +
    c.table.map(r => "<tr><td><span style='color:" + CAT[(r.cluster - 1) % CAT.length] + "'>&#9679;</span> " + r.cluster + "</td><td>" + r.size + "</td><td>" +
      Object.entries(r.datasets).map(([d, n]) => esc(d) + " " + n).join("<br>") + "</td><td>" +
      (r.pitch_hz ? r.pitch_hz.median + "<br><span class='muted'>" + r.pitch_hz.low + "&ndash;" + r.pitch_hz.high + "</span>" : "") + "</td><td class='chips'>" +
      r.members.map(m => "<span data-v='" + esc(m) + "'" + (m === r.medoid ? " style='font-weight:700'" : "") + ">" + esc(m) + (picked.has(m) ? " &#10003;" : "") + "</span>").join("") + "</td></tr>").join("");
  t.querySelectorAll("span[data-v]").forEach(s => s.onclick = () => play(s.dataset.v));
}}
function controls() {{
  const lv = levels(); if (!lv.includes(level)) level = lv.includes("group") ? "group" : lv[0];
  document.getElementById("level").innerHTML = lv.map(l => "<button data-v='" + l + "'" + (l === level ? " class='on'" : "") + ">" + l + (W().clusterings[l] ? " (" + W().clusterings[l].table.length + ")" : "") + "</button>").join("");
  document.getElementById("level").style.display = lv.length ? "" : "none";
  if (!W().layouts[layout]) layout = W().default_layout;
  document.querySelectorAll("#layout button").forEach(b => {{ b.disabled = !W().layouts[b.dataset.v]; b.classList.toggle("on", b.dataset.v === layout); }});
  document.querySelectorAll("#level button").forEach(b => b.onclick = () => {{ level = b.dataset.v; redraw(); }});
}}
function redraw() {{ controls(); drawMap(); drawTable(); }}
function seg(id, set) {{
  document.querySelectorAll("#" + id + " button").forEach(b => b.onclick = () => {{
    if (b.disabled) return; document.querySelectorAll("#" + id + " button").forEach(o => o.classList.toggle("on", o === b)); set(b.dataset.v); redraw();
  }});
}}
seg("way", v => {{ way = v; layout = D.ways[v].default_layout; }}); seg("colour", v => colour = v); seg("layout", v => layout = v);
redraw();
document.getElementById("map").on("plotly_click", e => play(e.points[0].customdata));
document.querySelectorAll("#picks span[data-v]").forEach(s => s.onclick = () => play(s.dataset.v));
</script></body></html>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pick", type=int, default=None, help="also suggest N voices that cover the map")
    parser.add_argument("--dedupe", action="store_true",
                        help="instead suggest one voice per tight cluster plus every loner (the number follows from the data)")
    args = parser.parse_args()
    result = build(pick=args.pick, dedupe=args.dedupe)
    out = save(result)
    for key in ("before", "way1", "way2"):
        w = result["ways"][key]
        print(f"\n{w['title']}: nearest 5 from own dataset {w['same_dataset_share']:.0%} (chance {w['chance']:.0%})")
        for name, c in w["clusterings"].items():
            print(f"  {name}: {len(c['table'])} clusters, stability {c['stability']}")
            for row in c["table"]:
                mix = ", ".join(f"{d} {n}" for d, n in row["datasets"].items())
                print(f"    {row['cluster']}: {row['size']:>3} voices ({mix}), pitch {row['pitch_hz']['median']} Hz, "
                      f"medoid {row['medoid']}")
    if result["pick"]:
        p = result["pick"]
        print(f"\npick {p['n']} {p['shares']}: {', '.join(p['voices'])}\n  coverage {p['coverage_z']} (random "
              f"{p['coverage_random_z']}); covers {p['covered']}")
        for v, members in p["stands_for"].items():
            tag = "loner" if v in p["loners"] else f"stands for {len(members) - 1}"
            print(f"    {v:16s} {tag:15s} {', '.join(members[1:])}")
    print(f"\n{out / 'voice_map.html'}")


if __name__ == "__main__":
    main()

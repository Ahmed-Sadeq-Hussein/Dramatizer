"""explore.py | Far-apart genomes for listening tests: up to 10 per voice, split into 3 piles for 3 listeners.

    python main_backend.py explore                  # prepare (once), render until done or --hours, then publish
    python main_backend.py explore --hours 6.5      # stop starting new renders after 6.5 hours, then publish
    python main_backend.py explore --publish-only   # render nothing; choose and publish from what is rendered
    python main_backend.py explore --status         # how far it is

The voices are the dedupe pick of the voice map (voice_map.py --dedupe: one voice
per tight cluster plus every loner, 24 validation voices on 9 Nov 2026), frozen
in plan.json when the batch is prepared. Every voice speaks the same line, the
shared comparison sentence of the baseline (praise, sarcasm or threat depending
on delivery), so listeners compare deliveries, not words.

How the genomes are made far apart, in three steps:

  1. Genome space, before rendering. A genome becomes a vector: which features it
     has (each of the 14 word features and the pause), where in the line its edits
     sit (start, middle, end), its pause kinds and its speed. Per voice,
     CANDIDATES genomes are chosen by farthest-point sampling: each next genome is
     the one farthest from the voice's genomes so far (and from the plain line),
     plus COVERAGE_WEIGHT times its distance from every genome any voice has, so
     the voices together cover the space and no genome is used twice. Gene counts
     cycle 1, 2, 3, 4, 5, so light and heavy edits are both tried.
  2. Rendering, round by round: candidate 1 of every voice, then candidate 2, and
     so on, so a stop leaves every voice with its most different candidates.
     Each render is judged by the sanity check (the same as search.py) and
     measured against the voice's plain render: length, pitch level, pitch
     range, loudness range, longest pause, word changes, speaker similarity.
     Saved after every render; a stopped batch continues where it was.
  3. Choosing, after rendering. Among the candidates judged fine, up to KEEP per
     voice by farthest-point sampling again, now mostly on what is HEARD (the
     measurements above, scaled by their spread over all renders; genome vector
     at HEARD_GENOME_WEIGHT), starting from the plain render, so each pick
     differs from the plain line and from every other pick.

Piles: the voices are dealt into PILES piles like cards (sorted by dataset, then
pitch, then loner or not, snake order), so each pile has a similar mix. Each
pile is one listener's work: about 8 runs of 1 plain + up to 10 genome clips.

Publishing writes one ordinary genome-search run per voice (search.py's format)
in RUNS_DIR, with "batch" and "pile" in run.json, so the MOS page (the client's
Genomes tab, filtered by pile) and ratings.py work on them as on any search. The
genome order within a run is shuffled (seeded), so the order a listener hears
them in says nothing about them.

Everything else goes in OUTPUT_DIR/genome_explore/<batch>/ (never committed):
plan.json, voices/<voice>/state.json and the candidate renders.
"""
import argparse
import json
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from backend import model
from backend.genome import search
from backend.genome.candidate import Candidate
from backend.genome.genome_factory import PAUSE_KINDS, TARGETED_FEATURES, GenomeFactory

BATCH = "mos-diverse-v1"
OUT_DIR = model.OUTPUT_DIR / "genome_explore"
# The shared comparison sentence of baseline v2 (backend/evaluation/baseline.py, LINES[-1])
LINE = "Well, you really did it this time, and I honestly didn't think you had it in you, so I'll be waiting."
GENE_CYCLE = (1, 2, 3, 4, 5)
POOL_PER_COUNT = 1500        # random genomes drawn per gene count, before removing ones with the same markup
CANDIDATES = 36              # rendered per voice (if time allows)
KEEP = 10                    # at most this many per voice go to the listeners
PILES = 3
COVERAGE_WEIGHT = 0.5
HEARD_GENOME_WEIGHT = 0.5
SEED = 0
SR = 16000
STATE = "state.json"

POSITION_FEATURES = TARGETED_FEATURES + ["pause"]


def _save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _load(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def safe(voice):
    return voice.replace("/", "_")


# --- Genome space (pure, tested in test_explore.py) ---------------------------

def genome_vector(genes):
    """A genome as numbers: features present, where edits sit (start/middle/end), pause kinds, speed."""
    v = np.zeros(len(POSITION_FEATURES) + 3 + len(PAUSE_KINDS) + 1)
    for gene in genes:
        feature = gene["feature"]
        if feature == "speed":
            v[-1] = (gene["rate"] - 1.0) / 0.15
            continue
        v[POSITION_FEATURES.index(feature)] += 1
        v[len(POSITION_FEATURES) + min(int(gene["position"] * 3), 2)] += 0.5
        if feature == "pause":
            v[len(POSITION_FEATURES) + 3 + PAUSE_KINDS.index(gene.get("kind", PAUSE_KINDS[0]))] += 0.5
    return v


def draw_pool(text=LINE, per_count=POOL_PER_COUNT, seed=SEED):
    """{gene count: [genes]} of random genomes, one per distinct markup on this line."""
    from backend.speech.tts_formatter import process_tts_string

    pool, seen = {}, set()
    for n in GENE_CYCLE:
        factory = GenomeFactory(min_operations=n, max_operations=n, seed=seed * 100 + n)
        pool[n] = []
        for _ in range(per_count):
            genome = factory.random_genome(text)
            plan = process_tts_string(text, genome.resolve(text))
            key = (plan.text, plan.speed, plan.instruct)
            if key not in seen:
                seen.add(key)
                pool[n].append(genome.operations)
    return pool


def spread_candidates(pool, voices, per_voice, coverage_weight=COVERAGE_WEIGHT):
    """
    {voice: [genes, ...]}: farthest-point sampling per voice in genome space, with a reward for being far from
    every voice's genomes, round by round over the voices, gene counts cycling through GENE_CYCLE.
    No genome is given to two voices.
    """
    vectors = {n: np.array([genome_vector(g) for g in genomes]) for n, genomes in pool.items()}
    plain = genome_vector([])
    own = {v: [plain] for v in voices}
    everyone = [plain]
    used = {n: np.zeros(len(pool[n]), bool) for n in pool}
    out = {v: [] for v in voices}
    counts = [n for n in GENE_CYCLE if n in pool]
    for rank in range(per_voice):
        for k, voice in enumerate(voices):
            n = counts[(rank + k) % len(counts)]          # voices start the cycle at different counts
            if used[n].all():
                continue
            X = vectors[n]
            d_own = np.min(np.linalg.norm(X[:, None] - np.array(own[voice])[None], axis=2), axis=1)
            d_all = np.min(np.linalg.norm(X[:, None] - np.array(everyone)[None], axis=2), axis=1)
            score = np.where(used[n], -np.inf, d_own + coverage_weight * d_all)
            best = int(np.argmax(score))
            used[n][best] = True
            own[voice].append(X[best])
            everyone.append(X[best])
            out[voice].append(pool[n][best])
    return out


def farthest_points(points, k, start):
    """Indices of up to k rows of `points`, each the farthest from `start` and the rows chosen before it."""
    points = np.asarray(points, dtype=float)
    if not len(points):
        return []
    chosen = []
    nearest = np.linalg.norm(points - np.asarray(start, dtype=float), axis=1)
    for _ in range(min(k, len(points))):
        i = int(np.argmax(np.where(np.isin(np.arange(len(points)), chosen), -np.inf, nearest)))
        chosen.append(i)
        nearest = np.minimum(nearest, np.linalg.norm(points - points[i], axis=1))
    return chosen


def deal_piles(voices, piles=PILES):
    """
    {voice: pile number from 1}: voices sorted by dataset, pitch and loner, dealt in snake order
    (1 2 3 3 2 1 ...), so every pile gets a similar mix. `voices`: [{"voice", "dataset", "pitch_hz", "loner"}].
    """
    order = sorted(voices, key=lambda v: (v["dataset"], -(v.get("pitch_hz") or 0), not v.get("loner"), v["voice"]))
    out = {}
    for i, v in enumerate(order):
        lap, place = divmod(i, piles)
        out[v["voice"]] = (place if lap % 2 == 0 else piles - 1 - place) + 1
    return out


# --- What is heard -------------------------------------------------------------

HEARD = ["length", "pitch_level", "pitch_range", "loudness_range", "longest_pause", "word_changes", "voice_change"]


def acoustics(path):
    """Seconds, median pitch and pitch spread (semitones), loudness spread (dB), longest inner pause (s)."""
    import librosa

    from backend.audio_io import load_mono

    wav = load_mono(path, SR)
    hop = 256
    rms = librosa.feature.rms(y=wav, frame_length=1024, hop_length=hop)[0]
    loud = rms > 0.1 * np.max(rms) if rms.size else np.zeros(0, bool)
    f0 = librosa.yin(wav, fmin=60, fmax=500, sr=SR, frame_length=1024, hop_length=hop)[:len(loud)]
    voiced = f0[loud[:len(f0)]]
    st = 12 * np.log2(voiced / 100) if voiced.size else np.zeros(1)
    db = 20 * np.log10(rms[loud] + 1e-9) if loud.any() else np.zeros(1)
    inner = np.flatnonzero(loud)
    pause = 0
    if inner.size:
        quiet, run = ~loud[inner[0]:inner[-1] + 1], 0
        for q in quiet:
            run = run + 1 if q else 0
            pause = max(pause, run)
    return {"seconds": len(wav) / SR, "pitch_st": float(np.median(st)), "pitch_spread_st": float(np.std(st)),
            "loudness_spread_db": float(np.std(db)), "longest_pause": pause * hop / SR}


def heard_change(test, plain, sanity):
    """What a genome changed, against the plain render (0 = no change)."""
    return {
        "length": float(np.log(test["seconds"] / plain["seconds"])),
        "pitch_level": test["pitch_st"] - plain["pitch_st"],
        "pitch_range": test["pitch_spread_st"] - plain["pitch_spread_st"],
        "loudness_range": test["loudness_spread_db"] - plain["loudness_spread_db"],
        "longest_pause": test["longest_pause"] - plain["longest_pause"],
        "word_changes": float(sanity.get("word_errors") or 0),
        "voice_change": 1.0 - float(sanity.get("similarity") or 1.0),
    }


# --- The batch: plan, render, choose, publish ---------------------------------

def batch_dir(batch=BATCH):
    return OUT_DIR / batch


def prepare(batch=BATCH, candidates=CANDIDATES):
    """plan.json (voices, piles, seeds) and each voice's state.json with its candidates; made once, then reused."""
    from backend.voice_data.split_voice_data import load_voice_manifest
    from backend.voice_data.voice_map import OUT_DIR as MAP_DIR

    folder = batch_dir(batch)
    plan = _load(folder / "plan.json")
    if plan:
        return plan
    vmap = _load(MAP_DIR / "voice_map.json")
    if not vmap or not vmap.get("pick") or vmap["pick"].get("kind") != "dedupe":
        raise SystemExit("Run `python main_backend.py voice-map --dedupe` first: the voices come from its pick.")
    manifest = {v["voice"]: v for v in load_voice_manifest()["voices"]}
    info = {v["voice"]: v for v in vmap["voices"]}
    loners = set(vmap["pick"]["loners"])
    voices = []
    for name in vmap["pick"]["voices"]:
        m = manifest[name]
        if m["split"] != "validation":
            raise SystemExit(f"{name} is not a validation voice; test voices stay locked.")
        voices.append({"voice": name, "dataset": m["dataset"], "speaker": m["speaker"], "clip_id": m["clip_id"],
                       "reference_audio": info[name]["audio"], "reference_text": m["transcript"],
                       "pitch_hz": info[name].get("pitch_hz"), "loner": name in loners,
                       "origin": {"dataset": m["dataset"], "speaker": m["speaker"], "clip_id": m["clip_id"],
                                  "details": m.get("details") or {}, "split": m["split"]}})
    piles = deal_piles(voices)
    for v in voices:
        v["pile"] = piles[v["voice"]]

    print(f"Drawing the genome pool and spreading {candidates} candidates over {len(voices)} voices...", flush=True)
    pool = draw_pool()
    spread = spread_candidates(pool, [v["voice"] for v in voices], candidates)
    plan = {"batch": batch, "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "line": LINE,
            "render_seed": search.render_seed_default(), "genome_seed": SEED, "candidates": candidates,
            "keep": KEEP, "piles": PILES, "coverage_weight": COVERAGE_WEIGHT,
            "heard_genome_weight": HEARD_GENOME_WEIGHT, "pool": {str(n): len(g) for n, g in pool.items()},
            "voice_map": {"sha256": vmap["voice_manifest_sha256"], "rule": vmap["pick"]["rule"]},
            "voices": voices}
    for v in voices:
        cands = []
        for i, genes in enumerate(spread[v["voice"]], 1):
            genome = Candidate(genes)
            cands.append({"number": i, "genes": genes, "described": genome.describe(LINE),
                          "operations": genome.resolve(LINE), "audio": None, "sanity": None, "verdict": None})
        _save(folder / "voices" / safe(v["voice"]) / STATE, {"voice": v["voice"], "plain": None, "candidates": cands})
    _save(folder / "plan.json", plan)
    return plan


def _plan_of(operations):
    from backend.speech.tts_formatter import process_tts_string
    return process_tts_string(LINE, operations)


def render_all(plan, hours=None, render=search.render_plan, judge=search.judge_with_sanity):
    """Plain renders first, then candidate 1 of every voice, candidate 2, ...; saved after each one."""
    folder = batch_dir(plan["batch"])
    deadline = time.time() + hours * 3600 if hours else None
    seed = plan["render_seed"]
    states = {v["voice"]: folder / "voices" / safe(v["voice"]) for v in plan["voices"]}
    done = timed = 0
    started_all = time.time()

    def out_of_time():
        return deadline is not None and time.time() > deadline

    for v in plan["voices"]:
        vdir = states[v["voice"]]
        state = _load(vdir / STATE)
        if state["plain"] is None:
            if out_of_time():
                return False
            path = vdir / "plain.wav"
            render(v["reference_audio"], v["reference_text"], _plan_of([]), path, seed)
            state["plain"] = {"audio": path.name, "acoustics": acoustics(path), "heard": None}
            _save(vdir / STATE, state)
            print(f"  plain {v['voice']}", flush=True)

    for rank in range(plan["candidates"]):
        for v in plan["voices"]:
            vdir = states[v["voice"]]
            state = _load(vdir / STATE)
            if rank >= len(state["candidates"]):
                continue
            cand = state["candidates"][rank]
            if cand["verdict"] is not None:
                continue
            if out_of_time():
                return False
            started = time.time()
            path = vdir / f"candidate_{cand['number']:02d}.wav"
            try:
                tts = _plan_of(cand["operations"])
                if cand["audio"] is None or not path.is_file():
                    render(v["reference_audio"], v["reference_text"], tts, path, seed)
                    cand.update(audio=path.name, markup=tts.text, speed=tts.speed, warnings=tts.warnings)
                result = search._plain(judge(v["reference_audio"], vdir / "plain.wav", path,
                                             state["plain"]["heard"], tts.speed))
                state["plain"]["heard"] = state["plain"]["heard"] or result.get("baseline_heard")
                cand.update(sanity=result, verdict="fine" if result["acceptable"] else "terrible",
                            heard=heard_change(acoustics(path), state["plain"]["acoustics"], result),
                            seconds=round(time.time() - started, 1), error=None)
            except Exception as error:   # one bad render must not stop a 7-hour batch
                cand.update(verdict="failed", error=f"{type(error).__name__}: {error}")
            _save(vdir / STATE, state)
            done += 1
            timed += time.time() - started
            left = sum(1 for s in states.values() for c in _load(s / STATE)["candidates"] if c["verdict"] is None) \
                if done % 24 == 0 else None
            note = f", {left} left, about {left * timed / done / 3600:.1f} h" if left is not None else ""
            print(f"  round {rank + 1} {v['voice']:16s} {cand['verdict']:9s} {cand.get('seconds', 0):5.1f} s{note}",
                  flush=True)
    print(f"All rendered ({(time.time() - started_all) / 3600:.2f} h this session).", flush=True)
    return True


def choose(plan):
    """{voice: [candidate, ...]}: up to KEEP fine candidates per voice, far apart in what is heard."""
    folder = batch_dir(plan["batch"])
    states = {v["voice"]: _load(folder / "voices" / safe(v["voice"]) / STATE) for v in plan["voices"]}
    fine = {name: [c for c in s["candidates"] if c["verdict"] == "fine" and c.get("heard")]
            for name, s in states.items()}
    every = [c for cs in fine.values() for c in cs]
    if not every:
        return {name: [] for name in states}
    heard = np.array([[c["heard"][k] for k in HEARD] for c in every])
    genome = np.array([genome_vector(c["genes"]) for c in every])
    h_scale = heard.std(axis=0) + 1e-9          # no centring: the plain render stays at 0
    g_scale = np.sqrt((genome ** 2).mean(axis=0)) + 1e-9
    out = {}
    for name, cs in fine.items():
        if not cs:
            out[name] = []
            continue
        X = np.hstack([np.array([[c["heard"][k] for k in HEARD] for c in cs]) / h_scale,
                       plan["heard_genome_weight"] * np.array([genome_vector(c["genes"]) for c in cs]) / g_scale])
        out[name] = [cs[i] for i in farthest_points(X, plan["keep"], np.zeros(X.shape[1]))]
    return out


def publish(plan, runs_dir=None):
    """One genome-search run per voice in RUNS_DIR, with the batch and pile; never over a run that has ratings."""
    from backend.genome import ratings
    from backend.metrics.reward import CONFIG

    folder = batch_dir(plan["batch"])
    chosen = choose(plan)
    published = []
    for v in plan["voices"]:
        vdir = folder / "voices" / safe(v["voice"])
        state = _load(vdir / STATE)
        picks = list(chosen[v["voice"]])
        if not picks or state["plain"] is None:
            print(f"  {v['voice']}: nothing fine to publish yet", flush=True)
            continue
        random.Random(f"{plan['batch']}/{v['voice']}").shuffle(picks)   # listening order says nothing
        run_id = f"{plan['batch']}_p{v['pile']}_{safe(v['voice'])}"
        run_dir = search.run_folder(run_id, runs_dir)
        if (run_dir / ratings.RATINGS_FILE).is_file() and ratings.is_rated(run_dir):
            print(f"  {run_id}: already rated, left as it is", flush=True)
            published.append(run_id)
            continue
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True)
        shutil.copyfile(vdir / "plain.wav", run_dir / "baseline.wav")
        genomes = []
        for number, c in enumerate(picks, 1):
            name = f"genome_{number:02d}.wav"
            shutil.copyfile(vdir / c["audio"], run_dir / name)
            genomes.append({"number": number, "genes": c["genes"], "described": c["described"],
                            "operations": c["operations"], "markup": c["markup"], "speed": c["speed"],
                            "warnings": c.get("warnings", []), "audio": name, "sanity": c["sanity"],
                            "verdict": c["verdict"], "seconds": c.get("seconds"),
                            "explore": {"candidate": c["number"], "heard": c["heard"]}})
        run = {"schema_version": search.SCHEMA_VERSION, "id": run_id, "method": "diverse",
               "method_label": "Diverse exploration for listening", "voice": v["voice"], "origin": v["origin"],
               "reference_audio": v["reference_audio"], "reference_text": v["reference_text"], "text": plan["line"],
               "genes_per_genome": "1-5", "seed": plan["genome_seed"], "render_seed": plan["render_seed"],
               "limits": {"wer": CONFIG.sanity_wer_limit, "words": CONFIG.sanity_wer_words,
                          "similarity": CONFIG.sanity_similarity, "similarity_short": CONFIG.sanity_similarity_short,
                          "long_seconds": CONFIG.sanity_long_seconds, "pace": CONFIG.sanity_pace_min},
               "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "status": "finished", "error": None,
               "finished": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "batch": plan["batch"], "pile": v["pile"],
               "baseline": {"audio": "baseline.wav", "heard": state["plain"]["heard"]}, "genomes": genomes}
        search.save_run(run_dir, run)
        published.append(run_id)
    piles = {}
    for v in plan["voices"]:
        piles.setdefault(v["pile"], []).append(v["voice"])
    _save(folder / "piles.json", {"batch": plan["batch"], "published": published,
                                  "piles": {str(p): {"voices": vs, "clips": sum(len(chosen[x]) + 1 for x in vs if chosen[x])}
                                            for p, vs in sorted(piles.items())}})
    return published


def status(plan):
    folder = batch_dir(plan["batch"])
    total = fine = done = 0
    for v in plan["voices"]:
        state = _load(folder / "voices" / safe(v["voice"]) / STATE)
        total += len(state["candidates"])
        done += sum(c["verdict"] is not None for c in state["candidates"])
        fine += sum(c["verdict"] == "fine" for c in state["candidates"])
    return {"rendered": done, "of": total, "fine": fine}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--batch", default=BATCH)
    parser.add_argument("--hours", type=float, help="start no new render after this many hours, then publish")
    parser.add_argument("--publish-only", action="store_true", help="render nothing; choose and publish")
    parser.add_argument("--status", action="store_true", help="show how far the batch is")
    args = parser.parse_args()

    plan = prepare(args.batch)
    if args.status:
        print(status(plan))
        return
    if not args.publish_only:
        render_all(plan, args.hours)
    published = publish(plan)
    piles = _load(batch_dir(args.batch) / "piles.json")
    print(f"\nPublished {len(published)} runs in {search.RUNS_DIR}")
    for p, info in piles["piles"].items():
        print(f"  pile {p}: {len(info['voices'])} voices, {info['clips']} clips: {', '.join(info['voices'])}")
    print(status(plan))


if __name__ == "__main__":
    main()

"""Train a regularized conditional logit ranker on human shortlist choices."""
import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend import model
from backend.genome.descriptors import LABELS, SIZE, genome_vector

CHOICE_FILE = "preference.json"
EMBEDDING_SIZE = 256
VOICE_DIMENSIONS = 8


def model_path():
    return model.PROJECT_DIR / model.setting("PREFERENCE_MODEL_PATH", "history/genomes/preference-model.json")


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        except BaseException:
            stream.close()
            temporary.unlink()
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _rendered_genes(run):
    from backend.genome.ratings import clip_key

    clips = {"baseline": []} if run["baseline"].get("audio") else {}
    clips.update({clip_key(g["number"]): g["genes"] for g in run["genomes"] if g.get("audio")})
    return clips


def record_choice(folder, heard, winner):
    """Save the explicit winner and exactly the clips the listener compared."""
    from backend.genome.search import load_run

    folder = Path(folder)
    run = load_run(folder)
    clips = _rendered_genes(run)
    if len(heard) < 2 or len(set(heard)) != len(heard) or winner not in heard:
        raise ValueError("Give at least two distinct heard clips, including the winner.")
    if any(key not in clips for key in heard):
        raise ValueError("Every heard clip must have been rendered in this run.")
    record = {
        "schema_version": 1, "run": run["id"], "voice": run["voice"],
        "origin": run.get("origin"), "reference_audio": run["reference_audio"],
        "text": run["text"], "starting_genome": run.get("starting_genome", []),
        "heard": [{"clip": key, "genes": clips[key]} for key in heard],
        "winner": winner,
    }
    _write_json(folder / CHOICE_FILE, record)
    return record


@dataclass(frozen=True)
class Session:
    speaker: str
    reference_audio: str
    starting_genome: list
    choices: list
    winners: tuple[int, ...]


def _resolve_reference(run, indexes):
    """Find the original reference clip when a run was exported from another PC."""
    path = Path(run["reference_audio"])
    if path.is_file():
        return str(path.resolve())

    from backend.voice_data.build_manifest import SOURCES
    from backend.voice_data.Voice_Importer import get_clip

    origin = run.get("origin") or {}
    dataset, speaker, clip_id = (origin.get(key) for key in ("dataset", "speaker", "clip_id"))
    if dataset not in SOURCES or not speaker or not clip_id:
        raise ValueError(f"{run['id']}: missing reference audio and exact clip identity.")
    if dataset not in indexes:
        source = SOURCES[dataset]
        setting, default = source["folder"]
        folder = Path(model.setting("VOICE_LIBRARY_DIR")) / model.setting(setting, default)
        indexes[dataset] = {(clip.speaker, clip.clip_id): clip for clip in source["reader"](folder)}
    clip = indexes[dataset].get((speaker, clip_id))
    if clip is None:
        raise ValueError(f"{run['id']}: reference {dataset}/{speaker}/{clip_id} is missing from the local voice library.")
    audio, _ = get_clip(clip)
    return str(audio.resolve())


def collect_sessions(runs_dir=None, splits_path=None):
    """Read human choices from development speakers in the locked manifest.

    An explicit choice takes precedence over ratings for its run. Otherwise,
    each listener's highest-rated rendered clips share the target probability
    among that listener's rated clips. Completely tied shortlists are skipped.
    """
    from backend.genome import ratings, search
    from backend.voice_data.split_voice_data import SPLITS_FILE, load_splits

    splits = load_splits(SPLITS_FILE if splits_path is None else splits_path)
    allowed = set(splits["validation"]) - set(splits["test"])
    sessions = []
    indexes = {}
    for folder, run in search.list_runs(runs_dir):
        origin = run.get("origin") or {}
        speaker = f"{origin.get('dataset')}/{origin.get('speaker')}"
        if speaker not in allowed:
            continue
        clips = _rendered_genes(run)
        if (folder / CHOICE_FILE).is_file():
            record = json.loads((folder / CHOICE_FILE).read_text(encoding="utf-8"))
            heard = [item["clip"] for item in record["heard"]]
            if (record.get("run") != run["id"]
                    or record.get("starting_genome") != run.get("starting_genome", [])
                    or record.get("reference_audio") != run["reference_audio"]
                    or record.get("text") != run["text"]
                    or any(clips.get(item["clip"]) != item["genes"] for item in record["heard"])):
                raise ValueError(f"The recorded shortlist no longer matches {folder}. Record it again.")
            comparisons = [(heard, (record["winner"],))]
        else:
            comparisons = []
            for scores in ratings.load(folder)["scores"].values():
                scores = {key: value for key, value in scores.items() if key in clips}
                if len(scores) < 2:
                    continue
                winners = [key for key, value in scores.items() if value == max(scores.values())]
                if len(winners) < len(scores):
                    comparisons.append((list(scores), tuple(winners)))
        for heard, winners in comparisons:
            if len(set(heard)) < 2 or any(key not in heard for key in winners) or any(key not in clips for key in heard):
                continue
            sessions.append(Session(speaker, _resolve_reference(run, indexes), run.get("starting_genome", []),
                                    [clips[key] for key in heard], tuple(heard.index(key) for key in winners)))
    return sessions


def _voice_embedding(path):
    from backend.metrics.Similarity import speaker_embedding

    vector = np.asarray(speaker_embedding(path), dtype=float)
    if vector.shape != (EMBEDDING_SIZE,) or not np.isfinite(vector).all():
        raise ValueError("Expected a finite 256-dimensional Resemblyzer embedding.")
    return vector


def _pca(sessions, embed):
    paths = sorted({s.reference_audio for s in sessions})
    voices = np.array([embed(path) for path in paths])
    mean = voices.mean(axis=0)
    _, singular, axes = np.linalg.svd(voices - mean, full_matrices=False)
    rank = min(VOICE_DIMENSIONS, int(np.count_nonzero(singular > 1e-8)))
    return mean, axes[:rank]


def _context(start, embedding, mean, axes):
    if mean is None:
        return np.ones(1)
    return np.r_[1.0, axes @ (embedding - mean), start]


def _design(sessions, mean, axes, embed):
    for session in sessions:
        start = np.asarray(genome_vector(session.starting_genome))
        context = _context(start, embed(session.reference_audio) if mean is not None else None, mean, axes)
        delta = np.asarray([genome_vector(genes) for genes in session.choices]) - start
        yield np.einsum("i,jk->jik", context, delta).reshape(len(delta), -1), list(session.winners)


def _fit(designs, penalty):
    # Gradient descent on convex softmax cross-entropy plus L2. A bound on
    # each softmax Hessian supplies a stable step size for every iteration.
    weights = np.zeros(designs[0][0].shape[1])
    curvature = np.mean([np.max(np.sum(x * x, axis=1)) for x, _ in designs])
    rate = 1 / (penalty + curvature)
    for _ in range(1000):
        gradient = penalty * weights
        for x, winners in designs:
            scores = x @ weights
            probabilities = np.exp(scores - scores.max())
            probabilities /= probabilities.sum()
            probabilities[winners] -= 1 / len(winners)
            gradient += (probabilities @ x) / len(designs)
        if np.linalg.norm(gradient) < 1e-7:
            break
        weights -= rate * gradient
    return weights


def _metrics(designs, weights):
    if not designs:
        return None
    correct, losses = [], []
    for x, winners in designs:
        scores = x @ weights
        # Give tied predictions their fraction of credit, as for a random tie break.
        tied = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-12))
        correct.append(len(set(winners).intersection(tied)) / len(tied))
        losses.append(np.logaddexp.reduce(scores) - scores[winners].mean())
    return {"choices": len(designs), "accuracy": float(np.mean(correct)), "log_loss": float(np.mean(losses))}


def _fit_ridge(sessions, mean, axes, embed, penalty):
    """Winner-only baseline: regress the signed winner delta on context."""
    contexts, targets = [], []
    for session in sessions:
        start = np.asarray(genome_vector(session.starting_genome))
        contexts.append(_context(start, embed(session.reference_audio), mean, axes))
        targets.append(np.mean([genome_vector(session.choices[i]) for i in session.winners], axis=0) - start)
    x, y = np.asarray(contexts), np.asarray(targets)
    regularizer = np.eye(x.shape[1]) * (penalty * len(sessions))
    return np.linalg.solve(x.T @ x + regularizer, x.T @ y)


def train(sessions, *, penalty=0.01, validation_fraction=0.2, seed=42, embedding=_voice_embedding):
    """Fit shared and conditioned rankers using a speaker-disjoint holdout."""
    if not sessions:
        raise ValueError("No eligible shortlist choices. Rate at least two clips from a development speaker.")
    if not math.isfinite(penalty) or penalty <= 0 or not 0 <= validation_fraction < 1:
        raise ValueError("Use positive L2 regularization and a validation fraction in [0, 1).")
    speakers = sorted({s.speaker for s in sessions})
    if validation_fraction and len(speakers) < 2:
        raise ValueError("Speaker holdout needs two development speakers. Use --validation-fraction 0 for a pilot.")
    random.Random(seed).shuffle(speakers)
    held_out = set()
    if validation_fraction:
        count = min(len(speakers) - 1, max(1, round(len(speakers) * validation_fraction)))
        held_out = set(speakers[:count])
    training = [s for s in sessions if s.speaker not in held_out]
    validation = [s for s in sessions if s.speaker in held_out]
    cache = {}

    def embed(path):
        if path not in cache:
            cache[path] = np.asarray(embedding(path), dtype=float)
            if cache[path].shape != (EMBEDDING_SIZE,) or not np.isfinite(cache[path]).all():
                raise ValueError(f"Invalid voice embedding for {path}.")
        return cache[path]

    mean, axes = _pca(training, embed)
    shared_train = list(_design(training, None, None, embed))
    shared_valid = list(_design(validation, None, None, embed))
    voice_train = list(_design(training, mean, axes, embed))
    voice_valid = list(_design(validation, mean, axes, embed))
    shared = _fit(shared_train, penalty)
    conditioned = _fit(voice_train, penalty)
    ridge = _fit_ridge(training, mean, axes, embed, penalty)
    return {
        "schema_version": 1, "descriptor_labels": list(LABELS),
        "embedding_model": "resemblyzer", "voice_mean": mean.tolist(), "voice_axes": axes.tolist(),
        "shared": shared.tolist(), "conditioned": conditioned.reshape(-1, SIZE).tolist(),
        "ridge": ridge.tolist(),
        "training": {"seed": seed, "penalty": penalty, "validation_fraction": validation_fraction,
                     "choices": len(training), "tied_choices": sum(len(s.winners) > 1 for s in training),
                     "train_speakers": sorted(set(speakers) - held_out),
                     "validation_speakers": sorted(held_out)},
        "validation": {
            "uniform": _metrics(shared_valid, np.zeros(SIZE)),
            "shared": _metrics(shared_valid, shared),
            "conditioned": _metrics(voice_valid, conditioned),
            "winner_delta_ridge": _metrics(voice_valid, ridge.ravel()),
        },
    }


def load_model(path=None):
    path = Path(path) if path is not None else model_path()
    if not path.is_file():
        raise ValueError(f"No preference model at {path}. Train one with python -m backend.genome.preference train.")
    raw = path.read_bytes()
    data = json.loads(raw)
    if (not isinstance(data, dict) or data.get("schema_version") != 1 or data.get("descriptor_labels") != list(LABELS)
            or data.get("embedding_model") != "resemblyzer"):
        raise ValueError("Preference model has an incompatible feature schema. Retrain it.")
    data["sha256"] = hashlib.sha256(raw).hexdigest()
    return data


def direction(artifact, starting_genome, reference_audio=None, *, conditioned=True):
    """Predict the score gradient in descriptor space for this voice and seed."""
    if not conditioned:
        result = np.asarray(artifact["shared"], dtype=float)
    else:
        axes = np.asarray(artifact["voice_axes"], dtype=float).reshape(-1, EMBEDDING_SIZE)
        mean = np.asarray(artifact["voice_mean"], dtype=float)
        context = _context(np.asarray(genome_vector(starting_genome)), _voice_embedding(reference_audio), mean, axes)
        result = context @ np.asarray(artifact["conditioned"], dtype=float)
    if result.shape != (SIZE,) or not np.isfinite(result).all():
        raise ValueError("Preference model returned an invalid direction.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    choose = commands.add_parser("choose", help="record a winner and the clips the listener heard")
    choose.add_argument("run", type=Path)
    choose.add_argument("--heard", nargs="+", required=True)
    choose.add_argument("--winner", required=True)
    fit = commands.add_parser("train", help="fit shared and voice-conditioned rankers")
    fit.add_argument("--runs-dir", type=Path)
    fit.add_argument("--output", type=Path, default=model_path())
    fit.add_argument("--penalty", type=float, default=0.01)
    fit.add_argument("--validation-fraction", type=float, default=0.2)
    fit.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        if args.command == "choose":
            record = record_choice(args.run, args.heard, args.winner)
            print(f"Recorded {record['winner']} among {len(record['heard'])} clips in {args.run / CHOICE_FILE}")
        else:
            sessions = collect_sessions(args.runs_dir)
            print(f"Loaded {len(sessions)} choices from {len({s.speaker for s in sessions})} speakers.", flush=True)
            artifact = train(sessions, penalty=args.penalty, validation_fraction=args.validation_fraction, seed=args.seed)
            _write_json(args.output, artifact)
            print(json.dumps({"training": artifact["training"], "validation": artifact["validation"]}, indent=2))
            print(f"Saved {args.output}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()

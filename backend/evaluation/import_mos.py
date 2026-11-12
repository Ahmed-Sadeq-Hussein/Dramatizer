"""Log a saved listening batch as a descriptive paired comparison."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from checks import EVAL_DIR, SPLIT, sha256

OBJECTIVE = (
    "For each listener and run, subtract the plain clip score from the mean score "
    "of the genome clips that listener rated. Average comparisons within each "
    "speaker and target line, then weight speaker-line cases equally."
)
LIMITATION = (
    "Descriptive comparison of the rated shortlist with the plain clip. "
    "This does not evaluate a system that selects a single genome before listening. "
    "The same listener rates both sides of each comparison."
)


def summarize(export):
    root = export / "runs" if (export / "runs").is_dir() else export
    folders = sorted(root.glob("*/ratings.json"))
    if not folders:
        raise ValueError(f"No run folders with ratings.json in {root}.")

    inputs, pairs, listener_ids = {}, [], {}
    cases = defaultdict(list)
    methods = set()
    skipped = 0
    for ratings_path in folders:
        run_path = ratings_path.with_name("run.json")
        for path in (run_path, ratings_path):
            inputs[path.relative_to(root).as_posix()] = sha256(path)
        run = json.loads(run_path.read_text(encoding="utf-8-sig"))
        ratings = json.loads(ratings_path.read_text(encoding="utf-8-sig"))
        origin = run.get("origin") or {}
        if not origin.get("dataset") or not origin.get("speaker") or not run.get("text"):
            raise ValueError(f"Missing speaker identity or target line in {run_path}.")
        speaker = f"{origin['dataset']}/{origin['speaker']}"
        clips = {"baseline"} if run["baseline"].get("audio") else set()
        clips.update(f"genome_{g['number']:02d}" for g in run["genomes"] if g.get("audio"))
        for listener, scores in sorted(ratings["scores"].items()):
            for clip, score in scores.items():
                if (clip not in clips or type(score) not in (int, float) or not 0 <= score <= 5):
                    raise ValueError(f"Invalid clip or score for {clip!r} in {ratings_path}.")
            candidates = {key: value for key, value in scores.items() if key != "baseline"}
            if "baseline" not in scores or not candidates:
                skipped += 1
                continue
            baseline = scores["baseline"]
            candidate_mean = mean(candidates.values())
            listener_id = listener_ids.setdefault(listener, f"listener_{len(listener_ids) + 1:02d}")
            case = (speaker, run["text"])
            cases[case].append((baseline, candidate_mean))
            methods.add(run.get("method", "unknown"))
            pairs.append({
                "run": run["id"], "speaker": speaker, "reference_clip": origin.get("clip_id"),
                "text": run["text"], "listener": listener_id, "method": run.get("method"),
                "render_seed": run.get("render_seed"), "scores": scores,
                "delta": candidate_mean - baseline,
            })

    if not pairs:
        raise ValueError("No listener rated both a plain clip and at least one genome in the same run.")
    baselines = [mean(b for b, _ in case) for case in cases.values()]
    candidates = [mean(g for _, g in case) for case in cases.values()]
    deltas = [g - b for b, g in zip(baselines, candidates)]
    metrics = {
        "mean_plain_rating": mean(baselines),
        "mean_genome_rating": mean(candidates),
        "mean_paired_delta": mean(deltas),
        "speaker_line_cases": len(cases),
        "speakers": len({speaker for speaker, _ in cases}),
        "listeners": len(listener_ids),
        "paired_listener_runs": len(pairs),
        "scores_used": sum(len(pair["scores"]) for pair in pairs),
        "unpaired_listener_runs": skipped,
    }
    digest = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    return {
        "objective": OBJECTIVE, "limitations": LIMITATION, "methods": sorted(methods),
        "inputs_sha256": digest, "input_files": inputs, "metrics": metrics, "pairs": pairs,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path, help="listening export folder or its runs directory")
    parser.add_argument("--name", help="MLflow run name")
    args = parser.parse_args()
    try:
        report = summarize(args.export)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Cannot import ratings: {error}\n")

    from track import log_run

    batch = args.export.parent.name if args.export.name == "runs" else args.export.name
    folder = EVAL_DIR / "mos" / report["inputs_sha256"]
    folder.mkdir(parents=True, exist_ok=True)
    artifact = folder / "mos-comparison.json"
    artifact.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    run_id = log_run(
        args.name or f"mos-{batch}",
        params={"batch": batch, "methods": ", ".join(report["methods"]),
                "objective": OBJECTIVE, "rating_scale": "0–5", "inputs_sha256": report["inputs_sha256"],
                "project_split_sha256": sha256(SPLIT)},
        metrics=report["metrics"], artifacts=[artifact], limitations=LIMITATION,
    )
    print(json.dumps(report["metrics"], indent=2))
    print(f"MLflow run ID: {run_id}")


if __name__ == "__main__":
    main()

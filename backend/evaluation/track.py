# backend/evaluation/track.py
"""track.py | Record evaluation runs with MLflow.

Runs are stored in history/eval (gitignored). See them with:
    mlflow ui --backend-store-uri sqlite:///history/eval/mlflow.db --port 5050
"""
import csv
import json
import subprocess
import sys

import mlflow

from checks import EVAL_DIR, MANIFEST, SPLIT, run_checks, sha256

DB_PATH = EVAL_DIR / "mlflow.db"
ARTIFACT_DIR = EVAL_DIR / "mlartifacts"
EXPERIMENT = "dramatizer-evaluation"

EVAL_DIR.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH.as_posix()}")
if mlflow.get_experiment_by_name(EXPERIMENT) is None:
    mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_DIR.as_uri())
mlflow.set_experiment(EXPERIMENT)


def log_environment():
    """The installed package versions, so a run can be repeated."""
    out = EVAL_DIR / "environment.txt"
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    out.write_text(freeze.stdout, encoding="utf-8")
    return out


def log_run(name, params, metrics, artifacts=(), limitations=""):
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for path in list(artifacts) + [log_environment()]:
            mlflow.log_artifact(str(path))
        mlflow.set_tag("limitations", limitations)


if __name__ == "__main__":
    with MANIFEST.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    results = run_checks(rows, split)

    log_run(
        name="data-checks",
        params={
            "manifest_sha256": sha256(MANIFEST),
            "split_sha256": sha256(SPLIT),
            "split_seed": 42,
            "test_fraction": 0.2,
            "validation_speakers": len(split["validation"]),
            "test_speakers": len(split["test"]),
        },
        metrics={
            "checks_passed": sum(1 for _, ok in results if ok),
            "checks_total": len(results),
        },
        artifacts=[SPLIT],
        limitations="Checks the data and the split only. No model has been evaluated yet.",
    )
    print(f"logged data-checks: {sum(ok for _, ok in results)}/{len(results)} checks passed")

"""measurements.py | Every measurement of a recording, kept, so nothing is measured twice.

Measuring a clone (reward.clone_metrics) loads Whisper, Resemblyzer and SQUIM and
takes seconds per clip. This file keeps what was measured, so the same clip with
the same inputs is never measured again, and every run stays on record.

    from backend.metrics.measurements import measure

    result = measure("reference.wav", "clone.wav", "the line", reference_text="...")
    result["metrics"]     # what reward.clone_metrics returns
    result["cached"]      # True: it had been measured before, nothing was run
    result["node"]        # the file that holds this recording's measurements

Each recording gets one node: MEASUREMENTS_DIR/<sha256 of the audio>.json. The
name comes from the audio's bytes, so the node is found again even if the file
is moved or renamed, and a changed recording gets a new node. A node holds:

    audio          where the recording was when it was last measured
    audio_sha256   the recording's checksum
    runs           every measurement, oldest first: its inputs, its metrics, when

A run is reused only when its inputs are the same: the reference recording
(by checksum), the text, the reference text, the measuring models and the
reward settings (RewardConfig). Change any of them and it is measured again.
force=True always measures again, and keeps the old runs.

Every new run is also added as one line to MEASUREMENTS_DIR/log.jsonl, a running
log of all measurements, for tracking the metrics over time.

MEASUREMENTS_DIR is "measurements" in the project (MEASUREMENTS_DIR in .env to
change it). It is never committed (.gitignore): it holds local paths and
measurements of real people's cloned voices.
"""
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

from backend import model
from backend.metrics.reward import CONFIG, clone_metrics, measuring_models

MEASUREMENTS_DIR = model.PROJECT_DIR / model.setting("MEASUREMENTS_DIR", "measurements")
LOG_NAME = "log.jsonl"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def shown(path):
    """A path relative to the project when it is inside it, so nodes do not depend on this PC's folders."""
    path = Path(path).resolve()
    try:
        return path.relative_to(model.PROJECT_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def node_path(audio):
    """Where the node of this recording is (it may not exist yet)."""
    return MEASUREMENTS_DIR / f"{file_sha256(audio)}.json"


def load_node(audio=None, node=None):
    """A recording's node as a dict, or None if it was never measured. Give the audio or the node's path."""
    path = Path(node) if node else node_path(audio)
    if not path.is_absolute():
        path = model.PROJECT_DIR / path
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def latest(audio=None, node=None):
    """The newest run of a recording ({"inputs", "metrics", "measured"}), or None."""
    data = load_node(audio, node)
    return data["runs"][-1] if data and data["runs"] else None


def inputs_of(reference_audio, text, reference_text=None, config=CONFIG):
    """Everything a measurement depends on, besides the recording itself."""
    settings = json.dumps(asdict(config), sort_keys=True, default=str)
    return {
        "reference": shown(reference_audio),
        "reference_sha256": file_sha256(reference_audio),
        "text": text,
        "reference_text": reference_text,
        "models": measuring_models(),
        "config_sha256": hashlib.sha256(settings.encode("utf-8")).hexdigest(),
    }


def _write(path, data):
    """Write the whole file at once, so a crash never leaves half a node."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def measure(reference_audio, clone_audio, text, reference_text=None, force=False, config=CONFIG):
    """
    The metrics of a clone against its reference: from its node when it was already
    measured with the same inputs, otherwise measured now and added to the node and the log.

    Returns {"metrics", "cached", "node", "runs"}: the metrics, whether they came from
    the node, the node's path (relative to the project) and how many runs it now holds.
    """
    sha = file_sha256(clone_audio)
    path = MEASUREMENTS_DIR / f"{sha}.json"
    data = load_node(node=path) or {"audio": shown(clone_audio), "audio_sha256": sha, "runs": []}
    inputs = inputs_of(reference_audio, text, reference_text, config)

    if not force:
        for run in reversed(data["runs"]):
            if run["inputs"] == inputs:
                return {"metrics": run["metrics"], "cached": True, "node": shown(path), "runs": len(data["runs"])}

    metrics = clone_metrics(reference_audio, clone_audio, text, reference_text, config)
    data["audio"] = shown(clone_audio)
    data["runs"].append({"inputs": inputs, "metrics": metrics, "measured": metrics["measured"]})
    _write(path, data)

    line = {
        "measured": metrics["measured"],
        "audio": data["audio"],
        "audio_sha256": sha,
        "node": shown(path),
        "reference": inputs["reference"],
        "wer": metrics["wer"],
        "similarity": metrics["similarity"],
        "pesq": metrics["quality"]["pesq"],
        "words_per_second": metrics["words_per_second"],
        "models": inputs["models"],
    }
    with (MEASUREMENTS_DIR / LOG_NAME).open("a", encoding="utf-8") as log:
        log.write(json.dumps(line, ensure_ascii=False) + "\n")

    return {"metrics": metrics, "cached": False, "node": shown(path), "runs": len(data["runs"])}

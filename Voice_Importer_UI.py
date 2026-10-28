"""Voice_UI.py | A browser interface for the voice library: filter clips, play them, read the script.

It uses the functions in Voice_Importer.py and adds nothing to them: this file
only turns their results into web pages and JSON.

    python Voice_UI.py
    then open http://127.0.0.1:5000

The folders come from .env:

    VOICE_LIBRARY_DIR   the folder that holds one sub-folder per dataset
    CREMA_D_FOLDER      the CREMA-D sub-folder (default CREMA-D)
    VCTK_FOLDER         the VCTK sub-folder (default VCTK)

A dataset whose folder is missing is left out, and the page shows the rest.

The server only listens on this computer (127.0.0.1). Audio is served by clip
id from the index built at start-up, never by a path from the browser.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, abort, jsonify, render_template, request, send_file

from Voice_Importer import find_clips, get_clip, list_speakers, read_crema_d, read_vctk

PROJECT_DIR = Path(__file__).resolve().parent

load_dotenv(PROJECT_DIR / ".env")

LIBRARY_DIR = Path(os.environ.get("VOICE_LIBRARY_DIR", "voice_library"))

# name shown on the page: (reader, folder inside the library, detail filters to offer)
DATASETS = {
    "CREMA-D": (read_crema_d, os.environ.get("CREMA_D_FOLDER", "CREMA-D"), ["emotion", "intensity"]),
    "VCTK": (read_vctk, os.environ.get("VCTK_FOLDER", "VCTK"), ["mic"]),
}

PAGE_SIZE = 50

app = Flask(__name__)

# name -> {"clips": [...], "by_id": {clip_id: Clip}, "filters": {...}}
library = {}


def load_library():
    """Read every dataset folder that exists. Takes a few seconds, once."""
    for name, (reader, folder, filter_keys) in DATASETS.items():
        path = LIBRARY_DIR / folder

        if not path.is_dir():
            print(f"{name}: folder not found, left out ({path})")
            continue

        clips = reader(path)
        library[name] = {
            "clips": clips,
            "by_id": {clip.clip_id: clip for clip in clips},
            "filters": {
                key: sorted({str(clip.details[key]) for clip in clips if key in clip.details})
                for key in filter_keys
            },
        }
        print(f"{name}: {len(clips)} clips from {len(list_speakers(clips))} speakers")


def dataset_or_404(name):
    if name not in library:
        abort(404, f"Unknown dataset {name!r}")
    return library[name]


def clip_json(clip):
    """What the page needs to know about one clip."""
    return {
        "id": clip.clip_id,
        "dataset": clip.dataset,
        "speaker": clip.speaker,
        "text": clip.text,
        "details": clip.details,
        "audio": f"/audio/{clip.dataset}/{clip.clip_id}.wav",
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/datasets")
def api_datasets():
    return jsonify([
        {
            "name": name,
            "clips": len(data["clips"]),
            "speakers": list_speakers(data["clips"]),
            "filters": data["filters"],
        }
        for name, data in library.items()
    ])


@app.route("/api/clips")
def api_clips():
    """Clips of one dataset, filtered, one page at a time."""
    data = dataset_or_404(request.args.get("dataset", ""))

    details = {
        key: request.args[key]
        for key in data["filters"]
        if request.args.get(key)
    }
    found = find_clips(
        data["clips"],
        speaker=request.args.get("speaker") or None,
        text=request.args.get("text") or None,
        **details,
    )

    pages = max(1, -(-len(found) // PAGE_SIZE))
    page = min(max(request.args.get("page", 1, type=int), 1), pages)
    start = (page - 1) * PAGE_SIZE

    return jsonify({
        "total": len(found),
        "page": page,
        "pages": pages,
        "clips": [clip_json(clip) for clip in found[start:start + PAGE_SIZE]],
    })


@app.route("/api/clip/<dataset>/<clip_id>")
def api_clip(dataset, clip_id):
    """One clip: its script, details and where its audio file is on disk."""
    clip = dataset_or_404(dataset)["by_id"].get(clip_id)

    if clip is None:
        abort(404, f"No clip {clip_id!r} in {dataset}")

    path, _ = get_clip(clip)

    return jsonify({**clip_json(clip), "file": str(path)})


@app.route("/audio/<dataset>/<clip_id>.wav")
def audio(dataset, clip_id):
    """The clip as a wav file. VCTK clips are exported from their parquet file first."""
    clip = dataset_or_404(dataset)["by_id"].get(clip_id)

    if clip is None:
        abort(404, f"No clip {clip_id!r} in {dataset}")

    path, _ = get_clip(clip)

    return send_file(path, mimetype="audio/wav")


if __name__ == "__main__":
    load_library()

    if not library:
        raise SystemExit(f"No dataset folders found in {LIBRARY_DIR}. Check VOICE_LIBRARY_DIR in .env.")

    # debug is off: the debugger would let anyone who reaches the page run code
    app.run(host="127.0.0.1", port=5000, debug=False)

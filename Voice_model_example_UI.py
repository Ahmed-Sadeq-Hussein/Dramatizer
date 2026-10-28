"""Voice_model_example_UI.py | Pick a voice from the library, then make it say something new.

The same page as Voice_Importer_UI.py (filter clips, play them, read the script), with a
panel added: the clip you picked becomes the reference voice, you type what it
should say, and CosyVoice 3 speaks it through generate_speech() in Voice_model.py.

    python Voice_model_example_UI.py
    then open http://127.0.0.1:5001

Needs the CosyVoice 3 setup of Voice_model.py (see .env) and a GPU for speed.

Generating is slow, so it runs as a background job: the page starts one, then
asks every second how it is going. The model loads on the first job (about a
minute) and stays loaded for the rest. Only one job runs at a time.

Results are written by generate_speech to OUTPUT_DIR/<voice name>/, and next to
each wav this file saves a .json with the text, the reference clip and the time.
"""
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

import Voice_Importer_UI as library_ui
import Voice_model
import model
from Voice_Importer import get_clip

OUTPUT_DIR = Path(Voice_model.OUTPUT_DIR)

MAX_TEXT_CHARS = 600
KEEP_JOBS = 50

app = Flask(__name__)

# The browsing routes are the ones from Voice_Importer_UI.py, shared as they are
for rule, view in [
    ("/api/datasets", library_ui.api_datasets),
    ("/api/clips", library_ui.api_clips),
    ("/api/clip/<dataset>/<clip_id>", library_ui.api_clip),
    ("/audio/<dataset>/<clip_id>.wav", library_ui.audio),
]:
    app.add_url_rule(rule, view_func=view)

# One worker: the GPU runs one generation at a time, the rest wait their turn
worker = ThreadPoolExecutor(max_workers=1)
jobs = {}
jobs_lock = threading.Lock()


def set_job(job_id, **changes):
    with jobs_lock:
        jobs[job_id].update(changes)


def run_job(job_id, clip, target_text, voice_name):
    """Runs on the worker thread: generate the speech and record the result."""
    try:
        reference_audio, reference_text = get_clip(clip)

        # The first job also loads the model, which takes much longer
        set_job(job_id, status="generating" if model.is_loaded("voice") else "loading model")
        started = time.time()

        path = Voice_model.generate_speech(reference_audio, reference_text, target_text, voice_name)

        record = {
            "target_text": target_text,
            "reference_clip": f"{clip.dataset}/{clip.clip_id}",
            "reference_text": reference_text,
            "voice_name": path.parent.name,
            "seconds": round(time.time() - started, 1),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        path.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")

        set_job(job_id, status="done", result=generated_json(path))

    except Exception as error:
        # Shown on the page; the full traceback is in this terminal
        import traceback
        traceback.print_exc()
        set_job(job_id, status="error", error=f"{type(error).__name__}: {error}")


def generated_json(path):
    """What the page needs to know about one generated wav."""
    relative = path.relative_to(OUTPUT_DIR).as_posix()
    info_file = path.with_suffix(".json")
    info = json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() else {}

    return {
        "file": path.name,
        "voice": path.parent.name,
        "audio": f"/generated/{relative}",
        "text": info.get("target_text", ""),
        "reference": info.get("reference_clip", ""),
        "created": info.get("created", ""),
        "seconds": info.get("seconds"),
    }


@app.route("/")
def index():
    return render_template("generate.html")


@app.route("/api/generate", methods=["POST"])
def api_generate():
    """Start a job: speak `text` in the voice of the library clip `clip_id`."""
    body = request.get_json(silent=True) or {}

    dataset = library_ui.dataset_or_404(body.get("dataset", ""))
    clip = dataset["by_id"].get(body.get("clip_id", ""))
    text = (body.get("text") or "").strip()

    if clip is None:
        return jsonify(error="Pick a reference clip first."), 400

    # Without a name, the output folder is named after the reference speaker
    name = (body.get("name") or "").strip() or clip.speaker

    if not text:
        return jsonify(error="Type the text to speak."), 400
    if len(text) > MAX_TEXT_CHARS:
        return jsonify(error=f"The text is {len(text)} characters; the limit is {MAX_TEXT_CHARS}."), 400

    # The name becomes a folder: letters, digits, dash and underscore only
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:60].strip("._")
    if not safe_name:
        return jsonify(error="Give the voice a name made of letters and digits."), 400

    job_id = uuid.uuid4().hex[:10]

    with jobs_lock:
        jobs[job_id] = {"id": job_id, "status": "queued", "created": time.time()}
        for old in sorted(jobs, key=lambda j: jobs[j]["created"])[:-KEEP_JOBS]:
            del jobs[old]

    worker.submit(run_job, job_id, clip, text, safe_name)

    return jsonify(id=job_id), 202


@app.route("/api/job/<job_id>")
def api_job(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            abort(404, "Unknown job")
        return jsonify({**job, "waited": round(time.time() - job["created"])})


@app.route("/api/generated")
def api_generated():
    """The most recent generated clips, newest first."""
    wavs = sorted(OUTPUT_DIR.glob("*/*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
    return jsonify([generated_json(path) for path in wavs[:30]])


@app.route("/generated/<voice>/<file>")
def generated(voice, file):
    """A generated wav. The path is checked to stay inside OUTPUT_DIR."""
    path = (OUTPUT_DIR / voice / file).resolve()

    if OUTPUT_DIR.resolve() not in path.parents or path.suffix != ".wav" or not path.is_file():
        abort(404)

    return send_file(path, mimetype="audio/wav")


if __name__ == "__main__":
    library_ui.load_library()

    if not library_ui.library:
        raise SystemExit(f"No dataset folders found in {library_ui.LIBRARY_DIR}. Check VOICE_LIBRARY_DIR in .env.")

    # Not 5000, so this can run beside Voice_Importer_UI.py. The debugger stays off.
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)

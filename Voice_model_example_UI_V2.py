"""Voice_model_example_UI_V2.py | Pick a voice with Picky_voice.py's options, then make it say something new.

One self-contained Flask file: the page is plain HTML forms, and Python does all
the work (no JavaScript, no separate template). Every setting lives in the URL,
so a selection can be bookmarked or shared, and the browser's Back button works.

    python Voice_model_example_UI_V2.py
    then open http://127.0.0.1:5002

How the page works:

1. Picky options (from Picky_voice.py, see its top for why each exists):
       Picky             off shows the whole dataset
       Min / max length  speech kept, in seconds (default 3 to 25)
       Duplicate actor   off keeps only each speaker's longest clip
       Emotions          off keeps neutral clips only
   then the usual filters (speaker, script, emotion...), then "Apply".
2. Play any clip in the list; "Use" makes it the reference voice.
3. Type the text and press Generate. CosyVoice 3 speaks it through
   Voice_model.generate_speech. The page waits until it is done (the first
   time also loads the model, about a minute).
4. "Export references" writes the longest picked clip of each speaker to
   output/references/<dataset>/<speaker>/reference.wav + reference.txt.

Reused, not copied: the library and audio routes of Voice_Importer_UI.py and
the generate job and output routes of Voice_model_example_UI.py.
"""
import threading
import time
import uuid
from functools import lru_cache
from urllib.parse import urlencode

from flask import Flask, redirect, render_template_string, request

import Voice_Importer_UI as library_ui
import Voice_model_example_UI as generator_ui
import model
from Picky_voice import (DUPLICATE_ACTOR, EMOTIONS, MAX_SECONDS, MIN_SECONDS,
                         export_references, measure, select_clips)
from Voice_Importer import find_clips, list_speakers

PORT = 5002
REFERENCES_DIR = model.PROJECT_DIR / "output" / "references"

app = Flask(__name__)

# Audio of library clips and of generated clips, served by the V1 code as it is
app.add_url_rule("/audio/<dataset>/<clip_id>.wav", view_func=library_ui.audio)
app.add_url_rule("/generated/<voice>/<file>", view_func=generator_ui.generated)

# dataset name -> {clip_id: seconds of speech}, filled at start-up
seconds = {}

# The GPU makes one voice at a time
generate_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Reading the settings from the URL
# --------------------------------------------------------------------------- #


def read_settings(args):
    """
    Everything the page shows, from the query string. A form sends unticked
    boxes as nothing at all, so the hidden field "sent" tells "unticked" apart
    from "first visit, use the defaults".
    """
    sent = "sent" in args

    def box(name, default):
        return (name in args) if sent else default

    def number(name, default):
        try:
            return float(args.get(name) or default)
        except ValueError:
            return default

    dataset = args.get("dataset") if args.get("dataset") in library_ui.library else next(iter(library_ui.library))

    return {
        "dataset": dataset,
        "picky": box("picky", True),
        "min": number("min", MIN_SECONDS),
        "max": number("max", MAX_SECONDS),
        "duplicate_actor": box("duplicate_actor", DUPLICATE_ACTOR),
        "emotions": box("emotions", EMOTIONS),
        "speaker": args.get("speaker", ""),
        "text": args.get("text", "").strip(),
        "details": {key: args.get(key, "") for key in library_ui.library[dataset]["filters"]},
        "page": max(args.get("page", 1, type=int), 1),
        "clip": args.get("clip", ""),
    }


def query_string(settings, **changes):
    """The URL query for these settings, with some of them changed."""
    s = {**settings, **changes}
    params = {"sent": 1, "dataset": s["dataset"], "min": f"{s['min']:g}", "max": f"{s['max']:g}"}

    for name in ("picky", "duplicate_actor", "emotions"):
        if s[name]:
            params[name] = 1
    for name in ("speaker", "text", "clip"):
        if s[name]:
            params[name] = s[name]
    params.update({key: value for key, value in s["details"].items() if value})
    if s["page"] > 1:
        params["page"] = s["page"]

    return urlencode(params)


# --------------------------------------------------------------------------- #
# Choosing the clips
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=32)
def picked(dataset, min_seconds, max_seconds, duplicate_actor, emotions):
    """select_clips for one dataset and one set of options, remembered."""
    clips = library_ui.library[dataset]["clips"]
    return tuple(select_clips(clips, min_seconds, max_seconds, duplicate_actor, emotions))


def picky_clips(settings):
    """The clips Picky keeps (all of them when Picky is off)."""
    if not settings["picky"]:
        return library_ui.library[settings["dataset"]]["clips"]
    return list(picked(settings["dataset"], settings["min"], settings["max"],
                       settings["duplicate_actor"], settings["emotions"]))


def length(clip):
    return seconds.get(clip.dataset, {}).get(clip.clip_id)


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #


@app.route("/")
def index():
    settings = read_settings(request.args)
    data = library_ui.library[settings["dataset"]]
    all_clips = data["clips"]
    error = request.args.get("error", "")

    if settings["picky"] and settings["min"] > settings["max"]:
        error = error or f"Min length ({settings['min']:g} s) is above max length ({settings['max']:g} s)."
        clips = []
    else:
        clips = picky_clips(settings)

    found = find_clips(
        clips,
        speaker=settings["speaker"] or None,
        text=settings["text"] or None,
        **{key: value for key, value in settings["details"].items() if value},
    )

    size = library_ui.PAGE_SIZE
    pages = max(1, -(-len(found) // size))
    page = min(settings["page"], pages)
    settings["page"] = page

    kept = list_speakers(clips)
    everyone = list_speakers(all_clips)

    reference = data["by_id"].get(settings["clip"])
    recent = [generator_ui.generated_json(path) for path in
              sorted(generator_ui.OUTPUT_DIR.glob("*/*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]]

    return render_template_string(
        PAGE,
        s=settings,
        q=lambda **changes: query_string(settings, **changes),
        datasets=list(library_ui.library),
        filters=data["filters"],
        rows=found[(page - 1) * size: page * size],
        total=len(found),
        page=page,
        pages=pages,
        picked_count=len(clips),
        all_count=len(all_clips),
        kept=kept,
        everyone=everyone,
        left_out=sorted(set(everyone) - set(kept)),
        length=length,
        reference=reference,
        result=request.args.get("result", ""),
        message=request.args.get("message", ""),
        error=error,
        recent=recent,
    )


@app.route("/generate", methods=["POST"])
def generate():
    """Speak the text in the reference clip's voice, then go back to the page with the result."""
    settings = read_settings(request.form)
    back = query_string(settings)
    clip = library_ui.library[settings["dataset"]]["by_id"].get(settings["clip"])
    text = request.form.get("target", "").strip()

    if clip is None:
        return redirect(f"/?{back}&error=Pick+a+reference+clip+first.")
    if not text:
        return redirect(f"/?{back}&error=Type+the+text+to+speak.")
    if len(text) > generator_ui.MAX_TEXT_CHARS:
        return redirect(f"/?{back}&" + urlencode({"error": f"The text is longer than {generator_ui.MAX_TEXT_CHARS} characters."}))

    # The name becomes a folder: letters, digits, dash and underscore only
    name = request.form.get("name", "").strip() or clip.speaker
    name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:60].strip("._") or clip.speaker

    # Run V1's job function right here, so it writes the same wav + .json
    job_id = uuid.uuid4().hex[:10]
    with generator_ui.jobs_lock:
        generator_ui.jobs[job_id] = {"id": job_id, "status": "queued", "created": time.time()}
    with generate_lock:
        generator_ui.run_job(job_id, clip, text, name)
    with generator_ui.jobs_lock:
        job = generator_ui.jobs.pop(job_id)

    if job["status"] != "done":
        return redirect(f"/?{back}&" + urlencode({"error": job.get("error", "Generating failed.")}))

    return redirect(f"/?{back}&" + urlencode({"result": job["result"]["audio"]}))


@app.route("/export", methods=["POST"])
def export():
    """Write the longest picked clip of each speaker to output/references/<dataset>/."""
    settings = read_settings(request.form)
    back = query_string(settings)

    if not settings["picky"]:
        return redirect(f"/?{back}&error=Switch+Picky+on+first:+export+uses+the+picked+clips.")
    if settings["min"] > settings["max"]:
        return redirect(f"/?{back}&error=Min+length+is+above+max+length.")

    selected = picky_clips(settings)
    if not selected:
        return redirect(f"/?{back}&error=No+clips+are+picked+with+these+options.")

    folder = export_references(selected, REFERENCES_DIR / settings["dataset"])
    speakers = len({clip.speaker for clip in selected})

    return redirect(f"/?{back}&" + urlencode({"message": f"Exported {speakers} references to {folder}"}))


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice Generator V2</title>
<link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
<style>
  .tabs a { padding: 6px 12px; border-radius: 6px; color: var(--muted); text-decoration: none; }
  .tabs a.on { background: var(--accent-soft); color: var(--accent); font-weight: 600; }
  .version { font-size: 0.75rem; color: var(--accent); background: var(--accent-soft); padding: 2px 7px; border-radius: 999px; vertical-align: middle; }
  .picky { display: flex; flex-wrap: wrap; align-items: end; gap: 10px 14px; padding: 10px 12px; margin-bottom: 10px;
           background: var(--band); border: 1px solid var(--rule); border-radius: 8px; }
  .picky label, form.filters label { display: grid; gap: 3px; font-size: 0.8rem; color: var(--muted); }
  .picky input[type=number] { width: 6.5em; min-width: 0; }
  .picky label.switch { display: flex; align-items: center; gap: 6px; font-size: 0.88rem; color: var(--ink); padding-bottom: 6px; }
  .picky label.switch input { width: auto; min-width: 0; padding: 0; margin: 0; accent-color: var(--accent); }
  .picky .summary { flex-basis: 100%; margin: 0; font-size: 0.82rem; color: var(--muted); }
  .row-filters { flex-basis: 100%; display: flex; flex-wrap: wrap; align-items: end; gap: 10px 14px; }
  td audio { width: 210px; height: 32px; }
  td.use { white-space: nowrap; }
  tbody tr { cursor: default; }
  .note { margin: 0 0 10px; padding: 8px 10px; border-radius: 6px; font-size: 0.9rem; background: var(--accent-soft); color: var(--accent); overflow-wrap: anywhere; }
  .note.error { background: #f9e1df; color: #9c2b22; }
  @media (prefers-color-scheme: dark) { .note.error { background: #3a1c1a; color: #f08a80; } }
  .recent li { display: grid; gap: 4px; }
  .recent audio { height: 32px; }
  a.button { display: inline-block; padding: 3px 10px; border: 1px solid var(--rule); border-radius: 6px; color: var(--accent); text-decoration: none; }
</style>
</head>
<body>
<header>
  <h1>Voice Generator <span class="version">V2</span></h1>
  <nav class="tabs" aria-label="Dataset">
    {% for name in datasets %}
      <a href="/?{{ q(dataset=name, speaker='', clip='', page=1, details={}) }}" class="{{ 'on' if name == s.dataset }}">{{ name }}</a>
    {% endfor %}
  </nav>
</header>

<main>
<section aria-label="Clips">
  {% if error %}<p class="note error">{{ error }}</p>{% endif %}
  {% if message %}<p class="note">{{ message }}</p>{% endif %}

  <form method="get" action="/" class="picky" autocomplete="off">
    <input type="hidden" name="sent" value="1">
    <input type="hidden" name="dataset" value="{{ s.dataset }}">
    {% if s.clip %}<input type="hidden" name="clip" value="{{ s.clip }}">{% endif %}

    <label class="switch"><input type="checkbox" name="picky" {{ 'checked' if s.picky }}> <strong>Picky</strong></label>
    <label>Min length (s) <input type="number" name="min" min="0" max="30" step="0.5" value="{{ '%g' % s.min }}"></label>
    <label>Max length (s) <input type="number" name="max" min="0" max="30" step="0.5" value="{{ '%g' % s.max }}"></label>
    <label class="switch" title="Off: only each speaker's longest clip">
      <input type="checkbox" name="duplicate_actor" {{ 'checked' if s.duplicate_actor }}> Duplicate actor</label>
    <label class="switch" title="Off: neutral clips only">
      <input type="checkbox" name="emotions" {{ 'checked' if s.emotions }}> Emotions</label>

    <div class="row-filters">
      <label>Speaker
        <select name="speaker">
          <option value="">All speakers</option>
          {% for id in everyone %}
            <option value="{{ id }}" {{ 'selected' if id == s.speaker }}>{{ id }} ({{ kept[id] if id in kept else 'left out' }})</option>
          {% endfor %}
        </select>
      </label>
      <label>Script contains <input type="search" name="text" value="{{ s.text }}" placeholder="e.g. jacket"></label>
      {% for key, values in filters.items() %}
        <label>{{ key|capitalize }}
          <select name="{{ key }}">
            <option value="">Any {{ key }}</option>
            {% for value in values %}<option {{ 'selected' if value == s.details[key] }}>{{ value }}</option>{% endfor %}
          </select>
        </label>
      {% endfor %}
      <button type="submit">Apply</button>
      <a class="button" href="/?dataset={{ s.dataset }}">Defaults</a>
      <button type="submit" formmethod="post" formaction="/export" class="quiet"
              title="Write the longest picked clip of each speaker to output/references/{{ s.dataset }}/">Export references</button>
    </div>

    <p class="summary">
      {% if s.picky %}
        Picked {{ '{:,}'.format(picked_count) }} of {{ '{:,}'.format(all_count) }} clips,
        {{ kept|length }} of {{ everyone|length }} speakers.
        {% if left_out %}Left out: {{ left_out[:12]|join(', ') }}{% if left_out|length > 12 %} and {{ left_out|length - 12 }} more{% endif %}.{% endif %}
      {% else %}
        Picky is off: all {{ '{:,}'.format(all_count) }} clips.
      {% endif %}
    </p>
  </form>

  <p class="count">{{ '{:,}'.format(total) }} clips</p>

  <div class="table-wrap">
    <table>
      <thead><tr><th>Speaker</th><th>Details</th><th>Script</th><th>Listen</th><th></th></tr></thead>
      <tbody>
        {% for clip in rows %}
          <tr aria-selected="{{ 'true' if clip.clip_id == s.clip else 'false' }}">
            <td class="speaker">{{ clip.speaker }}</td>
            <td class="details">{{ clip.details.values()|join(' · ') }}{% if length(clip) %} · {{ '%.1f' % length(clip) }} s{% endif %}</td>
            <td>{{ clip.text }}</td>
            <td><audio controls preload="none" src="/audio/{{ clip.dataset }}/{{ clip.clip_id }}.wav"></audio></td>
            <td class="use"><a class="button" href="/?{{ q(clip=clip.clip_id) }}">Use</a></td>
          </tr>
        {% else %}
          <tr><td class="empty" colspan="5">No clips match these options.</td></tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  <nav class="pager" aria-label="Pages">
    {% if page > 1 %}<a class="button" href="/?{{ q(page=page - 1) }}">Previous</a>{% endif %}
    <span>Page {{ page }} of {{ pages }}</span>
    {% if page < pages %}<a class="button" href="/?{{ q(page=page + 1) }}">Next</a>{% endif %}
  </nav>
</section>

<div class="side">
  <section class="generate" aria-label="Generate speech">
    <h2>Make this voice say something new</h2>

    {% if reference %}
      <p class="meta">Reference · {{ reference.dataset }} · {{ reference.speaker }}
        {% if length(reference) %} · {{ '%.1f' % length(reference) }} s{% endif %}</p>
      <blockquote>{{ reference.text }}</blockquote>
      <audio controls preload="auto" src="/audio/{{ reference.dataset }}/{{ reference.clip_id }}.wav"></audio>
      {% if length(reference) and length(reference) < 3 %}
        <p class="hint">Under 3 s of speech: short references clone less well.</p>
      {% endif %}
    {% else %}
      <p class="hint">Press "Use" on a clip. It becomes the reference voice.</p>
    {% endif %}

    <form method="post" action="/generate" autocomplete="off">
      <input type="hidden" name="sent" value="1">
      <input type="hidden" name="dataset" value="{{ s.dataset }}">
      <input type="hidden" name="clip" value="{{ s.clip }}">
      <input type="hidden" name="min" value="{{ '%g' % s.min }}">
      <input type="hidden" name="max" value="{{ '%g' % s.max }}">
      {% for name in ('picky', 'duplicate_actor', 'emotions') %}{% if s[name] %}<input type="hidden" name="{{ name }}" value="1">{% endif %}{% endfor %}
      {% if s.speaker %}<input type="hidden" name="speaker" value="{{ s.speaker }}">{% endif %}
      {% if s.text %}<input type="hidden" name="text" value="{{ s.text }}">{% endif %}
      {% for key, value in s.details.items() %}{% if value %}<input type="hidden" name="{{ key }}" value="{{ value }}">{% endif %}{% endfor %}
      <input type="hidden" name="page" value="{{ s.page }}">

      <label>Text to speak
        <textarea name="target" rows="3" maxlength="600">I can't believe we actually pulled that off. It was incredible.</textarea>
      </label>
      <label>Voice name <span class="hint">(output folder; blank = the speaker's id)</span>
        <input name="name" type="text" placeholder="e.g. {{ reference.speaker if reference else 'p247' }}">
      </label>
      <button type="submit" {{ 'disabled' if not reference }}>Generate</button>
      <p class="hint">The page waits while it generates. The first time also loads the model (about a minute).</p>
    </form>

    {% if result %}
      <p class="meta">New</p>
      <audio controls autoplay preload="auto" src="{{ result }}"></audio>
    {% endif %}

    <h3>Recent</h3>
    <ul class="recent">
      {% for item in recent %}
        <li><span><strong>{{ item.voice }}</strong> · {{ item.text or item.file }}</span>
            <audio controls preload="none" src="{{ item.audio }}"></audio></li>
      {% else %}
        <li class="empty">Nothing generated yet.</li>
      {% endfor %}
    </ul>
  </section>
</div>
</main>
</body>
</html>
"""


if __name__ == "__main__":
    library_ui.load_library()

    if not library_ui.library:
        raise SystemExit(f"No dataset folders found in {library_ui.LIBRARY_DIR}. Check VOICE_LIBRARY_DIR in .env.")

    # Speech lengths: read from output/picky/ (measured once, the first time takes a while)
    for name, data in library_ui.library.items():
        seconds[name] = measure(data["clips"])

    # Not 5000 or 5001, so all three pages can run side by side. The debugger stays off.
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)

"""Voice_model_example_UI_V3.py | V2, plus two ways to prepare the reference voice before cloning.

Everything V2 does (Picky options, filters, play, Use, Generate, Export), and:

    Combine   add a second clip of the SAME speaker. The two are joined into
              one longer reference (Combine_audio.py, with a pause between
              them), and their scripts are joined too. Short references clone
              less well (under 3 s, see Picky_voice.py), so two short clips of
              one voice make a better reference than either alone.
    Clean     remove background noise from the reference first
              (Clean_Audio_Distortion.py, DeepFilterNet 3). Strength 1 removes
              all it can; lower keeps some background and sounds more natural.

Order: combine first, then clean, then CosyVoice 3. "Preview reference" builds
the prepared reference without generating, so you can hear what the model will
hear. Prepared files are saved in output/prepared/references/.

Like V2, it is one Flask file with plain HTML forms and no JavaScript; every
setting lives in the URL.

    python Voice_model_example_UI_V3.py
    then open http://127.0.0.1:5003

Reused, not copied: V2's settings, Picky selection and export, the library and
audio routes of Voice_Importer_UI.py and the output routes of
Voice_model_example_UI.py.
"""
import json
import threading
import time
from pathlib import Path
from urllib.parse import parse_qsl, urlencode

import soundfile as sf
from flask import Flask, abort, redirect, render_template_string, request, send_file
from werkzeug.datastructures import MultiDict

import Voice_Importer_UI as library_ui
import Voice_model
import Voice_model_example_UI as generator_ui
import Voice_model_example_UI_V2 as v2
import model
from Clean_Audio_Distortion import clean_audio
from Combine_audio import combine_audio
from Picky_voice import measure
from Voice_Importer import find_clips, get_clip, list_speakers

PORT = 5003
PREPARED_DIR = model.PROJECT_DIR / "output" / "prepared" / "references"

# Defaults of the new options
GAP_SECONDS = 0.3
STRENGTH = 1.0

# CosyVoice refuses references over 30 s; keep a margin
MAX_REFERENCE_SECONDS = 29.0

app = Flask(__name__)

app.add_url_rule("/audio/<dataset>/<clip_id>.wav", view_func=library_ui.audio)
app.add_url_rule("/generated/<voice>/<file>", view_func=generator_ui.generated)

# The GPU makes one voice at a time
generate_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Settings: V2's, plus the second clip, the pause, and cleaning
# --------------------------------------------------------------------------- #


def _number(value, default, low, high):
    try:
        return min(max(float(value), low), high)
    except (TypeError, ValueError):
        return default


def read_settings(args):
    settings = v2.read_settings(args)
    settings.update(
        clip2=args.get("clip2", ""),
        gap=_number(args.get("gap"), GAP_SECONDS, 0.0, 3.0),
        clean=args.get("clean", "") not in ("", "0"),
        strength=_number(args.get("strength"), STRENGTH, 0.05, 1.0),
    )
    return settings


def query_string(settings, **changes):
    s = {**settings, **changes}
    extra = {}
    if s["clip2"]:
        extra["clip2"] = s["clip2"]
    if s["gap"] != GAP_SECONDS:
        extra["gap"] = f"{s['gap']:g}"
    if s["clean"]:
        extra["clean"] = 1
    if s["strength"] != STRENGTH:
        extra["strength"] = f"{s['strength']:g}"

    base = v2.query_string(s)
    return f"{base}&{urlencode(extra)}" if extra else base


def form_settings():
    """The page's settings (carried in the hidden "state" field) plus the generate panel's options."""
    settings = read_settings(MultiDict(parse_qsl(request.form.get("state", ""))))
    settings["clean"] = "clean" in request.form
    settings["gap"] = _number(request.form.get("gap"), settings["gap"], 0.0, 3.0)
    settings["strength"] = _number(request.form.get("strength"), settings["strength"], 0.05, 1.0)
    return settings


def back(settings, **params):
    """Redirect to the page with these settings and some messages."""
    params = {key: value for key, value in params.items() if value}
    return redirect(f"/?{query_string(settings)}" + (f"&{urlencode(params)}" if params else ""))


# --------------------------------------------------------------------------- #
# Preparing the reference: combine, then clean
# --------------------------------------------------------------------------- #


def reference_clips(settings):
    """The chosen clip and the optional second clip (None when there is none)."""
    by_id = library_ui.library[settings["dataset"]]["by_id"]
    first = by_id.get(settings["clip"])
    second = by_id.get(settings["clip2"]) if first else None

    if second is not None and (second.speaker != first.speaker or second.clip_id == first.clip_id):
        second = None

    return first, second


def prepare_reference(settings):
    """
    The reference CosyVoice will hear: (wav path, its script, what was done).
    Without combine or clean it is the library clip itself.
    """
    first, second = reference_clips(settings)
    if first is None:
        raise ValueError("Pick a reference clip first (press Use).")

    path, text = get_clip(first)
    steps = []

    if second is not None:
        second_path, second_text = get_clip(second)
        out = PREPARED_DIR / f"{first.clip_id}+{second.clip_id}_gap{settings['gap']:g}.wav"
        path = combine_audio(path, second_path, gap=settings["gap"], output=out)
        text = f"{text.strip()} {second_text.strip()}"
        steps.append(f"combined with {second.clip_id}")

    if settings["clean"]:
        out = PREPARED_DIR / f"{Path(path).stem}_clean{settings['strength']:g}.wav"
        if not out.exists():
            clean_audio(path, strength=settings["strength"], output=out)
        path = out
        steps.append(f"cleaned (strength {settings['strength']:g})")

    seconds = sf.info(str(path)).duration
    if seconds > MAX_REFERENCE_SECONDS:
        raise ValueError(f"The reference is {seconds:.1f} s; CosyVoice takes at most 30 s. Use shorter clips.")

    return Path(path), text, steps


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #


@app.route("/")
def index():
    s = read_settings(request.args)
    data = library_ui.library[s["dataset"]]
    all_clips = data["clips"]
    error = request.args.get("error", "")

    if s["picky"] and s["min"] > s["max"]:
        error = error or f"Min length ({s['min']:g} s) is above max length ({s['max']:g} s)."
        clips = []
    else:
        clips = v2.picky_clips(s)

    found = find_clips(
        clips,
        speaker=s["speaker"] or None,
        text=s["text"] or None,
        **{key: value for key, value in s["details"].items() if value},
    )

    size = library_ui.PAGE_SIZE
    pages = max(1, -(-len(found) // size))
    s["page"] = page = min(s["page"], pages)

    kept = list_speakers(clips)
    everyone = list_speakers(all_clips)
    first, second = reference_clips(s)

    prepared = request.args.get("prepared", "")
    prepared_path = PREPARED_DIR / prepared if prepared else None
    prepared_seconds = sf.info(str(prepared_path)).duration if prepared_path and prepared_path.is_file() else None

    recent = [generator_ui.generated_json(path) for path in
              sorted(generator_ui.OUTPUT_DIR.glob("*/*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)[:10]]

    return render_template_string(
        PAGE,
        s=s,
        q=lambda **changes: query_string(s, **changes),
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
        length=v2.length,
        first=first,
        second=second,
        combined_seconds=(v2.length(first) or 0) + s["gap"] + (v2.length(second) or 0) if second else None,
        prepared=prepared if prepared_seconds else "",
        prepared_seconds=prepared_seconds,
        steps=request.args.get("steps", ""),
        target=request.args.get("target", "I can't believe we actually pulled that off. It was incredible."),
        result=request.args.get("result", ""),
        message=request.args.get("message", ""),
        error=error,
        recent=recent,
    )


@app.route("/prepare", methods=["POST"])
def prepare():
    """Build the prepared reference and show it, without generating."""
    s = form_settings()
    target = request.form.get("target", "")

    try:
        path, _, steps = prepare_reference(s)
    except Exception as error:
        return back(s, error=str(error), target=target)

    if path.parent != PREPARED_DIR:
        return back(s, message="Nothing to prepare: tick Clean or add a second clip. The reference is the clip as it is.",
                    target=target)

    return back(s, prepared=path.name, steps=", ".join(steps), target=target)


@app.route("/generate", methods=["POST"])
def generate():
    """Prepare the reference, speak the text in its voice, then go back to the page with the result."""
    s = form_settings()
    text = request.form.get("target", "").strip()

    if not text:
        return back(s, error="Type the text to speak.")
    if len(text) > generator_ui.MAX_TEXT_CHARS:
        return back(s, error=f"The text is longer than {generator_ui.MAX_TEXT_CHARS} characters.", target=text)

    first, second = reference_clips(s)
    if first is None:
        return back(s, error="Pick a reference clip first (press Use).", target=text)

    # The name becomes a folder: letters, digits, dash and underscore only
    name = request.form.get("name", "").strip() or first.speaker
    name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:60].strip("._") or first.speaker

    try:
        with generate_lock:
            reference_audio, reference_text, steps = prepare_reference(s)
            started = time.time()
            path = Voice_model.generate_speech(reference_audio, reference_text, text, name)
    except Exception as error:
        import traceback
        traceback.print_exc()
        return back(s, error=f"{type(error).__name__}: {error}", target=text)

    # The same .json as V1 writes, plus how the reference was prepared
    record = {
        "target_text": text,
        "reference_clip": f"{first.dataset}/{first.clip_id}",
        "second_clip": f"{second.dataset}/{second.clip_id}" if second else None,
        "cleaned": s["strength"] if s["clean"] else None,
        "reference_audio": str(reference_audio),
        "reference_text": reference_text,
        "voice_name": path.parent.name,
        "seconds": round(time.time() - started, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path.with_suffix(".json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    prepared = reference_audio.name if reference_audio.parent == PREPARED_DIR else ""
    return back(s, result=generator_ui.generated_json(path)["audio"], prepared=prepared,
                steps=", ".join(steps), target=text)


@app.route("/export", methods=["POST"])
def export():
    """V2's export (the Picky form posts its own fields)."""
    return v2.export()


@app.route("/prepared/<file>")
def prepared_file(file):
    """A prepared reference. The path is checked to stay inside PREPARED_DIR."""
    path = (PREPARED_DIR / file).resolve()

    if PREPARED_DIR.resolve() != path.parent or path.suffix != ".wav" or not path.is_file():
        abort(404)

    return send_file(path, mimetype="audio/wav")


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice Generator V3</title>
<link rel="stylesheet" href="{{ url_for('static', filename='style.css') }}">
<style>
  .tabs a { padding: 6px 12px; border-radius: 6px; color: var(--muted); text-decoration: none; }
  .tabs a.on { background: var(--accent-soft); color: var(--accent); font-weight: 600; }
  .version { font-size: 0.75rem; color: var(--accent); background: var(--accent-soft); padding: 2px 7px; border-radius: 999px; vertical-align: middle; }
  .picky { display: flex; flex-wrap: wrap; align-items: end; gap: 10px 14px; padding: 10px 12px; margin-bottom: 10px;
           background: var(--band); border: 1px solid var(--rule); border-radius: 8px; }
  .picky label { display: grid; gap: 3px; font-size: 0.8rem; color: var(--muted); }
  .picky input[type=number], .options input[type=number] { width: 6.5em; min-width: 0; }
  label.switch { display: flex !important; align-items: center; gap: 6px; font-size: 0.88rem; color: var(--ink) !important; padding-bottom: 6px; }
  label.switch input { width: auto !important; min-width: 0; padding: 0; margin: 0; accent-color: var(--accent); }
  .picky .summary { flex-basis: 100%; margin: 0; font-size: 0.82rem; color: var(--muted); }
  .row-filters { flex-basis: 100%; display: flex; flex-wrap: wrap; align-items: end; gap: 10px 14px; }
  td audio { width: 210px; height: 32px; }
  td.use { white-space: nowrap; }
  tbody tr { cursor: default; }
  tbody tr.second { background: var(--band); }
  .note { margin: 0 0 10px; padding: 8px 10px; border-radius: 6px; font-size: 0.9rem; background: var(--accent-soft); color: var(--accent); overflow-wrap: anywhere; }
  .note.error { background: #f9e1df; color: #9c2b22; }
  @media (prefers-color-scheme: dark) { .note.error { background: #3a1c1a; color: #f08a80; } }
  .recent li { display: grid; gap: 4px; }
  .recent audio, .clipcard audio { height: 32px; }
  a.button { display: inline-block; padding: 3px 10px; border: 1px solid var(--rule); border-radius: 6px; color: var(--accent); text-decoration: none; font-size: 0.88rem; }
  .clipcard { border: 1px solid var(--rule); border-radius: 8px; padding: 10px; margin: 8px 0; display: grid; gap: 6px; }
  .clipcard .meta { margin: 0; }
  .clipcard p { margin: 0; }
  .options { display: grid; gap: 8px; padding: 10px; border-radius: 8px; background: var(--band); }
  .options .line { display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: end; }
  .buttons { display: flex; gap: 8px; flex-wrap: wrap; }
</style>
</head>
<body>
<header>
  <h1>Voice Generator <span class="version">V3</span></h1>
  <nav class="tabs" aria-label="Dataset">
    {% for name in datasets %}
      <a href="/?{{ q(dataset=name, speaker='', clip='', clip2='', page=1, details={}) }}" class="{{ 'on' if name == s.dataset }}">{{ name }}</a>
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
    {% for name in ('clip', 'clip2') %}{% if s[name] %}<input type="hidden" name="{{ name }}" value="{{ s[name] }}">{% endif %}{% endfor %}
    <input type="hidden" name="gap" value="{{ '%g' % s.gap }}">
    <input type="hidden" name="strength" value="{{ '%g' % s.strength }}">
    {% if s.clean %}<input type="hidden" name="clean" value="1">{% endif %}

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
          <tr aria-selected="{{ 'true' if first and clip.clip_id == first.clip_id else 'false' }}"
              class="{{ 'second' if second and clip.clip_id == second.clip_id }}">
            <td class="speaker">{{ clip.speaker }}</td>
            <td class="details">{{ clip.details.values()|join(' · ') }}{% if length(clip) %} · {{ '%.1f' % length(clip) }} s{% endif %}</td>
            <td>{{ clip.text }}</td>
            <td><audio controls preload="none" src="/audio/{{ clip.dataset }}/{{ clip.clip_id }}.wav"></audio></td>
            <td class="use">
              <a class="button" href="/?{{ q(clip=clip.clip_id, clip2='') }}">Use</a>
              {% if first and clip.speaker == first.speaker and clip.clip_id != first.clip_id %}
                <a class="button" href="/?{{ q(clip2=clip.clip_id) }}" title="Join this clip after the reference">+ Combine</a>
              {% endif %}
            </td>
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

    {% if first %}
      <div class="clipcard">
        <p class="meta">Reference · {{ first.dataset }} · {{ first.speaker }}{% if length(first) %} · {{ '%.1f' % length(first) }} s{% endif %}</p>
        <p>{{ first.text }}</p>
        <audio controls preload="none" src="/audio/{{ first.dataset }}/{{ first.clip_id }}.wav"></audio>
      </div>

      {% if second %}
        <div class="clipcard">
          <p class="meta">Combined with · {{ second.clip_id }}{% if length(second) %} · {{ '%.1f' % length(second) }} s{% endif %}
            · <a href="/?{{ q(clip2='') }}">remove</a></p>
          <p>{{ second.text }}</p>
          <audio controls preload="none" src="/audio/{{ second.dataset }}/{{ second.clip_id }}.wav"></audio>
          <p class="hint">Together about {{ '%.1f' % combined_seconds }} s of speech.</p>
        </div>
      {% else %}
        <p class="hint">To combine: press "+ Combine" on another clip of {{ first.speaker }}.
          <a href="/?{{ q(speaker=first.speaker, page=1) }}">Show {{ first.speaker }}'s clips</a>
          {% if not s.duplicate_actor %}(turn on Duplicate actor to see more than one){% endif %}</p>
      {% endif %}
    {% else %}
      <p class="hint">Press "Use" on a clip. It becomes the reference voice.</p>
    {% endif %}

    <form method="post" action="/generate" autocomplete="off">
      <input type="hidden" name="state" value="{{ q() }}">

      <div class="options">
        <div class="line">
          <label class="switch" title="Remove background noise with DeepFilterNet 3 before cloning">
            <input type="checkbox" name="clean" {{ 'checked' if s.clean }}> Clean the reference first</label>
          <label>Strength <input type="number" name="strength" min="0.05" max="1" step="0.05" value="{{ '%g' % s.strength }}"></label>
        </div>
        {% if second %}
          <div class="line">
            <label>Pause between the clips (s) <input type="number" name="gap" min="0" max="3" step="0.1" value="{{ '%g' % s.gap }}"></label>
          </div>
        {% else %}
          <input type="hidden" name="gap" value="{{ '%g' % s.gap }}">
        {% endif %}
      </div>

      {% if prepared %}
        <div class="clipcard">
          <p class="meta">Prepared reference · {{ '%.1f' % prepared_seconds }} s{% if steps %} · {{ steps }}{% endif %}</p>
          <audio controls preload="auto" src="/prepared/{{ prepared }}"></audio>
        </div>
      {% endif %}

      <label>Text to speak
        <textarea name="target" rows="3" maxlength="600">{{ target }}</textarea>
      </label>
      <label>Voice name <span class="hint">(output folder; blank = the speaker's id)</span>
        <input name="name" type="text" placeholder="e.g. {{ first.speaker if first else 'p247' }}">
      </label>
      <div class="buttons">
        <button type="submit" formaction="/prepare" class="quiet" {{ 'disabled' if not first }}>Preview reference</button>
        <button type="submit" {{ 'disabled' if not first }}>Generate</button>
      </div>
      <p class="hint">The page waits while it works. The first time also loads the models (about a minute).</p>
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

    # Speech lengths, shared with V2's helpers (read from output/picky/)
    for name, data in library_ui.library.items():
        v2.seconds[name] = measure(data["clips"])

    # Not 5000-5002, so every page can run side by side. The debugger stays off.
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)

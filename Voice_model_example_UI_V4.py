"""Voice_model_example_UI_V4.py | V3, made safe to reach from outside this PC (for example through playit.gg).

Everything V3 does, and:

    Password  every page, clip and file needs a login. The password is
              UI_PASSWORD in .env; if it is missing, a strong one is made and
              saved there on the first start. After 5 wrong tries the login
              locks for 30 s, doubling up to 15 minutes. The lock is shared by
              all visitors, because through a tunnel they all seem to come from
              this PC. A login lasts 12 hours, or until Log out or a restart.
    HTTPS     the whole connection is encrypted, from the browser to this
              program. A tunnel only passes the encrypted bytes along.

The certificate is made by this file on the first start (certs/, never
committed). Nobody has vouched for it, so browsers warn about it once. Before
accepting the warning, check that the certificate's SHA-256 fingerprint matches
the one this program prints when it starts: that check is what proves the
browser is really talking to this PC. To use a real certificate instead, set
UI_TLS_CERT and UI_TLS_KEY in .env.

Anything on the internet can reach a tunnel, including scanners that connect and
then say nothing. Each connection does its HTTPS handshake on its own thread, so
one of those cannot hold up the others, and is closed after IDLE_SECONDS of silence.

Random voices: a panel above the clips picks a number of random neutral voices,
one per actor, from one dataset or an even share of all (split_voice_data.py). It
can stay inside the validation speakers (80%) or the final test speakers (20%). Press Use on a voice to make it the reference; the list
stays while you go through them. The seed is shown, and the same settings and seed
give the same voices. Custom voices are never picked.

CosyVoice starts loading in the background as soon as the page is up, so the first
Generate is ready sooner (about half a minute on this PC).

Examine: every clip under Recent shows where its voice comes from (dataset,
speaker, clip, labels, split), and has an Examine button that measures it against
its reference with reward.clone_metrics: WER, speaker similarity, quality,
speaking rate and the rewards. The numbers go in the clip's node in measurements/
(measurements.py, never committed), and the clip's .json links to it, so they show
again without measuring. "Measure again" adds a new run and keeps the old ones.

    python Voice_model_example_UI_V4.py
    then open https://127.0.0.1:5004

Through playit.gg: add a TCP tunnel to local address 127.0.0.1, port 5004, then
open https://<the address playit gives>:<its port>. A visit without https:// is
sent there: phones fall back to plain http when they do not trust the certificate.

Reused, not copied: all of V3. Only this page is protected: the older UIs
(ports 5000-5003) have no login, so do not put a tunnel in front of them.
"""
import hmac
import json
import os
import re
import secrets
import socket
import ssl
import threading
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit

from flask import abort, redirect, render_template_string, request, session
from werkzeug.datastructures import MultiDict
from werkzeug.serving import WSGIRequestHandler

import Voice_model_example_UI_V3 as v3
import model
import measurements
import split_voice_data
from Picky_voice import MAX_SECONDS, MIN_SECONDS
from Voice_Importer import get_clip, read_custom_voices

PORT = 5004
ENV_FILE = model.PROJECT_DIR / ".env"
CERT_DIR = model.PROJECT_DIR / "certs"

MIN_PASSWORD_LENGTH = 12
FREE_TRIES = 5              # wrong passwords before the login locks
LOCK_SECONDS = 30           # the first lock; doubles with every further wrong try
MAX_LOCK_SECONDS = 900
LOGIN_HOURS = 12
IDLE_SECONDS = 30           # a connection silent for this long is closed

VOICES_FROM_ALL = "all"     # the random voices come from an even share of every dataset
# The speakers are split once, per dataset (split_voice_data.py): most for validation, the rest
# locked away as the final test. The page says which is which, and how big each part is.
TEST_PERCENT = round(100 * split_voice_data.TEST_FRACTION)
SPLIT_SIZES = {name: len(keys) for name, keys in split_voice_data.load_splits().items()}
VOICE_SPLITS = {
    "validation": f"Validation {100 - TEST_PERCENT}%",
    "test": f"Final test {TEST_PERCENT}%",
    "all": "Both",
}
SPLIT_NAMES = {"validation": f"validation ({100 - TEST_PERCENT}%)", "test": f"final test ({TEST_PERCENT}%)"}
VOICE_SPLITS_ONLY = ["validation", "test"]   # the order the blocks show in, when both are picked
VOICE_SPLIT = "validation"  # the final test stays untouched unless it is asked for
VOICES_SHOWN = 15           # the number first shown in the box

app = v3.app
# A new key at every start: restarting logs everyone out
app.secret_key = secrets.token_hex(32)
app.config.update(
    SESSION_COOKIE_NAME="voice_ui_v4",
    SESSION_COOKIE_SECURE=True,         # only ever sent over HTTPS
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=timedelta(hours=LOGIN_HOURS),
)

# Your own voices get a tab of their own. VOICES_DIR is a full path, so it does not
# depend on the voice library's folder.
v3.library_ui.DATASETS["Custom voices"] = (read_custom_voices, model.setting("VOICES_DIR"), [])

# On a phone the page is three tabs, with a bar at the bottom and swiping between
# them. Without JavaScript, or on a wide screen, everything shows as in V3.
PHONE_STYLE = """<style>
  .phone-tabs, .filters-toggle { display: none; }
  .sampler { margin-bottom: 10px; padding: 10px 12px; background: var(--band); border: 1px solid var(--rule); border-radius: 8px; }
  .sampler summary { cursor: pointer; font-weight: 600; }
  .sampler-form { display: flex; flex-wrap: wrap; align-items: end; gap: 10px 14px; margin: 10px 0; }
  .sampler-form label { display: grid; gap: 3px; font-size: 0.8rem; color: var(--muted); }
  .sampler-form input[type=number] { width: 6.5em; min-width: 0; }
  .sampler .summary { margin: 0 0 8px; font-size: 0.82rem; color: var(--muted); }
  .sampler .table-wrap { background: var(--surface); }
  .recent .origin { font-size: 0.8rem; color: var(--muted); }
  .recent form.examine { margin: 0; }
  .recent li.examined { outline: 2px solid var(--accent); outline-offset: 4px; border-radius: 4px; }
  .metrics summary { cursor: pointer; font-size: 0.88rem; }
  .metrics table { width: 100%; font-size: 0.82rem; border-collapse: collapse; margin-top: 6px; }
  .metrics th { text-align: left; font-weight: 600; color: var(--muted); padding: 3px 8px 3px 0; vertical-align: top; width: 38%; }
  .metrics td { padding: 3px 0; overflow-wrap: anywhere; }
  .sampler td audio { width: 150px; }

  /* With both splits shown, each is its own block: validation in the accent colour, the final test in amber */
  .sampler { --test: #9a5b00; --test-soft: #fbefd9; }
  @media (prefers-color-scheme: dark) { .sampler { --test: #f0b45a; --test-soft: #3a2a10; } }
  .split-group + .split-group { margin-top: 16px; }
  .split-head { display: flex; align-items: center; gap: 8px; margin: 0 0 6px; font-size: 0.88rem; font-weight: 400; color: var(--muted); }
  .split-tag { padding: 2px 9px; border-radius: 999px; font-weight: 600; background: var(--accent-soft); color: var(--accent); }
  .split-test .split-tag { background: var(--test-soft); color: var(--test); }
  .split-none .split-tag { background: var(--band); color: var(--muted); }
  .sampler .split-group:has(.split-head) { padding-left: 10px; border-left: 4px solid var(--accent); }
  .sampler .split-test:has(.split-head) { border-left-color: var(--test); }
  .sampler .split-none:has(.split-head) { border-left-color: var(--rule); }
  @media (max-width: 860px) {
    .sampler-form label { flex: 1 1 40%; }
    .sampler-form select, .sampler-form input[type=number] { width: 100%; box-sizing: border-box; }
    .sampler-form button { flex: 1 1 100%; }
    .sampler .table-wrap { background: none; }
    .sampler td audio { width: 100%; }
    .sampler td.use { order: 9; }
    .sampler summary { min-height: 28px; display: list-item; }
    body { padding-bottom: calc(64px + env(safe-area-inset-bottom)); }
    header { flex-wrap: wrap; gap: 6px 12px; }
    header h1 { font-size: 1.1rem; margin: 0; }
    .tabs { order: 3; flex-basis: 100%; overflow-x: auto; }
    .tabs a { white-space: nowrap; padding: 10px 12px; }
    main { display: block; padding-bottom: 8px; }
    .side { position: static; max-height: none; overflow: visible; display: contents; }

    /* One tab at a time */
    body[data-tab] main > section, body[data-tab] .side > section { display: none; }
    body[data-tab=clips] section[aria-label=Clips],
    body[data-tab=generate] section.generate,
    body[data-tab=recent] section.recent-part { display: block; }

    .phone-tabs { display: grid; grid-template-columns: repeat(3, 1fr); position: fixed; left: 0; right: 0; bottom: 0; z-index: 10;
                  background: var(--surface); border-top: 1px solid var(--rule); padding-bottom: env(safe-area-inset-bottom); }
    body:not([data-tab]) .phone-tabs { display: none; }
    .phone-tabs button { all: unset; box-sizing: border-box; min-height: 56px; text-align: center; font-size: 0.95rem; color: var(--muted); cursor: pointer;
                         border-top: 3px solid transparent; -webkit-tap-highlight-color: transparent; }
    .phone-tabs button[aria-current=page] { color: var(--accent); font-weight: 600; border-top-color: var(--accent); background: var(--accent-soft); }
    .phone-tabs button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }

    /* iPhones zoom the page when a text box under 16px gets focus */
    input, select, textarea, button, a.button { font-size: 16px; }
    button, a.button { min-height: 44px; box-sizing: border-box; }
    a.button { display: inline-flex; align-items: center; justify-content: center; padding: 6px 14px; }
    input[type=search], input[type=text], select, textarea { width: 100%; box-sizing: border-box; }
    .row-filters label { flex: 1 1 100%; }
    audio, td audio, .recent audio, .clipcard audio { width: 100%; height: 40px; }

    body[data-tab] .filters-toggle { display: block; width: 100%; margin-bottom: 10px; }
    body[data-tab]:not(.filters-open) .picky { display: none; }

    /* The clips as cards: no table to scroll sideways, and room to swipe */
    .table-wrap { overflow: visible; background: none; border: 0; }
    .table-wrap thead { display: none; }
    .table-wrap table, .table-wrap tbody { display: block; }
    .table-wrap tr { display: grid; gap: 6px; padding: 12px; margin-bottom: 10px; background: var(--surface);
                     border: 1px solid var(--rule); border-radius: 8px; }
    .table-wrap td { display: block; padding: 0; border: 0; }
    .table-wrap td.use { display: flex; gap: 8px; }
    .table-wrap td.use a { flex: 1; }
    .pager a.button { flex: 1; }
  }
</style>
"""

PHONE_TABS = """<nav class="phone-tabs" aria-label="Parts of the page">
  <button type="button" data-tab="clips">Clips</button>
  <button type="button" data-tab="generate">Generate</button>
  <button type="button" data-tab="recent">Recent</button>
</nav>
<script>
(function () {
  var order = ["clips", "generate", "recent"];
  var body = document.body;
  var buttons = document.querySelectorAll(".phone-tabs button");
  var scrolled = {};

  // Storage can be off, for example in a private window: the page then opens on Clips
  function remembered() { try { return sessionStorage.getItem("voice_tab"); } catch (e) { return null; } }
  function remember(tab) { try { sessionStorage.setItem("voice_tab", tab); } catch (e) {} }

  function show(tab) {
    if (body.dataset.tab) scrolled[body.dataset.tab] = window.scrollY;
    body.dataset.tab = tab;
    remember(tab);
    buttons.forEach(function (button) {
      if (button.dataset.tab === tab) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    });
    window.scrollTo(0, scrolled[tab] || 0);
  }

  buttons.forEach(function (button) {
    button.addEventListener("click", function () { show(button.dataset.tab); });
  });

  // A finished generation or preview opens where its result is
  var query = new URLSearchParams(location.search);
  var saved = remembered();
  show(query.get("result") || query.get("prepared") ? "generate" : order.indexOf(saved) >= 0 ? saved : "clips");
  var made = query.get("result") && document.querySelector("section.generate > audio");
  if (made) made.scrollIntoView({ block: "center" });

  // Messages sit in the Clips part, but must show on every tab
  if (window.matchMedia("(max-width: 860px)").matches) {
    var main = document.querySelector("main");
    document.querySelectorAll("section[aria-label=Clips] > .note").forEach(function (note) {
      main.parentNode.insertBefore(note, main);
    });
  }

  document.querySelector(".filters-toggle").addEventListener("click", function () {
    this.setAttribute("aria-expanded", body.classList.toggle("filters-open"));
  });

  // Choosing a clip goes on to Generate; asking for a speaker's clips goes back to Clips
  document.addEventListener("click", function (event) {
    var link = event.target.closest("a");
    if (!link) return;
    if (link.closest("td.use")) remember("generate");
    else if (link.closest(".generate") && link.search.indexOf("speaker=") >= 0) remember("clips");
  });

  // Swiping sideways, but not while using a player, a box or a slider
  var start = null;
  document.querySelector("main").addEventListener("touchstart", function (event) {
    var touch = event.touches[0];
    start = event.touches.length > 1 || event.target.closest("audio, input, select, textarea") ? null : { x: touch.clientX, y: touch.clientY };
  }, { passive: true });
  document.querySelector("main").addEventListener("touchend", function (event) {
    if (!start || !window.matchMedia("(max-width: 860px)").matches) return;
    var touch = event.changedTouches[0];
    var across = touch.clientX - start.x;
    var down = touch.clientY - start.y;
    start = null;
    if (Math.abs(across) < 70 || Math.abs(across) < 2 * Math.abs(down)) return;
    var next = order.indexOf(body.dataset.tab) + (across < 0 ? 1 : -1);
    if (next >= 0 && next < order.length) show(order[next]);
  }, { passive: true });

  // The page waits while it generates: say so, and do not send it twice
  var form = document.querySelector("form[action='/generate']");
  var sending = false;
  form.addEventListener("submit", function (event) {
    if (sending) { event.preventDefault(); return; }
    sending = true;
    if (event.submitter) { event.submitter.dataset.label = event.submitter.textContent; event.submitter.textContent = "Working..."; }
  });
  window.addEventListener("pageshow", function () {
    sending = false;
    form.querySelectorAll("button[data-label]").forEach(function (button) { button.textContent = button.dataset.label; });
  });
})();
</script>
"""

# The random voices. "Use" opens a voice in its own dataset; the list stays in the URL.
SAMPLER = """{% set sample = random_voices(s) %}
  <details class="sampler" {{ 'open' if s.voices }}>
    <summary>Random voices{% if sample.rows %} ({{ sample.rows|length }}){% endif %}</summary>
    <form method="post" action="/random" class="sampler-form" autocomplete="off">
      <input type="hidden" name="state" value="{{ q() }}">
      <label>How many <input type="number" name="voices" min="1" required value="{{ s.voices or voices_shown }}"></label>
      <label>From
        <select name="voices_from">
          <option value="{{ from_all }}">All, even share</option>
          {% for name in sample.sources %}<option {{ 'selected' if name == s.voices_from }}>{{ name }}</option>{% endfor %}
        </select>
      </label>
      <label>Speakers
        <select name="voices_split">
          {% for value, label in voice_splits.items() %}<option value="{{ value }}" {{ 'selected' if value == s.voices_split }}>{{ label }}</option>{% endfor %}
        </select>
      </label>
      <label>Min length (s) <input type="number" name="voices_min" min="0" max="30" step="0.5" value="{{ '%g' % s.voices_min }}"></label>
      <label>Max length (s) <input type="number" name="voices_max" min="0" max="30" step="0.5" value="{{ '%g' % s.voices_max }}"></label>
      <label>Seed <input type="number" name="voices_seed" min="0" placeholder="new"></label>
      <button type="submit">Pick voices</button>
      {% if s.voices %}<a class="button" href="/?{{ q(voices=0) }}">Clear</a>{% endif %}
    </form>
    <p class="summary">Every dataset's speakers are split once: {{ 100 - test_percent }}% for validation ({{ split_sizes.validation }} speakers),
      to tune and listen with, and {{ test_percent }}% for the final test ({{ split_sizes.test }} speakers), kept for the end.
      A speaker is only ever in one of them.</p>
    {% if sample.error %}<p class="note error">{{ sample.error }}</p>{% endif %}
    {% if sample.rows %}
      <p class="summary">Seed {{ s.voices_seed }}, {{ 'from the ' ~ split_names[s.voices_split] if s.voices_split in split_names else 'from both splits, shown apart below' }}. The same settings and seed give the same voices.</p>
      {% for group in sample.groups %}
      <div class="split-group split-{{ group.split or 'none' }}">
      {% if group.label %}<h4 class="split-head"><span class="split-tag">{{ group.label }}</span>
        {{ group.rows|length }} voice{{ 's' if group.rows|length != 1 }}</h4>{% endif %}
      <div class="table-wrap">
        <table>
          <thead><tr><th></th><th>Voice</th><th>Details</th><th>Script</th><th>Listen</th></tr></thead>
          <tbody>
            {% for clip in group.rows %}
              <tr aria-selected="{{ 'true' if first and clip.dataset == first.dataset and clip.clip_id == first.clip_id else 'false' }}">
                <td class="use"><a class="button" href="/?{{ q(dataset=clip.dataset, clip=clip.clip_id, clip2='', speaker='', text='', page=1, details={}) }}">Use</a></td>
                <td class="speaker">{{ clip.speaker }}<br><span class="details">{{ clip.dataset }}</span></td>
                <td class="details">{{ clip.details.values()|join(' · ') }}{% if length(clip) %} · {{ '%.1f' % length(clip) }} s{% endif %}</td>
                <td>{{ clip.text }}</td>
                <td><audio controls preload="none" src="/audio/{{ clip.dataset }}/{{ clip.clip_id }}.wav"></audio></td>
              </tr>
            {% endfor %}
          </tbody>
        </table>
      </div>
      </div>
      {% endfor %}
    {% endif %}
  </details>
"""

# So that Apply in the filters does not lose the random voices
SAMPLER_KEPT = """{% if s.voices %}{% for name in voice_fields %}<input type="hidden" name="{{ name }}" value="{{ voice_values(s)[name] }}">{% endfor %}{% endif %}
    """

# A clip under Recent: where its voice comes from, Examine, and the numbers once measured
RECENT_ITEM_V3 = """        <li><span><strong>{{ item.voice }}</strong> · {{ item.text or item.file }}</span>
            <audio controls preload="none" src="{{ item.audio }}"></audio></li>"""

NUMBER = "{% macro n(value, form='%.2f') %}{{ form % value if value is not none else '-' }}{% endmacro %}"

RECENT_ITEM = """        {% set info = clip_info(item) %}
        <li class="{{ 'examined' if info and request.args.get('examined') == info.path }}">
          <span><strong>{{ item.voice }}</strong> · {{ item.text or item.file }}</span>
          {% if info and info.origin_text %}<span class="origin">Voice from {{ info.origin_text }}</span>{% endif %}
          <audio controls preload="none" src="{{ item.audio }}"></audio>
          {% if info %}
            <form method="post" action="/examine" class="examine">
              <input type="hidden" name="state" value="{{ q() }}">
              <input type="hidden" name="clip" value="{{ info.path }}">
              {% if info.metrics %}<input type="hidden" name="force" value="1">{% endif %}
              <button type="submit" class="quiet">{{ 'Measure again' if info.metrics else 'Examine' }}</button>
            </form>
            {% set m = info.metrics %}
            {% if m %}
              <details class="metrics" {{ 'open' if request.args.get('examined') == info.path }}>
                <summary>WER {{ n(m.wer) }} · similarity {{ n(m.similarity) }} · PESQ {{ n(m.quality.pesq) }}</summary>
                <table>
                  <tr><th>Heard</th><td>{{ m.heard }}</td></tr>
                  <tr><th>Word error rate</th><td>{{ n(m.wer) }}</td></tr>
                  <tr><th>Similarity to reference</th><td>{{ n(m.similarity, '%.3f') }}</td></tr>
                  <tr><th>Quality: PESQ · STOI · SI-SDR</th><td>clone {{ n(m.quality.pesq) }} · {{ n(m.quality.stoi) }} · {{ n(m.quality.si_sdr, '%.1f') }} dB<br>
                      reference {{ n(m.reference_quality.pesq) }} · {{ n(m.reference_quality.stoi) }} · {{ n(m.reference_quality.si_sdr, '%.1f') }} dB</td></tr>
                  <tr><th>Speaking rate</th><td>clone {{ n(m.words_per_second) }} words/s · reference {{ n(m.reference_words_per_second) }}
                      {% if m.rate_vs_reference %}({{ n(m.rate_vs_reference) }}×){% endif %}</td></tr>
                  <tr><th>Length</th><td>{{ n(m.seconds, '%.1f') }} s, {{ n(m.speech_seconds, '%.1f') }} s of speech</td></tr>
                  <tr><th>Rewards (0 to 1)</th><td>WER {{ n(m.rewards.wer) }} · similarity {{ n(m.rewards.similarity) }} · quality {{ n(m.rewards.quality) }}</td></tr>
                  <tr><th>Measured</th><td>{{ m.measured }} with {{ m.models.wer }}, {{ m.models.similarity }}, {{ m.models.quality }}</td></tr>
                  <tr><th>Logged in</th><td>{{ info.node }} ({{ info.runs }} run{{ 's' if info.runs != 1 }})</td></tr>
                </table>
              </details>
            {% endif %}
          {% endif %}
        </li>"""

# Examining waits for the models: say so, do not send twice, and show the result
EXAMINE_SCRIPT = """<script>
(function () {
  var busy = false;
  document.querySelectorAll("form.examine").forEach(function (form) {
    form.addEventListener("submit", function (event) {
      if (busy) { event.preventDefault(); return; }
      busy = true;
      var button = form.querySelector("button");
      button.dataset.label = button.textContent;
      button.textContent = "Measuring...";
    });
  });
  window.addEventListener("pageshow", function () {
    busy = false;
    document.querySelectorAll("form.examine button[data-label]").forEach(function (button) { button.textContent = button.dataset.label; });
  });
  var examined = document.querySelector(".recent li.examined");
  if (examined) examined.scrollIntoView({ block: "center" });
})();
</script>
"""

assert v3.PAGE.count(RECENT_ITEM_V3) == 1, "V3's Recent list has changed"

# The page is V3's, with its own name, a Log out button, the random voices, Examine and the phone tabs
v3.PAGE = (v3.PAGE
           .replace("<body>", "<body>\n" + NUMBER, 1)
           .replace(RECENT_ITEM_V3, RECENT_ITEM, 1)
           .replace("</body>", EXAMINE_SCRIPT + "</body>", 1)
           .replace("<title>Voice Generator V3</title>", "<title>Voice Generator V4</title>")
           .replace('<span class="version">V3</span>', '<span class="version">V4</span>')
           .replace("</header>", """  <form method="post" action="/logout"><button type="submit">Log out</button></form>
</header>""", 1)
           .replace("</head>", PHONE_STYLE + "</head>", 1)
           .replace('<form method="get" action="/" class="picky"',
                    SAMPLER + '  <button type="button" class="filters-toggle quiet" aria-expanded="false">Filters</button>\n'
                    '  <form method="get" action="/" class="picky"', 1)
           .replace('<input type="hidden" name="sent" value="1">',
                    SAMPLER_KEPT + '<input type="hidden" name="sent" value="1">', 1)
           .replace("<h3>Recent</h3>", '</section>\n  <section class="recent-part" aria-label="Recent">\n    <h3>Recent</h3>', 1)
           .replace("</body>", PHONE_TABS + "</body>", 1))


# --------------------------------------------------------------------------- #
# Random voices: V3's settings and page, plus the list in the URL
# --------------------------------------------------------------------------- #

v3_read_settings = v3.read_settings
v3_query_string = v3.query_string


def read_settings(args):
    settings = v3_read_settings(args)
    source = args.get("voices_from", VOICES_FROM_ALL)
    split = args.get("voices_split", VOICE_SPLIT)
    settings.update(
        voices=max(args.get("voices", 0, type=int), 0),
        voices_from=source if source in split_voice_data.PICKERS else VOICES_FROM_ALL,
        voices_split=split if split in VOICE_SPLITS else VOICE_SPLIT,
        voices_min=v3._number(args.get("voices_min"), MIN_SECONDS, 0.0, 30.0),
        voices_max=v3._number(args.get("voices_max"), MAX_SECONDS, 0.0, 30.0),
        voices_seed=max(args.get("voices_seed", 0, type=int), 0),
    )
    return settings


def voice_values(settings):
    """The random voices' settings, as they go in a URL or a form."""
    return {
        "voices": settings["voices"],
        "voices_from": settings["voices_from"],
        "voices_split": settings["voices_split"],
        "voices_min": f"{settings['voices_min']:g}",
        "voices_max": f"{settings['voices_max']:g}",
        "voices_seed": settings["voices_seed"],
    }


def query_string(settings, **changes):
    s = {**settings, **changes}
    base = v3_query_string(s)
    return f"{base}&{urlencode(voice_values(s))}" if s["voices"] else base


v3.read_settings = read_settings
v3.query_string = query_string


def split_shares(number):
    """How many of `number` voices come from each split when both are asked for: 80 % validation, 20 % final test."""
    test = round(number * split_voice_data.TEST_FRACTION)
    return {"validation": number - test, "test": test}


@lru_cache(maxsize=32)
def pick_voices(number, source, split, min_seconds, max_seconds, seed):
    # Both splits: each is picked on its own, so the 80/20 share holds in every list, not just on average
    if split == "all":
        return [clip for name, share in split_shares(number).items() if share
                for clip in pick_voices(share, source, name, min_seconds, max_seconds, seed)]

    library = v3.library_ui.library
    missing = [name for name in split_voice_data.PICKERS if name not in library]

    if source != VOICES_FROM_ALL:
        if source in missing:
            raise ValueError(f"{source} is not in the voice library.")
        picker = split_voice_data.PICKERS[source]
        return picker(library[source]["clips"], number, min_seconds, max_seconds, seed, split)

    if missing:
        raise ValueError(f"An even share needs every dataset, and {', '.join(missing)} is not in the voice library.")
    clips = {name: library[name]["clips"] for name in split_voice_data.PICKERS}
    return split_voice_data.pick_even(number, min_seconds, max_seconds, seed, split, clips)


def random_voices(settings):
    """What the panel shows: the voices, or why there are none."""
    sources = [name for name in split_voice_data.PICKERS if name in v3.library_ui.library]
    if not settings["voices"]:
        return {"rows": [], "error": "", "sources": sources}

    try:
        rows = pick_voices(settings["voices"], settings["voices_from"], settings["voices_split"],
                           settings["voices_min"], settings["voices_max"], settings["voices_seed"])
    except ValueError as error:
        return {"rows": [], "error": str(error), "sources": sources}

    return {"rows": rows, "error": "", "sources": sources, "groups": split_groups(rows, settings["voices_split"])}


def split_groups(rows, split):
    """
    The voices in blocks for the page. From one split they are one block; from both,
    the validation voices first, then the final test ones, each block labelled.
    """
    if split != "all":
        return [{"split": split, "label": "", "rows": rows}]

    by_split = {}
    for clip in rows:
        by_split.setdefault(split_voice_data.split_of(clip.dataset, clip.speaker), []).append(clip)

    order = [*VOICE_SPLITS_ONLY, None]
    return [{"split": name, "label": VOICE_SPLITS.get(name, "In no split"), "rows": by_split[name]}
            for name in order if by_split.get(name)]


@app.context_processor
def sampler_helpers():
    return {
        "random_voices": random_voices,
        "voice_values": voice_values,
        "voice_fields": list(voice_values(read_settings(MultiDict()))),
        "voice_splits": VOICE_SPLITS,
        "split_names": SPLIT_NAMES,
        "test_percent": TEST_PERCENT,
        "split_sizes": SPLIT_SIZES,
        "from_all": VOICES_FROM_ALL,
        "voices_shown": VOICES_SHOWN,
    }


@app.route("/random", methods=["POST"])
def random_pick():
    """Pick the voices. Without a seed a new one is drawn, and it goes in the URL so the pick can be repeated."""
    s = read_settings(MultiDict(parse_qsl(request.form.get("state", ""))))
    fields = read_settings(request.form)
    seed = request.form.get("voices_seed", "").strip()

    if fields["voices"] < 1:
        return v3.back(s, error="Ask for at least one random voice.")

    s.update({name: fields[name] for name in voice_values(fields)})
    s["voices_seed"] = fields["voices_seed"] if seed else secrets.randbelow(10_000)
    return v3.back(s)


# --------------------------------------------------------------------------- #
# Examine: a generated clip measured against its reference
# --------------------------------------------------------------------------- #


def generated_wav(relative):
    """The wav for a path like "1001/1001_20261103_103302_543.wav", only if it is a clip in OUTPUT_DIR."""
    root = v3.generator_ui.OUTPUT_DIR.resolve()
    path = (root / relative).resolve()
    if path.parent.parent != root or path.suffix != ".wav" or not path.is_file():
        return None
    return path


def library_clip(record):
    """The library clip a record names in "reference_clip" (dataset/clip id), or None."""
    dataset, _, clip_id = record.get("reference_clip", "").partition("/")
    return v3.library_ui.library.get(dataset, {}).get("by_id", {}).get(clip_id)


def origin_of(record):
    """
    Where the voice comes from: saved with the clip, or worked out for clips made before
    that. The split is always looked up again, so a clip made under an older split
    shows the speaker's part of the current one.
    """
    if record.get("origin"):
        origin = dict(record["origin"])
        if origin.get("dataset") and origin.get("speaker"):
            origin["split"] = split_voice_data.split_of(origin["dataset"], origin["speaker"])
        return origin
    clip = library_clip(record)
    return split_voice_data.voice_origin(clip) if clip else None


def origin_text(origin):
    if not origin:
        return ""
    parts = [origin.get("dataset"), origin.get("speaker"), origin.get("clip_id"),
             *(origin.get("details") or {}).values()]
    split = origin.get("split")
    parts.append(SPLIT_NAMES.get(split, f"{split} split") if split else "in no split")
    return " · ".join(str(part) for part in parts if part)


def read_record(path):
    info = path.with_suffix(".json")
    return json.loads(info.read_text(encoding="utf-8")) if info.is_file() else None


def clip_info(item):
    """For the page: the clip's path, where its voice comes from, and its newest measurement if any."""
    relative = item["audio"].removeprefix("/generated/")
    path = generated_wav(relative)
    record = read_record(path) if path else None
    if record is None:
        return None

    # The node the clip links to; a clip measured outside this page is found by its checksum
    node = measurements.load_node(node=record["measurements"]) if record.get("measurements") else None
    node = node or measurements.load_node(audio=path)
    runs = node["runs"] if node else []

    return {
        "path": relative,
        "origin_text": origin_text(origin_of(record)),
        "metrics": runs[-1]["metrics"] if runs else None,
        "node": record.get("measurements") or (measurements.shown(measurements.node_path(path)) if node else None),
        "runs": len(runs),
    }


def examine_clip(path, force=False):
    """
    The clip's metrics against its reference, from its node if it was measured with the same inputs
    (unless force), otherwise measured now. Its .json then links to the node and records the origin.
    """
    record = read_record(path)
    if record is None or not record.get("target_text"):
        raise ValueError("There is no record of what this clip should say, so it cannot be measured.")

    reference = record.get("reference_audio")
    if not reference or not Path(reference).is_file():
        clip = library_clip(record)
        if clip is None:
            raise ValueError("The reference recording of this clip cannot be found.")
        reference = get_clip(clip)[0]

    result = measurements.measure(reference, path, record["target_text"], record.get("reference_text"), force=force)
    record["measurements"] = result["node"]
    record["origin"] = origin_of(record)
    record.pop("metrics", None)     # kept in the node now, not here
    path.with_suffix(".json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


@app.context_processor
def examine_helpers():
    return {"clip_info": clip_info}


@app.route("/examine", methods=["POST"])
def examine():
    s = read_settings(MultiDict(parse_qsl(request.form.get("state", ""))))
    relative = request.form.get("clip", "")
    path = generated_wav(relative)

    if path is None:
        return v3.back(s, error="That clip cannot be found.")

    try:
        # The models share the GPU with generating, so one at a time
        with v3.generate_lock:
            result = examine_clip(path, force=request.form.get("force") == "1")
    except Exception as error:
        import traceback
        traceback.print_exc()
        return v3.back(s, error=f"Examining failed: {type(error).__name__}: {error}")

    message = "Already measured with the same inputs: shown from its saved node, nothing was run." if result["cached"] else ""
    return v3.back(s, examined=relative, message=message)


failed_logins = 0
locked_until = 0.0
login_lock = threading.Lock()

LOGIN_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice Generator V4: log in</title>
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, sans-serif; display: grid; place-items: center; min-height: 90vh; margin: 0; padding: 16px; }
  form { display: grid; gap: 10px; width: min(320px, 100%); }
  h1 { font-size: 1.2rem; margin: 0; }
  input, button { font: inherit; padding: 8px 10px; }
  .error { margin: 0; color: #c0392b; }
</style>
</head>
<body>
<form method="post" action="/login">
  <h1>Voice Generator V4</h1>
  {% if error %}<p class="error">{{ error }}</p>{% endif %}
  <label for="password">Password</label>
  <input id="password" name="password" type="password" autocomplete="new-password" autofocus required>
  <label><input type="checkbox" onchange="document.getElementById('password').type = this.checked ? 'text' : 'password'"> Show password</label>
  <button type="submit">Log in</button>
</form>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# The password
# --------------------------------------------------------------------------- #


def password():
    """UI_PASSWORD from .env. If it is missing, make a strong one and save it there."""
    value = os.environ.get("UI_PASSWORD", "").strip()

    if not value:
        value = secrets.token_urlsafe(18)
        with open(ENV_FILE, "a", encoding="utf-8") as env:
            env.write(f"\n# The password of Voice_model_example_UI_V4.py\nUI_PASSWORD={value}\n")
        print(f"Made a new password: it is UI_PASSWORD in {ENV_FILE}")

    if len(value) < MIN_PASSWORD_LENGTH:
        raise SystemExit(f"UI_PASSWORD in .env is too short: use at least {MIN_PASSWORD_LENGTH} characters.")

    return value


def check_password(given):
    """(right, seconds to wait). While locked, even the right password is refused."""
    global failed_logins, locked_until

    with login_lock:
        now = time.monotonic()
        if now < locked_until:
            return False, locked_until - now

        if hmac.compare_digest(given.encode("utf-8"), app.config["UI_PASSWORD"].encode("utf-8")):
            failed_logins = 0
            return True, 0

        failed_logins += 1
        if failed_logins < FREE_TRIES:
            return False, 0

        wait = min(LOCK_SECONDS * 2 ** (failed_logins - FREE_TRIES), MAX_LOCK_SECONDS)
        locked_until = now + wait
        return False, wait


# --------------------------------------------------------------------------- #
# Every request passes through here
# --------------------------------------------------------------------------- #


@app.before_request
def require_login():
    # A form on another website must not be able to act for someone logged in here
    origin = request.headers.get("Origin")
    if request.method == "POST" and origin and urlsplit(origin).netloc != request.host:
        abort(403)

    if request.endpoint != "login" and not session.get("logged_in"):
        return redirect("/login")


@app.after_request
def safety_headers(response):
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    # The settings and clip names are in the URL: do not pass it to other sites.
    # Not "no-referrer": with that, browsers send Origin: null and every form is refused.
    response.headers["Referrer-Policy"] = "same-origin"
    if response.mimetype == "text/html":
        response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return redirect("/") if session.get("logged_in") else render_template_string(LOGIN_PAGE, error=None)

    # Copying from .env can bring the line's name or a space along
    given = request.form.get("password", "").strip().removeprefix("UI_PASSWORD=").strip()
    right, wait = check_password(given)
    if not right:
        # Never the password itself: enough to tell a typo from a different password
        print(f"Login refused: {len(given)} characters given, {len(app.config['UI_PASSWORD'])} expected")
    if right:
        session.clear()
        session.permanent = True
        session["logged_in"] = True
        return redirect("/")

    message = f"Too many wrong tries. Try again in {int(wait) + 1} seconds." if wait else "Wrong password."
    return render_template_string(LOGIN_PAGE, error=message), 429 if wait else 401


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect("/login")


# --------------------------------------------------------------------------- #
# The certificate
# --------------------------------------------------------------------------- #


def make_certificate(cert_path, key_path):
    """A self-signed certificate for this program, valid for two years."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Voice Generator V4")])
    now = datetime.now(timezone.utc)
    names = [x509.DNSName("localhost"), x509.IPAddress(ip_address("127.0.0.1"))]
    certificate = (x509.CertificateBuilder()
                   .subject_name(name).issuer_name(name)
                   .public_key(key.public_key())
                   .serial_number(x509.random_serial_number())
                   .not_valid_before(now - timedelta(minutes=5))
                   .not_valid_after(now + timedelta(days=730))
                   .add_extension(x509.SubjectAlternativeName(names), critical=False)
                   # iPhones and Macs refuse a server certificate that does not say it is one
                   .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
                   .sign(key, hashes.SHA256()))

    CERT_DIR.mkdir(exist_ok=True)
    key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                           serialization.NoEncryption()))
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))


class HandshakeOnItsOwnThread(ssl.SSLContext):
    """By default the handshake runs while accepting, so a silent visitor stops all the others."""

    def wrap_socket(self, sock, *args, **kwargs):
        kwargs["do_handshake_on_connect"] = False
        return super().wrap_socket(sock, *args, **kwargs)


class ClosesIdleConnections(WSGIRequestHandler):
    # The limit is on the connection going quiet: a long generation is not affected
    timeout = IDLE_SECONDS

    def handle(self):
        # Every HTTPS connection opens with this byte. socket.socket: the raw bytes, not decrypted ones
        try:
            first = socket.socket.recv(self.connection, 1, socket.MSG_PEEK)
        except OSError:
            return

        if first == b"\x16":
            super().handle()
        elif first:
            self.send_to_https()

    def send_to_https(self):
        """Answers a plain http visit with where to go. Nothing else is ever served unencrypted."""
        try:
            sent = socket.socket.recv(self.connection, 4096).decode("latin-1")
            host = re.search(r"^Host: *([\w.:\[\]-]+)\r?$", sent, re.M | re.I)
            answer = (f"HTTP/1.1 302 Found\r\nLocation: https://{host.group(1)}/\r\n" if host else
                      "HTTP/1.1 400 Bad Request\r\n")
            socket.socket.sendall(self.connection, f"{answer}Content-Length: 0\r\nConnection: close\r\n\r\n".encode("latin-1"))
        except OSError:
            pass


def tls_context():
    """TLS 1.2 or newer, with UI_TLS_CERT and UI_TLS_KEY, or else this program's own certificate."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes

    own_certificate = os.environ.get("UI_TLS_CERT", "").strip()
    cert_path = Path(own_certificate or CERT_DIR / "ui_cert.pem")
    key_path = Path(os.environ.get("UI_TLS_KEY", "").strip() or CERT_DIR / "ui_key.pem")

    if not cert_path.is_file() or not key_path.is_file():
        if own_certificate:
            raise SystemExit(f"Certificate or key not found: {cert_path}, {key_path} (UI_TLS_CERT, UI_TLS_KEY in .env)")
        make_certificate(cert_path, key_path)

    context = HandshakeOnItsOwnThread(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(cert_path, key_path)

    certificate = x509.load_pem_x509_certificate(cert_path.read_bytes())
    print(f"Certificate fingerprint (SHA-256): {certificate.fingerprint(hashes.SHA256()).hex(':').upper()}")
    return context


if __name__ == "__main__":
    # Windows lets two programs listen on one port, and then visitors reach either at random
    with socket.socket() as probe:
        if probe.connect_ex(("127.0.0.1", PORT)) == 0:
            raise SystemExit(f"V4 is already running on port {PORT}. Close that one first, or keep using it.")

    app.config["UI_PASSWORD"] = password()
    context = tls_context()

    v3.library_ui.load_library()

    if not v3.library_ui.library:
        raise SystemExit(f"No dataset folders found in {v3.library_ui.LIBRARY_DIR}. Check VOICE_LIBRARY_DIR in .env.")

    # Speech lengths, shared with V2's helpers (read from output/picky/)
    for name, data in v3.library_ui.library.items():
        v3.v2.seconds[name] = v3.measure(data["clips"])

    # Load CosyVoice now, in the background, so the first Generate does not wait for it.
    # A Generate pressed before it is ready waits for this load instead of starting another.
    def preload():
        started = time.time()
        try:
            model.voice_model()
            print(f"CosyVoice 3 is ready ({time.time() - started:.0f} s)")
        except Exception as error:
            print(f"CosyVoice 3 did not load in the background ({error}); the first Generate will try again")

    threading.Thread(target=preload, name="preload-cosyvoice", daemon=True).start()

    # Only this PC can connect directly; a tunnel's agent runs on this PC too.
    # The debugger stays off.
    print(f"Open https://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True, ssl_context=context,
            request_handler=ClosesIdleConnections)

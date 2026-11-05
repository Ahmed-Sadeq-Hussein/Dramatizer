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
can stay inside one split. Press Use on a voice to make it the reference; the list
stays while you go through them. The seed is shown, and the same settings and seed
give the same voices. Custom voices are never picked.

    python Voice_model_example_UI_V4.py
    then open https://127.0.0.1:5004

Through playit.gg: add a TCP tunnel to local address 127.0.0.1, port 5004, then
open https://<the address playit gives>:<its port>. A visit without https:// is
sent there: phones fall back to plain http when they do not trust the certificate.

Reused, not copied: all of V3. Only this page is protected: the older UIs
(ports 5000-5003) have no login, so do not put a tunnel in front of them.
"""
import hmac
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
import split_voice_data
from Picky_voice import MAX_SECONDS, MIN_SECONDS
from Voice_Importer import read_custom_voices

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
VOICE_SPLITS = {"train": "Train", "validation": "Validation", "test": "Final test", "all": "Every speaker"}
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
  .sampler td audio { width: 150px; }
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
    {% if sample.error %}<p class="note error">{{ sample.error }}</p>{% endif %}
    {% if sample.rows %}
      <p class="summary">Seed {{ s.voices_seed }}, {{ voice_splits[s.voices_split]|lower }}. The same settings and seed give the same voices.</p>
      <div class="table-wrap">
        <table>
          <thead><tr><th></th><th>Voice</th><th>Details</th><th>Script</th><th>Listen</th></tr></thead>
          <tbody>
            {% for clip in sample.rows %}
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
    {% endif %}
  </details>
"""

# So that Apply in the filters does not lose the random voices
SAMPLER_KEPT = """{% if s.voices %}{% for name in voice_fields %}<input type="hidden" name="{{ name }}" value="{{ voice_values(s)[name] }}">{% endfor %}{% endif %}
    """

# The page is V3's, with its own name, a Log out button, the random voices and the phone tabs
v3.PAGE = (v3.PAGE
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


@lru_cache(maxsize=32)
def pick_voices(number, source, split, min_seconds, max_seconds, seed):
    library = v3.library_ui.library
    missing = [name for name in split_voice_data.PICKERS if name not in library]
    split = None if split == "all" else split

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

    return {"rows": rows, "error": "", "sources": sources}


@app.context_processor
def sampler_helpers():
    return {
        "random_voices": random_voices,
        "voice_values": voice_values,
        "voice_fields": list(voice_values(read_settings(MultiDict()))),
        "voice_splits": VOICE_SPLITS,
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

    # Only this PC can connect directly; a tunnel's agent runs on this PC too.
    # The debugger stays off.
    print(f"Open https://127.0.0.1:{PORT}")
    app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True, ssl_context=context,
            request_handler=ClosesIdleConnections)

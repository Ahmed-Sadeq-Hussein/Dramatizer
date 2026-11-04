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
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlsplit

from flask import abort, redirect, render_template_string, request, session
from werkzeug.serving import WSGIRequestHandler

import Voice_model_example_UI_V3 as v3
import model
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
  @media (max-width: 860px) {
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

# The page is V3's, with its own name, a Log out button and the phone tabs
v3.PAGE = (v3.PAGE
           .replace("<title>Voice Generator V3</title>", "<title>Voice Generator V4</title>")
           .replace('<span class="version">V3</span>', '<span class="version">V4</span>')
           .replace("</header>", """  <form method="post" action="/logout"><button type="submit">Log out</button></form>
</header>""", 1)
           .replace("</head>", PHONE_STYLE + "</head>", 1)
           .replace('<form method="get" action="/" class="picky"',
                    '<button type="button" class="filters-toggle quiet" aria-expanded="false">Filters</button>\n'
                    '  <form method="get" action="/" class="picky"', 1)
           .replace("<h3>Recent</h3>", '</section>\n  <section class="recent-part" aria-label="Recent">\n    <h3>Recent</h3>', 1)
           .replace("</body>", PHONE_TABS + "</body>", 1))

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

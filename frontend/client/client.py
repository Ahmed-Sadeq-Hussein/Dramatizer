"""client.py | The front end for someone else's PC: a page in their browser that uses the Dramatizer API.

    python main_frontend.py connect https://<playit address>:<port>

Then the page opens at http://127.0.0.1:5010. Nothing heavy runs here: no models,
no voice data, no GPU. Every action is sent to the API on the host's PC.

The first time, it asks for:

    the API key        given to you by the host (one per person)
    the fingerprint    of the host's certificate: the host reads it to you, and the client
                       refuses to talk to anything else, so nobody in between can listen in

Both are saved in ~/.dramatizer_client.json, so next time only the address is
needed (or nothing: the last address is remembered). --forget deletes that file.

Only needs Python with flask and requests:  pip install flask requests
"""
import argparse
import getpass
import hashlib
import json
import socket
import ssl
import sys
import threading
import webbrowser
from pathlib import Path
from urllib.parse import urlsplit

import requests
import urllib3
from flask import Flask, Response, abort, request, send_from_directory
from requests.adapters import HTTPAdapter

HERE = Path(__file__).resolve().parent
CONFIG = Path.home() / ".dramatizer_client.json"
LOCAL_PORT = 5010
API = "/api/v1"
TIMEOUT = 120         # the first voice list after the host starts can take a while (the library is read from disk)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# --------------------------------------------------------------------------- #
# Talking to the host, safely
# --------------------------------------------------------------------------- #

def normal_fingerprint(text):
    hex_only = "".join(c for c in str(text) if c in "0123456789abcdefABCDEF").lower()
    if len(hex_only) != 64:
        raise ValueError("A fingerprint is 64 hexadecimal characters (pairs like AB:CD:..).")
    return hex_only


def shown_fingerprint(hex_only):
    return ":".join(hex_only[i:i + 2] for i in range(0, 64, 2)).upper()


def server_fingerprint(url):
    """The SHA-256 fingerprint of the certificate the server shows right now."""
    parts = urlsplit(url)
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    with socket.create_connection((parts.hostname, parts.port or 443), timeout=15) as sock:
        with context.wrap_socket(sock, server_hostname=parts.hostname) as tls:
            return hashlib.sha256(tls.getpeercert(binary_form=True)).hexdigest()


class Pinned(HTTPAdapter):
    """Accept only the host's own certificate, by its fingerprint (it is self-signed, so no CA can vouch for it)."""

    def __init__(self, fingerprint, **kwargs):
        self.fingerprint = fingerprint
        super().__init__(**kwargs)

    def init_poolmanager(self, *args, **kwargs):
        kwargs["assert_fingerprint"] = self.fingerprint
        super().init_poolmanager(*args, **kwargs)


def session_for(url, key, fingerprint):
    session = requests.Session()
    session.mount(url.rstrip("/") + "/", Pinned(fingerprint))
    session.verify = False          # the fingerprint above is the check, not a CA
    session.headers["Authorization"] = f"Bearer {key}"
    return session


# --------------------------------------------------------------------------- #
# Settings: address, key, fingerprint
# --------------------------------------------------------------------------- #

def load_config():
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_config(config):
    CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")


def setup(url, key=None, fingerprint=None):
    """Ask for what is missing, check it against the host, and remember it."""
    config = load_config()
    url = (url or config.get("url") or "").strip().rstrip("/")
    if not url:
        raise SystemExit("Give the host's address: python main_frontend.py connect https://<playit address>:<port>")
    if not url.startswith("https://"):
        url = "https://" + url.split("://", 1)[-1]
    same_host = config.get("url") == url

    seen = server_fingerprint(url)
    if fingerprint:
        expected = normal_fingerprint(fingerprint)
    elif same_host and config.get("fingerprint"):
        expected = config["fingerprint"]
    else:
        print(f"\nThe host at {url} shows this certificate fingerprint:\n\n    {shown_fingerprint(seen)}\n")
        answer = input("Is it exactly the one the host gave you? Type yes to trust it: ").strip().lower()
        if answer != "yes":
            raise SystemExit("Not connected. Ask the host for the fingerprint (python main_backend.py api --fingerprint).")
        expected = seen
    if seen != expected:
        raise SystemExit("STOP: the host's certificate is not the one you trusted. Someone may be in between, or the "
                         "host made a new certificate. Ask the host for the new fingerprint and run with "
                         "--fingerprint, or --forget to start over.")

    key = key or (config.get("key") if same_host else None)
    if not key:
        key = getpass.getpass("Your API key (from the host; it is not shown while you type): ").strip()

    session = session_for(url, key, expected)
    try:
        reply = session.get(url + API + "/me", timeout=TIMEOUT)
    except requests.RequestException as error:
        raise SystemExit(f"Cannot reach the API at {url}: {error}")
    if reply.status_code == 401:
        raise SystemExit("The host does not accept this key. Ask for a new one, then run with --key or --forget.")
    reply.raise_for_status()

    save_config({"url": url, "key": key, "fingerprint": expected})
    return url, session, reply.json()["name"]


# --------------------------------------------------------------------------- #
# The local page, passing every /api call on to the host
# --------------------------------------------------------------------------- #

def make_app(url, session, name):
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

    @app.before_request
    def same_page_only():
        # Another website open in this browser must not be able to use the key through this app
        origin = request.headers.get("Origin")
        if request.method != "GET" and origin and urlsplit(origin).netloc != request.host:
            abort(403)

    @app.get("/")
    def index():
        return send_from_directory(HERE / "static", "index.html")

    @app.get("/static/<path:name>")
    def static_file(name):
        return send_from_directory(HERE / "static", name)

    @app.get("/connection")
    def connection():
        return {"host": urlsplit(url).netloc, "name": name}

    @app.route(f"{API}/<path:path>", methods=["GET", "POST"])
    def forward(path):
        try:
            reply = session.request(request.method, f"{url}{API}/{path}", params=request.args,
                                    data=request.get_data() if request.method == "POST" else None,
                                    headers={"Content-Type": "application/json"} if request.method == "POST" else None,
                                    timeout=TIMEOUT, stream=True)
        except requests.exceptions.SSLError:
            return {"error": "The host's certificate changed: stopped for safety. Restart the client."}, 502
        except requests.RequestException as error:
            return {"error": f"The host cannot be reached ({type(error).__name__}). Is its API and tunnel running?"}, 502
        kept = {k: v for k, v in reply.headers.items() if k.lower() in ("content-type", "content-disposition")}
        return Response(reply.iter_content(64 * 1024), status=reply.status_code, headers=kept)

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description="Use the Dramatizer API from this PC, in your browser.")
    parser.add_argument("url", nargs="?", help="the host's address, https://<playit address>:<port>")
    parser.add_argument("--key", help="your API key (asked for if not given)")
    parser.add_argument("--fingerprint", help="the host's certificate fingerprint (asked to confirm if not given)")
    parser.add_argument("--port", type=int, default=LOCAL_PORT, help=f"the local page's port (default {LOCAL_PORT})")
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    parser.add_argument("--forget", action="store_true", help=f"delete the saved address, key and fingerprint ({CONFIG})")
    args = parser.parse_args(argv)

    if args.forget:
        CONFIG.unlink(missing_ok=True)
        print(f"Forgotten: {CONFIG}")
        if not args.url:
            return
    url, session, name = setup(args.url, args.key, args.fingerprint)
    page = f"http://127.0.0.1:{args.port}"
    print(f"\nConnected to {url} as {name}. Open {page} (Ctrl+C to stop).\n", flush=True)
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(page)).start()
    make_app(url, session, name).run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main(sys.argv[1:])

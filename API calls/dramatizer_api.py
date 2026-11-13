"""dramatizer_api.py | Pull data from the Dramatizer API and save it: JSON as .json, audio as .wav.

Only Python's standard library, so it runs with any Python 3.8+ (no project install needed).

    python dramatizer_api.py --url https://127.0.0.1:5005 --fingerprint AB:CD:... --user test_user_1 health
    python dramatizer_api.py ... get genome/searches                     # any GET route, saved as JSON
    python dramatizer_api.py ... get "voices?dataset=VCTK&page=2"
    python dramatizer_api.py ... search mos-diverse-v1_p1_CREMA-D_1014   # one search: JSON and every clip
    python dramatizer_api.py ... searches                                # every search you may see
    python dramatizer_api.py ... searches --no-audio                     # the same, JSON only

Who you are (one of):
    --user NAME      an account; the password is asked for, or read from DRAMATIZER_PASSWORD
    --key KEY        an API key from the host (python main_backend.py api-key add NAME), or DRAMATIZER_KEY

The server's certificate is its own (self-signed), so no certificate authority can vouch for it.
Instead this script checks its SHA-256 fingerprint against --fingerprint (or DRAMATIZER_FINGERPRINT),
the one the host reads to you (python main_backend.py api --fingerprint). A different certificate
means someone in between, or a new certificate on the host: the script stops before sending anything.

Everything is saved under --out (default: downloads/ next to this file, git-ignored).
"""
import argparse
import getpass
import hashlib
import http.client
import json
import os
import ssl
import sys
from pathlib import Path
from urllib.parse import quote, urlsplit

API = "/api/v1"
HERE = Path(__file__).resolve().parent


class Api:
    def __init__(self, url, fingerprint):
        parts = urlsplit(url if "://" in url else "https://" + url)
        if parts.scheme != "https":
            raise SystemExit("The API speaks HTTPS only: give an https:// address.")
        self.host, self.port = parts.hostname, parts.port or 443
        self.fingerprint = "".join(c for c in (fingerprint or "") if c in "0123456789abcdefABCDEF").lower()
        if len(self.fingerprint) != 64:
            raise SystemExit("Give the host's certificate fingerprint (--fingerprint AB:CD:..., 64 hex digits). "
                             "The host prints it with: python main_backend.py api --fingerprint")
        self.token = None

    def _connect(self):
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE          # checked by fingerprint below instead
        conn = http.client.HTTPSConnection(self.host, self.port, context=context, timeout=60)
        conn.connect()
        seen = hashlib.sha256(conn.sock.getpeercert(binary_form=True)).hexdigest()
        if seen != self.fingerprint:
            conn.close()
            raise SystemExit("The server's certificate does not match the fingerprint. Nothing was sent. "
                             "Ask the host for the current fingerprint.")
        return conn

    def request(self, method, path, body=None):
        """(status, content type, bytes)."""
        conn = self._connect()
        headers = {"Accept": "application/json, audio/wav"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        data = None
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        try:
            conn.request(method, API + "/" + path.lstrip("/"), body=data, headers=headers)
            reply = conn.getresponse()
            return reply.status, reply.getheader("Content-Type", ""), reply.read()
        finally:
            conn.close()

    def json(self, method, path, body=None):
        status, _, data = self.request(method, path, body)
        try:
            parsed = json.loads(data or b"{}")
        except ValueError:
            parsed = {}
        if status >= 400:
            raise SystemExit(f"{method} {path}: {status} {parsed.get('error', '')}".strip())
        return parsed

    def log_in(self, user=None, key=None):
        if key:
            self.token = key
            return
        password = os.environ.get("DRAMATIZER_PASSWORD") or getpass.getpass(f"Password for {user}: ")
        self.token = self.json("POST", "login", {"username": user, "password": password})["token"]

    def log_out(self, user):
        if user and self.token:
            try:
                self.request("POST", "logout", {})
            except Exception:
                pass


def safe(name):
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name).strip("_") or "reply"


def save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  saved {path}")


def save_audio(api, path_in_api, path):
    status, kind, data = api.request("GET", path_in_api)
    if status != 200 or not data.startswith(b"RIFF"):
        print(f"  skipped {path_in_api}: {status}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return True


def get(api, route, out, name=None):
    """Any GET route: JSON is saved as .json, audio as .wav."""
    status, kind, data = api.request("GET", route)
    if status >= 400:
        raise SystemExit(f"GET {route}: {status} {data[:200].decode(errors='replace')}")
    if data.startswith(b"RIFF"):
        path = out / f"{name or safe(route)}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(f"  saved {path}")
    else:
        save_json(json.loads(data), out / f"{name or safe(route)}.json")


def one_search(api, run_id, out, audio=True):
    """A search's JSON (genomes, verdicts, scores, ranking) and, with audio, the plain line and every genome."""
    folder = out / "searches" / safe(run_id)
    data = api.json("GET", f"genome/searches/{quote(run_id)}")
    save_json(data, folder / "search.json")
    if not audio:
        return
    names = ["baseline.wav"] + [g["audio"] for g in data["run"].get("genomes", []) if g.get("audio")]
    done = sum(save_audio(api, f"genome/searches/{quote(run_id)}/audio/{quote(n)}", folder / n) for n in names)
    print(f"  {done} of {len(names)} clips -> {folder}")


def main():
    parser = argparse.ArgumentParser(description="Pull data from the Dramatizer API and save it.")
    parser.add_argument("--url", default=os.environ.get("DRAMATIZER_URL", "https://127.0.0.1:5005"))
    parser.add_argument("--fingerprint", default=os.environ.get("DRAMATIZER_FINGERPRINT"))
    parser.add_argument("--user", default=os.environ.get("DRAMATIZER_USER"), help="log in as this account")
    parser.add_argument("--key", default=os.environ.get("DRAMATIZER_KEY"), help="or use an API key")
    parser.add_argument("--out", type=Path, default=HERE / "downloads", help="where to save (default: downloads/)")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("health", help="is the API up? (no login needed)")
    one = sub.add_parser("get", help="any GET route under /api/v1, e.g. genome/searches")
    one.add_argument("route")
    one.add_argument("--name", help="the file name to save as (default: from the route)")
    s = sub.add_parser("search", help="one search: its JSON and its clips")
    s.add_argument("run_id")
    s.add_argument("--no-audio", action="store_true")
    every = sub.add_parser("searches", help="every search you may see: the list, each one's JSON and clips")
    every.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()

    api = Api(args.url, args.fingerprint)
    if args.action == "health":
        print(api.json("GET", "health"))
        return
    if not (args.user or args.key):
        raise SystemExit("Say who you are: --user NAME (an account) or --key KEY (an API key).")
    api.log_in(args.user, args.key)
    try:
        if args.action == "get":
            get(api, args.route, args.out, args.name)
        elif args.action == "search":
            one_search(api, args.run_id, args.out, not args.no_audio)
        else:
            listing = api.json("GET", "genome/searches")
            save_json(listing, args.out / "searches.json")
            for run in listing["searches"]:
                print(run["id"])
                one_search(api, run["id"], args.out, not args.no_audio)
    finally:
        api.log_out(args.user)


if __name__ == "__main__":
    sys.exit(main())

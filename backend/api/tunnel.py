"""tunnel.py | Put the website and API on a public https:// address with a trusted certificate (Cloudflare).

    python main_backend.py api          # first, in its own window: the API on https://127.0.0.1:5005
    python main_backend.py tunnel       # then this: prints the public address, e.g. https://xyz.trycloudflare.com

Anyone can then open that address on a phone or computer, log in, and use the website: no client app,
no fingerprint, no certificate warning. It runs until you stop it (Ctrl+C); the address is new each time.

How it works: cloudflared (Cloudflare's tunnel program, installed with
`winget install --id Cloudflare.cloudflared`) connects out from this PC to Cloudflare, and Cloudflare
gives a random trycloudflare.com address with its own trusted certificate. Requests go phone ->
Cloudflare -> cloudflared on this PC -> the API on 127.0.0.1:5005.

What to know:
    - Cloudflare decrypts the traffic in its network (that is how it gives a trusted certificate).
      The playit tunnel with the pinned fingerprint (SHARING.md) stays end-to-end encrypted; both work.
    - cloudflared talks to the API on this same PC, where the API's own certificate cannot be checked
      by name (it is self-signed), so --no-tls-verify is used for that local hop only.
    - A quick tunnel needs no Cloudflare account. A fixed address needs an account and a domain
      (Cloudflare's "named tunnels"); --url and the rest of the API stay the same.
"""
import argparse
import re
import shutil
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

from backend.api.server import PORT

PLACES = [Path(r"C:\Program Files (x86)\cloudflared\cloudflared.exe"), Path(r"C:\Program Files\cloudflared\cloudflared.exe")]
ADDRESS = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def find_cloudflared():
    found = shutil.which("cloudflared")
    if found:
        return found
    for place in PLACES:
        if place.is_file():
            return str(place)
    raise SystemExit("cloudflared is not installed. Install it with:\n\n"
                     "    winget install --id Cloudflare.cloudflared\n\nthen open a new window and run this again.")


def api_is_up(port):
    context = ssl.create_default_context()
    context.check_hostname, context.verify_mode = False, ssl.CERT_NONE      # only asks this PC whether the API runs
    try:
        with urllib.request.urlopen(f"https://127.0.0.1:{port}/api/v1/health", context=context, timeout=5) as reply:
            return reply.status == 200
    except OSError:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description="A public https:// address for the website and API (Cloudflare).")
    parser.add_argument("--port", type=int, default=PORT, help=f"the API's local port (default {PORT})")
    args = parser.parse_args(argv)
    if not api_is_up(args.port):
        raise SystemExit(f"The API is not running on https://127.0.0.1:{args.port}. Start it first, in another "
                         "window:  python main_backend.py api")
    command = [find_cloudflared(), "tunnel", "--no-autoupdate", "--url", f"https://127.0.0.1:{args.port}",
               "--no-tls-verify"]
    print("Starting the Cloudflare tunnel (Ctrl+C stops it)...", flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    shown = False
    try:
        for line in process.stdout:
            found = ADDRESS.search(line)
            if found and not shown:
                shown = True
                print(f"\n  The website is open at:  {found.group(0)}\n\n"
                      "  Send that address to the people who rate. They log in with their own account.\n"
                      "  It changes every time the tunnel starts.\n", flush=True)
            elif "ERR" in line or "error" in line.lower():
                print("  cloudflared: " + line.strip(), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        process.terminate()
    return process.wait()


if __name__ == "__main__":
    sys.exit(main())

"""tunnel.py | Put the website and API on a public https:// address with a trusted certificate (Cloudflare).

    python main_backend.py api          # first, in its own window: the API on https://127.0.0.1:5005
    python main_backend.py tunnel       # then this: the fixed address (once set up), or a random one

Anyone can then open that address on a phone or computer, log in, and use the website: no client app,
no fingerprint, no certificate warning. It runs until you stop it (Ctrl+C).

The fixed address (a "named tunnel" on our own domain, dramatizer.dev), set up once:

    cloudflared tunnel login                                        # you: sign in, pick dramatizer.dev
    python main_backend.py tunnel --setup app.dramatizer.dev --port 5105
    python main_backend.py tunnel                                   # from now on: https://app.dramatizer.dev

--setup makes the tunnel "dramatizer" (once), points the name at it in Cloudflare's DNS, and writes
~/.cloudflared/config.yml. Run it again to change the port or the name; the tunnel itself is kept. The login
certificate and the tunnel's credentials stay in ~/.cloudflared/: they are secrets and never go in the
repository. Without a config.yml, or with --quick, a quick tunnel gives a new random trycloudflare.com address.

How it works: cloudflared (Cloudflare's tunnel program, installed with
`winget install --id Cloudflare.cloudflared`) connects out from this PC to Cloudflare, and Cloudflare
serves app.dramatizer.dev (or a random trycloudflare.com address) with its own trusted certificate.
Requests go phone -> Cloudflare -> cloudflared on this PC -> the API on 127.0.0.1 (the port in config.yml).
Nothing listens for the internet on this PC: the tunnel is an outgoing connection.

What to know:
    - Cloudflare decrypts the traffic in its network (that is how it gives a trusted certificate).
      The playit tunnel with the pinned fingerprint (SHARING.md) stays end-to-end encrypted; both work.
    - cloudflared talks to the API on this same PC, where the API's own certificate cannot be checked
      by name (it is self-signed), so --no-tls-verify is used for that local hop only.
    - A quick tunnel needs no Cloudflare account. The fixed address needs the account that owns
      dramatizer.dev (bought 23 Nov 2026) and the one-time --setup above; the API itself is the same.
"""
import argparse
import json
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
HOME = Path.home() / ".cloudflared"
LOGIN = HOME / "cert.pem"                 # written by `cloudflared tunnel login`
CONFIG = HOME / "config.yml"              # written by --setup
NAME = "dramatizer"
HOSTNAME = re.compile(r"^(?=.{1,253}$)([a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}$")


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


def config_text(tunnel_id, credentials, hostname, port):
    """~/.cloudflared/config.yml: this hostname to the API on this PC; anything else gets a 404."""
    return (f"# Written by `python main_backend.py tunnel --setup` (backend/api/tunnel.py). No secrets in here: the\n"
            f"# credentials file it names is the secret, and stays in this folder.\n"
            f"tunnel: {tunnel_id}\n"
            f"credentials-file: {credentials}\n"
            f"ingress:\n"
            f"  - hostname: {hostname}\n"
            f"    service: https://127.0.0.1:{port}\n"
            f"    originRequest:\n"
            f"      noTLSVerify: true        # the API's own certificate is self-signed; this hop never leaves this PC\n"
            f"  - service: http_status:404\n")


def read_config():
    """{"tunnel", "hostname", "port"} from config.yml, or None."""
    if not CONFIG.is_file():
        return None
    text = CONFIG.read_text(encoding="utf-8")
    tunnel, host, port = (re.search(pattern, text, re.M) for pattern in
                          (r"^tunnel:\s*(\S+)", r"hostname:\s*(\S+)", r"service:\s*https://127\.0\.0\.1:(\d+)"))
    if not (tunnel and host and port):
        return None
    return {"tunnel": tunnel.group(1), "hostname": host.group(1), "port": int(port.group(1))}


def cloudflared(*args):
    """Run a cloudflared command; its output, or stop with what it said."""
    done = subprocess.run([find_cloudflared(), *args], capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"cloudflared {' '.join(args)} failed:\n{(done.stderr or done.stdout).strip()}")
    return done.stdout


def setup(hostname, port, name=NAME):
    hostname = hostname.strip().lower()
    if not HOSTNAME.match(hostname):
        raise SystemExit(f"Not a host name: {hostname!r} (for example app.dramatizer.dev)")
    if not LOGIN.is_file():
        raise SystemExit("This PC is not linked to the Cloudflare account yet. Run this once and pick the domain:\n\n"
                         f"    \"{find_cloudflared()}\" tunnel login\n")
    tunnels = json.loads(cloudflared("tunnel", "list", "--name", name, "--output", "json") or "[]")
    if tunnels:
        tunnel_id = tunnels[0]["id"]
        print(f"The tunnel {name} exists ({tunnel_id}): kept.")
    else:
        tunnel_id = json.loads(cloudflared("tunnel", "create", "--output", "json", name))["id"]
        print(f"Made the tunnel {name} ({tunnel_id}).")
    credentials = HOME / f"{tunnel_id}.json"
    if not credentials.is_file():
        raise SystemExit(f"The tunnel's credentials are not on this PC ({credentials.name}). They are only written "
                         "when a tunnel is made: delete the tunnel in Cloudflare and run --setup again.")
    cloudflared("tunnel", "route", "dns", tunnel_id, hostname)
    print(f"{hostname} now points to the tunnel (Cloudflare DNS).")
    CONFIG.write_text(config_text(tunnel_id, credentials, hostname, port), encoding="utf-8")
    print(f"Wrote {CONFIG}: https://{hostname} -> https://127.0.0.1:{port}\n\n"
          "Start it with:  python main_backend.py tunnel")


def run_named(config):
    if not api_is_up(config["port"]):
        raise SystemExit(f"The API is not running on https://127.0.0.1:{config['port']} (the port in {CONFIG}). "
                         "Start it first, or run --setup again with --port.")
    command = [find_cloudflared(), "tunnel", "--no-autoupdate", "--config", str(CONFIG), "run"]
    print("Starting the Cloudflare tunnel (Ctrl+C stops it)...", flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    shown = False
    try:
        for line in process.stdout:
            if "Registered tunnel connection" in line and not shown:
                shown = True
                print(f"\n  The website is open at:  https://{config['hostname']}\n\n"
                      "  The same address every time. People log in with their own account.\n", flush=True)
            elif " ERR " in line:
                print("  cloudflared: " + line.strip(), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        process.terminate()
    return process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description="A public https:// address for the website and API (Cloudflare).")
    parser.add_argument("--port", type=int, default=None, help=f"the API's local port (default {PORT}; "
                        "the offline site uses 5105)")
    parser.add_argument("--setup", metavar="HOSTNAME", help="set up the fixed address once, e.g. app.dramatizer.dev")
    parser.add_argument("--quick", action="store_true", help="a random trycloudflare.com address, even when set up")
    args = parser.parse_args(argv)
    if args.setup:
        return setup(args.setup, args.port or PORT)
    config = None if args.quick else read_config()
    if config:
        if args.port and args.port != config["port"]:
            raise SystemExit(f"The fixed address goes to port {config['port']}. To change it: "
                             f"python main_backend.py tunnel --setup {config['hostname']} --port {args.port}")
        return run_named(config)
    args.port = args.port or PORT
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
                      "  It changes every time the tunnel starts (the fixed one is https://app.dramatizer.dev).\n",
                      flush=True)
            elif "ERR" in line or "error" in line.lower():
                print("  cloudflared: " + line.strip(), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        process.terminate()
    return process.wait()


if __name__ == "__main__":
    sys.exit(main())

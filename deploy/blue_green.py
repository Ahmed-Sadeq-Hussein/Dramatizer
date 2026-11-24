"""blue_green.py | Release and roll back with two slots behind the front door (Milestone 4, step 5).

    python deploy/blue_green.py release IMAGE --release deploy/release.json      # start, check, switch
    python deploy/blue_green.py rollback                                          # back to the previous version
    python deploy/blue_green.py status

One GPU cannot hold two fully loaded copies for long, so this is blue/green with a checked switch, not a gradual
rollout: the candidate starts in the idle slot beside the live one, deploy/smoke.py checks it, and only if every
check passes does the front door (deploy/front_door.py) point at it. The old container is then stopped, not
removed, so a rollback only has to start it again (a cold start: CosyVoice loads in about 70 s).

A failed check leaves the live version as it was: the candidate is stopped and the front door never moves.

Both slots share one state volume (the database, accounts, the TLS certificate), so clients see the same
certificate whichever slot is live. The job queue is held in each server's memory: a job still waiting in the old
version when the switch happens is lost, so switch when the queue is empty.

Every step, with its time, is written to reports/milestone4/05-release/. The mounts and ports are arguments, so the
same script runs a staging pair (its own volume, ports and front door) beside the live site.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "API calls"))
REPORTS = ROOT / "reports" / "milestone4" / "05-release"
READY_SECONDS = 300


class Log:
    """The steps of one release or rollback, each with its time."""

    def __init__(self, action, name):
        self.record = {"action": action, "release": name, "started": now(), "steps": [], "result": None}
        self.t0 = time.time()

    def step(self, what, **detail):
        entry = {"at": now(), "seconds": round(time.time() - self.t0, 1), "step": what, **detail}
        self.record["steps"].append(entry)
        print(f"{entry['seconds']:>7} s  {what}" + (f"  {detail}" if detail else ""), flush=True)

    def save(self, result):
        self.record.update(result=result, finished=now(), seconds=round(time.time() - self.t0, 1))
        REPORTS.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = REPORTS / f"{stamp}-{self.record['action']}-{self.record['release'] or 'unnamed'}.json"
        out.write_text(json.dumps(self.record, indent=2) + "\n", encoding="utf-8")
        shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
        print(f"{result.upper()}: written to {shown}", flush=True)
        return out


def now():
    return datetime.now().isoformat(timespec="seconds")


def docker(*args, check=True):
    done = subprocess.run(["docker", *args], capture_output=True, text=True)
    if check and done.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args[:2])}: {done.stderr.strip()}")
    return done.stdout.strip()


def exists(name):
    return bool(docker("ps", "-a", "--filter", f"name=^{name}$", "--format", "{{.Names}}", check=False))


def image_id(image):
    return docker("image", "inspect", "--format", "{{.Id}}", image)


class Slots:
    def __init__(self, args):
        self.args = args
        self.state_file = Path(args.state)
        self.ports = {"blue": args.blue_port, "green": args.green_port}

    def name(self, slot):
        return f"{self.args.prefix}-{slot}"

    def load(self):
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"live": None, "previous": None}

    def save(self, state):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.state_file)                     # the front door never reads half a file

    def run(self, slot, image, env_file):
        a = self.args
        cmd = ["run", "-d", "--init", "--name", self.name(slot), "--gpus", "all",   # --init: stop reaches the server
               "--env-file", str(env_file),
               "-p", f"127.0.0.1:{self.ports[slot]}:5005",
               "-v", f"{a.volume}:/app/history",
               "-v", f"{a.output}:/output",
               "-v", f"{a.models}:/models:ro",
               "-v", f"{a.voice_pack}:/voices:ro"]
        if a.secrets_env and Path(a.secrets_env).is_file():
            cmd[7:7] = ["--env-file", a.secrets_env]
        docker(*cmd, image)

    def wait_ready(self, port):
        from dramatizer_api import Api

        api = Api(f"https://127.0.0.1:{port}", self.args.fingerprint)
        deadline = time.time() + READY_SECONDS
        while time.time() < deadline:
            try:
                status, _, _ = api.request("GET", "ready")
                if status == 200:
                    return True
            except (OSError, ValueError, SystemExit):
                pass
            time.sleep(2)
        return False


def release(args):
    slots, log = Slots(args), Log("release", args.name)
    state = slots.load()
    live = state.get("live")
    slot = "green" if live and live["slot"] == "blue" else "blue"
    candidate = {"slot": slot, "port": slots.ports[slot], "container": slots.name(slot), "image": args.image,
                 "image_id": image_id(args.image), "release": args.name}
    log.step("candidate", **candidate, live_before=live and live["release"])
    if exists(candidate["container"]):
        docker("rm", "-f", candidate["container"])
        log.step("removed the slot's old container", container=candidate["container"])
    # The candidate keeps the settings it was checked with: a copy of its env file, by release name
    frozen = slots.state_file.parent / "releases" / f"{args.name}.env"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(Path(args.env).read_text(encoding="utf-8"), encoding="utf-8")
    slots.run(slot, args.image, frozen)
    log.step("started the candidate", container=candidate["container"], env=frozen.name)

    smoke = subprocess.run([sys.executable, str(ROOT / "deploy" / "smoke.py"), f"https://127.0.0.1:{candidate['port']}",
                            "--release", args.release, "--fingerprint", args.fingerprint,
                            "--password-file", args.password_file], capture_output=True, text=True)
    print(smoke.stdout, end="", flush=True)
    log.step("smoke test", exit=smoke.returncode, output=smoke.stdout.strip().splitlines()[-1:])
    if smoke.returncode != 0:
        docker("stop", candidate["container"])
        log.step("BLOCKED: stopped the candidate; the front door did not move", live=live and live["release"])
        log.save("blocked")
        return 1

    slots.save({"live": candidate, "previous": live, "switched": now()})
    log.step("switched the front door", live=candidate["release"], port=candidate["port"])
    if live and exists(live["container"]):
        docker("stop", live["container"])
        log.step("stopped the old version (kept for a rollback)", container=live["container"])
    log.save("released")
    return 0


def rollback(args):
    slots = Slots(args)
    state = slots.load()
    live, previous = state.get("live"), state.get("previous")
    log = Log("rollback", previous and previous["release"])
    if not previous or not exists(previous["container"]):
        log.step("nothing to roll back to", previous=previous)
        log.save("failed")
        return 1
    log.step("rolling back", from_release=live and live["release"], to_release=previous["release"])
    docker("start", previous["container"])
    log.step("started the previous version", container=previous["container"])
    if not slots.wait_ready(previous["port"]):
        docker("stop", previous["container"])
        log.step("the previous version did not get ready; the front door did not move")
        log.save("failed")
        return 1
    log.step("ready", port=previous["port"])
    slots.save({"live": previous, "previous": live, "switched": now()})
    log.step("switched the front door", live=previous["release"], port=previous["port"])
    if live and exists(live["container"]):
        docker("stop", live["container"])
        log.step("stopped the rolled-back version", container=live["container"])
    log.save("rolled back")
    return 0


def status(args):
    print(json.dumps(Slots(args).load(), indent=2))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Release and roll back with two slots behind the front door.")
    parser.add_argument("--prefix", default="dramatizer", help="container names: <prefix>-blue, <prefix>-green")
    parser.add_argument("--state", default=str(ROOT / "history" / "front_door.json"))
    parser.add_argument("--blue-port", type=int, default=5106)
    parser.add_argument("--green-port", type=int, default=5107)
    parser.add_argument("--volume", default="dramatizer-state")
    parser.add_argument("--output", default="D:/Dramatizer 2 renders")
    parser.add_argument("--models", default="D:/Dramatizer models")
    parser.add_argument("--voice-pack", default=os.environ.get("VOICE_PACK_HOST_DIR", ""))
    parser.add_argument("--secrets-env", default="D:/Dramatizer secrets/secrets.env")
    parser.add_argument("--fingerprint", default=os.environ.get("DRAMATIZER_FINGERPRINT", ""))
    parser.add_argument("--password-file", default="D:/Dramatizer secrets/smoke-test.txt")
    sub = parser.add_subparsers(dest="command", required=True)
    rel = sub.add_parser("release")
    rel.add_argument("image")
    rel.add_argument("--release", default=str(ROOT / "deploy" / "release.json"))
    rel.add_argument("--name", help="default: the release.json's name")
    rel.add_argument("--env", default=str(ROOT / "deploy" / "container.env"),
                     help="the settings to start it with (the same file release.py hashed)")
    sub.add_parser("rollback")
    sub.add_parser("status")
    args = parser.parse_args(argv)
    if args.command == "release":
        args.name = args.name or json.loads(Path(args.release).read_text(encoding="utf-8")).get("release")
        if not args.voice_pack or not args.fingerprint:
            raise SystemExit("Give --voice-pack (the host folder mounted at /voices) and --fingerprint.")
    return {"release": release, "rollback": rollback, "status": status}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

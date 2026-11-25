"""test_blue_green.py | Milestone 4, step 5: the front door, release, blocked release and rollback (Docker faked).

    python deploy/test_blue_green.py
"""

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deploy"))

import blue_green  # noqa: E402
import front_door  # noqa: E402
import release  # noqa: E402
import smoke  # noqa: E402
from backend import release_info  # noqa: E402


class FakeDocker:
    """Containers by name: running or stopped. Records every command."""

    def __init__(self):
        self.containers, self.calls = {}, []

    def __call__(self, *args, check=True):
        self.calls.append(args)
        if args[0] == "run":
            self.containers[args[args.index("--name") + 1]] = "running"
        elif args[0] == "stop":
            self.containers[args[1]] = "stopped"
        elif args[0] == "start":
            self.containers[args[1]] = "running"
        elif args[0] == "rm":
            self.containers.pop(args[2], None)
        elif args[:2] == ("image", "inspect"):
            return "sha256:" + "ab" * 32
        elif args[0] == "ps":
            name = args[args.index("--filter") + 1][len("name=^"):-1]
            return name if name in self.containers else ""
        return ""


class BlueGreenTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.docker = FakeDocker()
        self.smoke_exit = 0
        self.free_gpu = False
        patches = [mock.patch.object(blue_green, "docker", self.docker),
                   mock.patch.object(blue_green, "REPORTS", self.dir / "reports"),
                   mock.patch.object(blue_green.subprocess, "run", self.fake_run),
                   mock.patch.object(blue_green.Slots, "wait_ready", lambda s, port: True)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def fake_run(self, cmd, **kw):
        return subprocess.CompletedProcess(cmd, self.smoke_exit, stdout="ALLOWED\n" if not self.smoke_exit else
                                           "BLOCKED\n", stderr="")

    def args(self, name="v1"):
        return argparse.Namespace(prefix="t", state=str(self.dir / "state.json"), blue_port=5106, green_port=5107,
                                  volume="vol", output="out", models="models", voice_pack="pack", secrets_env="",
                                  team_voices="", free_gpu=self.free_gpu, network="net",
                                  fingerprint="f" * 64, password_file="pw", image=f"img:{name}",
                                  release="release.json", name=name,
                                  env=[str(ROOT / "deploy" / "container.env")])

    def state(self):
        return json.loads((self.dir / "state.json").read_text())

    def test_first_release_goes_live_in_blue(self):
        self.assertEqual(blue_green.release(self.args("v1")), 0)
        self.assertEqual(self.state()["live"]["slot"], "blue")
        self.assertEqual(front_door.live_port(self.dir / "state.json"), 5106)

    def test_second_release_switches_to_green_and_keeps_the_old_one_stopped(self):
        blue_green.release(self.args("v1"))
        self.assertEqual(blue_green.release(self.args("v2")), 0)
        state = self.state()
        self.assertEqual((state["live"]["release"], state["previous"]["release"]), ("v2", "v1"))
        self.assertEqual(self.docker.containers, {"t-blue": "stopped", "t-green": "running"})

    def test_a_failed_smoke_test_blocks_and_the_front_door_does_not_move(self):
        blue_green.release(self.args("v1"))
        self.smoke_exit = 1
        self.assertEqual(blue_green.release(self.args("bad")), 1)
        self.assertEqual(self.state()["live"]["release"], "v1")
        self.assertEqual(self.docker.containers, {"t-blue": "running", "t-green": "stopped"})
        reports = sorted((self.dir / "reports").glob("*blocked*")) or sorted((self.dir / "reports").glob("*bad*"))
        self.assertEqual(json.loads(reports[-1].read_text())["result"], "blocked")

    def test_rollback_starts_the_previous_version_and_switches_back(self):
        blue_green.release(self.args("v1"))
        blue_green.release(self.args("v2"))
        self.assertEqual(blue_green.rollback(self.args()), 0)
        self.assertEqual(self.state()["live"]["release"], "v1")
        self.assertEqual(self.docker.containers, {"t-blue": "running", "t-green": "stopped"})

    def test_the_container_runs_with_the_settings_release_py_hashed(self):
        # The live site's two env files: the frozen copy must hash like the release record, or /version differs
        files = [str(ROOT / "deploy" / "container.env"), str(ROOT / "deploy" / "live.env")]
        args = self.args("v1")
        args.env, args.team_voices = files, "D:/voice data/Voices"
        self.assertEqual(blue_green.release(args), 0)
        frozen = self.dir / "releases" / "v1.env"
        self.assertEqual(release_info.config_sha256(release_info.read_env_file(frozen)),
                         release.make("v1", files, commit="c")["config_sha256"])
        self.assertEqual(release_info.read_env_file(frozen)["VOICES_DIR"], "/team-voices")
        run = next(c for c in self.docker.calls if c[0] == "run")
        self.assertNotIn("-e", run)                                # no setting outside the hashed files

    def test_slots_join_the_network_and_restart_after_a_reboot(self):
        blue_green.release(self.args("v1"))
        run = next(c for c in self.docker.calls if c[0] == "run")
        self.assertEqual(run[run.index("--network") + 1], "net")
        self.assertEqual(run[run.index("--restart") + 1], "unless-stopped")
        self.assertIn(("network", "create", "net"), self.docker.calls)

    def test_rollback_with_nothing_before_fails(self):
        blue_green.release(self.args("v1"))
        self.assertEqual(blue_green.rollback(self.args()), 1)
        self.assertEqual(self.state()["live"]["release"], "v1")


    def test_free_gpu_stops_the_live_version_first_and_restarts_it_when_blocked(self):
        blue_green.release(self.args("v1"))
        self.free_gpu, self.smoke_exit = True, 1
        self.assertEqual(blue_green.release(self.args("bad")), 1)
        stops = [c[1] for c in self.docker.calls if c[0] == "stop"]
        self.assertEqual(stops, ["t-blue", "t-green"])            # live first, then the blocked candidate
        self.assertEqual(self.docker.containers["t-blue"], "running")
        self.assertEqual(self.state()["live"]["release"], "v1")

    def test_free_gpu_release_and_rollback(self):
        self.free_gpu = True
        blue_green.release(self.args("v1"))
        self.assertEqual(blue_green.release(self.args("v2")), 0)
        self.assertEqual(self.docker.containers, {"t-blue": "stopped", "t-green": "running"})
        self.assertEqual(blue_green.rollback(self.args()), 0)
        self.assertEqual(self.docker.containers, {"t-blue": "running", "t-green": "stopped"})


class FrontDoorTest(unittest.TestCase):
    def test_no_state_means_nothing_live(self):
        self.assertIsNone(front_door.live_port(Path(tempfile.mkdtemp()) / "missing.json"))

    def test_bytes_go_through_to_the_live_port(self):
        async def scenario():
            async def echo(reader, writer):
                writer.write(b"from backend: " + await reader.read(100))
                await writer.drain()
                writer.close()

            backend = await asyncio.start_server(echo, "127.0.0.1", 0)
            port = backend.sockets[0].getsockname()[1]
            state = Path(tempfile.mkdtemp()) / "state.json"
            state.write_text(json.dumps({"live": {"port": port}}))
            door = await asyncio.start_server(front_door.handler(state, "127.0.0.1"), "127.0.0.1", 0)
            reader, writer = await asyncio.open_connection("127.0.0.1", door.sockets[0].getsockname()[1])
            writer.write(b"hello")
            await writer.drain()
            reply = await asyncio.wait_for(reader.read(100), 5)
            writer.close()
            backend.close()
            door.close()
            return reply

        self.assertEqual(asyncio.run(scenario()), b"from backend: hello")

    def test_in_docker_it_forwards_to_the_live_container_by_name(self):
        state = Path(tempfile.mkdtemp()) / "state.json"
        state.write_text(json.dumps({"live": {"port": 5106, "container": "dramatizer-blue"}}))
        self.assertEqual(front_door.live_target(state, by_name=True), ("dramatizer-blue", 5005))
        self.assertEqual(front_door.live_target(state), ("127.0.0.1", 5106))
        state.write_text(json.dumps({"live": None}))
        self.assertIsNone(front_door.live_target(state, by_name=True))


class PasswordFileTest(unittest.TestCase):
    def test_bare_password_and_account_add_format(self):
        folder = Path(tempfile.mkdtemp())
        (folder / "a.txt").write_text("secret-1\n", encoding="utf-8")
        (folder / "b.txt").write_text("other\tx\nsmoke-test\tsecret-2\n", encoding="utf-8")
        self.assertEqual(smoke.read_password(folder / "a.txt"), "secret-1")
        self.assertEqual(smoke.read_password(folder / "b.txt"), "secret-2")


if __name__ == "__main__":
    unittest.main()

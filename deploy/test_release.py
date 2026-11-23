"""test_release.py | Milestone 4, steps 3 and 4: the release record, /version, /ready and the smoke checks.

    python deploy/test_release.py
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy"))

import release  # noqa: E402
import smoke  # noqa: E402

from backend import release_info  # noqa: E402


class ReleaseInfo(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_the_settings_hash_ignores_secrets_and_order(self):
        a = {"DEVICE": "cuda", "OUTPUT_DIR": "/output"}
        b = {"OUTPUT_DIR": "/output", "DEVICE": "cuda", "UI_PASSWORD": "never hashed", "HF_TOKEN": "nor this"}
        self.assertEqual(release_info.config_sha256(a), release_info.config_sha256(b))
        self.assertNotEqual(release_info.config_sha256(a), release_info.config_sha256({**a, "MAX_QUEUE": "20"}))
        self.assertEqual(release_info.config_sha256({"DEVICE": " cuda "}), release_info.config_sha256({"DEVICE": "cuda"}))
        self.assertFalse([k for k in release_info.CONFIG_KEYS if "PASSWORD" in k or "TOKEN" in k or "KEY" in k])

    def test_release_json_is_made_from_the_env_file_and_the_host_folders(self):
        env = self.dir / "container.env"
        env.write_text("# a comment\nDEVICE=cuda\nVOICE_PACK_DIR=/voices\nMODELS_DIR=/models\n", encoding="utf-8")
        pack, store = self.dir / "pack", self.dir / "models"
        pack.mkdir(), store.mkdir()
        (pack / "pack.json").write_text(json.dumps({"revision": "abc123", "manifest_sha256": "m1", "clips": []}))
        (store / "models.json").write_text(json.dumps({"models": {"cosyvoice": {"repo": "FunAudioLLM/x",
                                                                                "revision": "r1", "files": {}}}}))
        made = release.make("v8", env, image="img@sha256:1", base="base@sha256:2", voice_pack_dir=pack,
                            models_dir=store, commit="c0ffee")
        self.assertEqual(made["voice_pack"], {"revision": "abc123", "manifest_sha256": "m1"})
        self.assertEqual(made["models"], {"cosyvoice": {"repo": "FunAudioLLM/x", "revision": "r1"}})
        self.assertEqual(made["config_sha256"], release_info.config_sha256(
            {"DEVICE": "cuda", "VOICE_PACK_DIR": "/voices", "MODELS_DIR": "/models"}))
        self.assertNotIn(str(self.dir), json.dumps(made))           # host paths are read, never written

    def test_images_are_named_by_digest(self):
        with self.assertRaises(SystemExit):
            release.main(["v8", "--image", "ghcr.io/x/dramatizer:latest", "--out", str(self.dir / "r.json")])

    def test_differences(self):
        want = {"commit": "a", "voice_pack": {"revision": "1"}, "models": None, "config_sha256": "h"}
        self.assertEqual(release_info.differences(want, {**want, "loaded": ["voice"]}), [])
        self.assertEqual(release_info.differences(want, {**want, "voice_pack": {"revision": "2"}}),
                         [("voice_pack", {"revision": "1"}, {"revision": "2"})])
        self.assertEqual([f for f, *_ in release_info.differences(want, {})], list(release_info.COMPARED))


class Routes(unittest.TestCase):
    def setUp(self):
        from backend.api import server

        self.server = server
        self.client = server.app.test_client()

    def test_version_needs_no_key_and_shows_no_path(self):
        with mock.patch.dict("os.environ", {"OUTPUT_DIR": "D:/secret place", "GIT_COMMIT": "c0ffee"}):
            got = self.client.get("/api/v1/version").get_json()
        self.assertEqual(got["commit"], "c0ffee")
        self.assertEqual(set(release_info.COMPARED) - set(got), set())
        self.assertNotIn("secret place", json.dumps(got))

    def test_ready_is_503_until_the_voice_model_and_voices_are_there(self):
        loaded = set()
        with mock.patch.object(self.server.model, "is_loaded", lambda key: key in loaded), \
                mock.patch.dict(self.server._library, {"clips": None}):
            reply = self.client.get("/api/v1/ready")
            self.assertEqual(reply.status_code, 503)
            self.assertEqual(reply.get_json()["missing"], ["voice_model", "voices"])
            loaded.add("voice")
            self.server._library["clips"] = {"VCTK": {"clips": ["one"], "by_id": {}}}
            reply = self.client.get("/api/v1/ready")
            self.assertEqual((reply.status_code, reply.get_json()["checks"]["voices"]), (200, "1 read"))

    def test_a_failed_preload_says_why(self):
        def broken():
            raise FileNotFoundError("CosyVoice weights not found")

        with mock.patch.object(self.server.model, "voice_model", broken), \
                mock.patch.dict(self.server._preload, {"state": "off", "problem": None}):
            self.server.preload_voice_model()
            got = self.client.get("/api/v1/ready").get_json()
        self.assertIn("weights not found", got["checks"]["voice_model"])


class FakeApi:
    """The candidate as smoke.py sees it."""

    def __init__(self, version, job_status="done", answer="There are 176 voices."):
        self.version, self.job_status, self.answer = version, job_status, answer
        self.token, self.timeout, self.asked = None, 60, []

    def request(self, method, path, body=None):
        self.asked.append(path)
        if path == "ready":
            return 200, "application/json", b'{"ready": true, "checks": {}}'
        if path.endswith("/audio"):
            return 200, "audio/wav", b"RIFF"
        raise AssertionError(path)

    def json(self, method, path, body=None):
        self.asked.append(path)
        if path == "version":
            return self.version
        if path == "login":
            return {"token": "t"}
        if path == "speak":
            return {"job": {"id": "j", "status": "queued"}}
        if path == "jobs/j":
            return {"job": {"id": "j", "status": self.job_status, "seconds": 9.5, "error": None}}
        if path == "assistant/ask":
            return {"answer": self.answer, "source": "records"}
        raise AssertionError(path)


class Smoke(unittest.TestCase):
    RELEASE = {"release": "v8", "commit": "c0ffee", "voice_pack": {"revision": "abc123", "manifest_sha256": "m"},
               "models": {"cosyvoice": {"repo": "r", "revision": "1"}}, "config_sha256": "h"}

    def setUp(self):
        self.password = Path(tempfile.mkdtemp()) / "smoke-test.txt"
        self.password.write_text("not a real password\n", encoding="utf-8")
        patch = mock.patch("soundfile.info", lambda path: mock.Mock(duration=4.2))
        patch.start()
        self.addCleanup(patch.stop)

    def check(self, api, release=None, wer=0.0):
        run = smoke.Smoke(api, release or self.RELEASE, self.password, poll=0, wer=lambda path, text: wer)
        return run.all(), {r["check"]: r["result"] for r in run.results}, run

    def test_a_good_candidate_is_allowed(self):
        allowed, results, _ = self.check(FakeApi({**self.RELEASE, "loaded": ["voice"]}))
        self.assertTrue(allowed)
        self.assertEqual(set(results.values()), {"pass"})

    def test_the_wrong_voice_pack_blocks_the_release(self):
        """The required evidence: release.json names another pack revision than the one the candidate loaded."""
        wrong = {**self.RELEASE, "voice_pack": {"revision": "not-this-one", "manifest_sha256": "m"}}
        allowed, results, run = self.check(FakeApi(self.RELEASE), release=wrong)
        self.assertFalse(allowed)
        self.assertEqual(results["version"], "FAIL")
        detail = next(r["detail"] for r in run.results if r["check"] == "version")
        self.assertEqual(list(detail), ["voice_pack"])

    def test_a_bad_render_blocks(self):
        self.assertEqual(self.check(FakeApi(self.RELEASE), wer=0.4)[1]["render"], "FAIL")
        self.assertEqual(self.check(FakeApi(self.RELEASE, job_status="failed"))[1]["render"], "FAIL")

    def test_an_answer_without_a_number_blocks(self):
        self.assertEqual(self.check(FakeApi(self.RELEASE, answer="I do not know."))[1]["chatbot"], "FAIL")

    def test_no_password_file_fails_login_and_skips_what_needs_it(self):
        self.password.unlink()
        allowed, results, _ = self.check(FakeApi(self.RELEASE))
        self.assertFalse(allowed)
        self.assertEqual((results["login"], results["render"], results["chatbot"]), ("FAIL", "skipped", "skipped"))

    def test_the_password_is_never_in_the_results(self):
        _, _, run = self.check(FakeApi(self.RELEASE))
        self.assertNotIn("not a real password", json.dumps(run.results))


if __name__ == "__main__":
    unittest.main()

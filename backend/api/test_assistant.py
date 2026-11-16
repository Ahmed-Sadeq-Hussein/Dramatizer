"""Tests of assistant.py: the worker's start, its replies and its failure, with a tiny fake worker (no model).

    python -m unittest backend.api.test_assistant
"""
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))    # run as a file by main_backend.py test

from backend.api import assistant  # noqa: E402

WORKER = """
import json, sys
print("loading the model...", flush=True)
print("@@reply " + json.dumps({"ready": True, "model": "fake-gemma"}), flush=True)
for line in sys.stdin:
    q = json.loads(line)
    if q["question"] == "die":
        sys.exit(3)
    if q["question"] == "slow":
        continue
    print("@@reply " + json.dumps({"id": q["id"], "answer": q["question"].upper() + " as " + q["token"],
                                    "tools": [], "seconds": 0.1}), flush=True)
"""


class Worker(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        script = Path(folder.name) / "worker.py"
        script.write_text(WORKER, encoding="utf-8")
        self.a = assistant.Assistant()
        for p in (mock.patch.object(self.a, "command", lambda: [sys.executable, str(script)]),
                  mock.patch.object(assistant, "LOG_PATH", Path(folder.name) / "assistant.log")):
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.a.stop)

    def wait_for(self, state):
        for _ in range(1200):                                        # up to 60 s: a busy PC (a model loading) is slow
            if self.a.status()["state"] == state:
                return
            time.sleep(0.05)
        self.fail(f"never {state}: {self.a.status()}")

    def test_starts_answers_with_the_token_and_reports_a_crash(self):
        self.assertEqual(self.a.status()["state"], "idle")
        with self.assertRaises(RuntimeError):
            self.a.ask("hi", "t")                                    # not started yet
        self.a.start()
        self.wait_for("ready")
        self.assertEqual(self.a.status()["model"], "fake-gemma")
        self.assertEqual(self.a.ask("hello", "tok-1")["answer"], "HELLO as tok-1")
        with self.assertRaises(RuntimeError):
            self.a.ask("slow", "t", timeout=0.3)                     # too slow: the caller falls back
        with self.assertRaises(RuntimeError):
            self.a.ask("die", "t", timeout=60)
        self.wait_for("failed")
        self.a.start()                                               # a failed worker can be started again
        self.wait_for("ready")

    def test_off_without_settings(self):
        b = assistant.Assistant()
        with mock.patch.object(assistant.Assistant, "_optional", staticmethod(lambda name: "")):
            b.configure("https://127.0.0.1:5005", "ab" * 32)
            self.assertEqual(b.status()["state"], "off")
            b.start()
            self.assertEqual(b.status()["state"], "off")


if __name__ == "__main__":
    unittest.main()

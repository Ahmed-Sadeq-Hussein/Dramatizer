"""Tests of the Report evaluation's design and blindness (report_eval.py), without rendering anything.

    python backend/genome/test_report_eval.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))    # run as a file by main_backend.py test

from backend.genome import report_eval as RE  # noqa: E402

VOICES = [f"{d}/{i}" for d in RE.DATASETS for i in range(4)]


class Design(unittest.TestCase):
    def setUp(self):
        self.d = RE.design(list(VOICES))

    def test_every_listener_hears_every_voice_and_sentence_evenly(self):
        b = RE.balance(self.d)
        self.assertEqual(b["items"], 48)
        for name, mine in b["per_listener"].items():
            with self.subTest(name):
                self.assertEqual((mine["items"], mine["clips"], mine["voices"]), (24, 72, 12))
                self.assertEqual(set(mine["per_sentence"].values()), {6})
        self.assertEqual(set(b["items_shared"].values()), {8})                     # every two listeners share 8
        self.assertEqual(set(b["condition_first"].values()), {16})                 # no condition is always first

    def test_each_item_has_two_listeners_all_three_conditions_and_a_gene_count(self):
        for item in self.d["items"]:
            self.assertEqual(len(set(item["listeners"])), 2)
            self.assertEqual(sorted(item["order"]), sorted(RE.CONDITIONS))
            self.assertIn(item["genes"], RE.GENE_COUNTS)
        self.assertEqual(len({i["item"] for i in self.d["items"]}), 48)

    def test_the_design_is_the_same_every_time(self):
        self.assertEqual(RE.design(list(VOICES)), self.d)
        self.assertEqual(RE.stable_seed(1, "a"), RE.stable_seed(1, "a"))

    def test_voices_are_drawn_from_both_pitch_halves_of_each_dataset(self):
        pitch = {v: float(i) for v in VOICES + [f"{d}/x{i}" for d in RE.DATASETS for i in range(4)]
                 for i in [int(v.rsplit("/", 1)[1].lstrip("x"))]}
        chosen = RE.choose_voices(list(pitch), pitch)
        self.assertEqual(len(chosen), 12)
        for d in RE.DATASETS:
            mine = [v for v in chosen if v.startswith(d + "/")]
            middle = sorted(pitch[v] for v in pitch if v.startswith(d + "/"))
            median = (middle[3] + middle[4]) / 2
            self.assertEqual(sum(pitch[v] <= median for v in mine), 2)
        self.assertEqual(RE.choose_voices(list(pitch), pitch), chosen)


class Blind(unittest.TestCase):
    def test_published_runs_show_no_condition_and_the_key_holds_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            d = RE.design(list(VOICES))
            item = d["items"][0]
            plan = {"batch": "rx", "seed": 1, "render_seed": 0, "sentences": RE.SENTENCES, "items": [item],
                    "listeners": RE.LISTENERS, "piles": d["piles"],
                    "voices": [{"voice": item["voice"], "origin": {}, "reference_audio": "r.wav", "reference_text": "r"}]}
            item_dir = root / "out" / "rx" / "items" / item["item"]
            item_dir.mkdir(parents=True)
            renders = {}
            for c in RE.CONDITIONS:
                (item_dir / f"{c}.wav").write_bytes(c.encode())
                renders[c] = {"audio": f"{c}.wav", "genes": [] if c == "plain" else [{"feature": "sigh"}],
                              "markup": "with [sigh]" if c != "plain" else "plain"}
            (item_dir / "state.json").write_text(json.dumps({**item, "renders": renders}), encoding="utf-8")
            with mock.patch.object(RE, "OUT_DIR", root / "out"):
                published = RE.publish(plan, runs_dir=root / "runs")
            run_dir = root / "runs" / published[0]
            run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            text = json.dumps(run)
            self.assertNotIn("sigh", text)
            self.assertNotIn("plain", text.replace('"plain"', ""))                 # no condition name anywhere
            self.assertIsNone(run["baseline"]["audio"])
            self.assertEqual([g["genes"] for g in run["genomes"]], [[], [], []])
            self.assertTrue(run["blind"] and run["sealed"])
            self.assertEqual(run["only"], sorted(RE.LISTENERS[n] for n in item["listeners"]))
            key = json.loads((root / "out" / "rx" / "key.json").read_text(encoding="utf-8"))["runs"][published[0]]
            for clip, condition in key["versions"].items():
                self.assertEqual((run_dir / f"{clip}.wav").read_bytes(), condition.encode())

    def test_only_the_piles_two_listeners_may_open_it_admins_included(self):
        from flask import g

        from backend.api import server

        run = {"only": ["ash", "min"], "batch": "rx", "pile": 4}
        cases = {"ash": True, "min": True, "rama": False}
        for user, allowed in cases.items():
            with server.app.test_request_context("/"), self.subTest(user):
                g.account = SimpleNamespace(username=user, can=lambda _: True, may_rate_pile=lambda *_: True)
                self.assertEqual(server.may_see(run), allowed)
        with server.app.test_request_context("/"):
            g.account = None                                                     # an API key
            self.assertFalse(server.may_see(run))


if __name__ == "__main__":
    unittest.main()

"""Tests of the results bundle's safety: what may never be published (publish_results.py), without building it.

    python backend/test_publish_results.py
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # run as a file by main_backend.py test

from backend import publish_results as P  # noqa: E402

HEADER = "rater,account,batch,pile,voice,run,clip,genome,score\n"


class Check(unittest.TestCase):
    def check(self, files, sealed=()):
        with tempfile.TemporaryDirectory() as folder:
            for name, text in files.items():
                path = Path(folder) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(text) if isinstance(text, bytes) else path.write_text(text, encoding="utf-8")
            return [what for _, what in P.check(folder, set(sealed))]

    def test_a_clean_bundle_passes(self):
        self.assertEqual(self.check({"scores.csv": HEADER + "Ash,,mos-diverse-v1,1,VCTK/p250,r1,genome_01,1,4\n",
                                     "charts/q1.jpg": b"\xff\xd8"}), [])

    def test_everything_that_must_stay_on_this_pc_is_caught(self):
        cases = {"the sealed key file": "see key.json", "the sealed Report evaluation": "batch report-evaluation",
                 "a test account": "test_user_3 scored", "an email address": "someone@example.com",
                 "a credential word": "Bearer abc", "a local path": r"C:\Users\someone\x.json",
                 "the sealed run run-42": "run-42"}
        for what, text in cases.items():
            with self.subTest(what):
                self.assertIn(what, self.check({"x.json": text}, sealed=["run-42"]))
        self.assertIn("an audio or archive file", self.check({"a.wav": b"RIFF"}))


class Scores(unittest.TestCase):
    def test_sealed_runs_test_accounts_and_login_names_are_left_out(self):
        with tempfile.TemporaryDirectory() as folder:
            who = Path(folder) / "who_rated_what.csv"
            who.write_text(HEADER + "Ash,ash,b,1,V/1,r1,genome_01,1,4\n"
                                    "Test,test_user_1,b,1,V/1,r1,genome_01,1,2\n"
                                    "Min,min,report-evaluation,1,V/2,sealed1,genome_01,1,5\n", encoding="utf-8-sig")
            with mock.patch("backend.genome.ratings.who_path", lambda: who):
                text, kept = P.scores_csv({"sealed1"})
        self.assertEqual(kept, 1)
        self.assertEqual(text.splitlines()[0], "rater,batch,pile,voice,run,clip,genome,score")
        self.assertNotIn("sealed1", text)
        self.assertNotIn("test_user", text)


class RatingData(unittest.TestCase):
    def test_runs_keep_every_genome_detail_but_no_audio_paths_or_logins(self):
        run = {"id": "r1", "voice": "VCTK/p250", "batch": "b", "pile": 1, "method": "random", "seed": 7,
               "reference_audio": r"C:\Users\x\ref.wav", "only": ["min"],
               "baseline": {"audio": "baseline.wav", "heard": "Well", "seconds": 2.0},
               "genomes": [{"number": 1, "audio": "genome_01.wav", "genes": [{"feature": "sigh"}], "speed": 1.1,
                            "markup": "Well [sigh]", "verdict": "fine", "sanity": {"wer": 0.0, "similarity": 0.8}}]}
        sealed = {"id": "s1", "sealed": True, "baseline": {}, "genomes": []}
        rated = {"r1": {"raters": '["Ash", "Test user 3"]', "picked": "[]", "updated": "2030-01-01"}}
        out = P.runs_data([run, sealed], rated, {"s1"})
        self.assertEqual(list(out), ["r1"])
        r1 = out["r1"]
        self.assertNotIn("reference_audio", r1)
        self.assertNotIn("only", r1)
        self.assertEqual(r1["baseline"], {"heard": "Well", "seconds": 2.0})
        self.assertEqual(r1["genomes"][0], {"number": 1, "genes": [{"feature": "sigh"}], "speed": 1.1,
                                            "markup": "Well [sigh]", "verdict": "fine",
                                            "sanity": {"wer": 0.0, "similarity": 0.8}})
        self.assertEqual(r1["ratings"]["raters"], ["Ash"])

    def test_history_has_every_change_without_logins_tests_or_sealed_runs(self):
        entries = [{"time": "t1", "run": "r1", "rater": "Ash", "account": "ash", "submitted_by": "Ash", "clip": "genome_01",
                    "score": 3, "previous": None, "genes": [{"feature": "sigh"}]},
                   {"time": "t2", "run": "r1", "rater": "Ash", "account": "ash", "submitted_by": "Ash", "clip": "genome_01",
                    "score": 4, "previous": 3},
                   {"time": "t3", "run": "r1", "rater": "Test user 3", "account": "test_user_3", "clip": "genome_01",
                    "score": 1},
                   {"time": "t4", "run": "s1", "rater": "Min", "account": "min", "clip": "genome_01", "score": 5}]
        text, kept = P.history_csv(entries, {"s1"})
        self.assertEqual(kept, 2)
        self.assertNotIn("account", text.splitlines()[0])
        self.assertNotIn("ash,", text)
        self.assertNotIn("Test user", text)
        self.assertNotIn("s1", text)

    def test_piles_by_display_name_without_sealed_piles_or_test_accounts(self):
        accounts = [{"username": "ash", "display_name": "Ash", "role": "admin"},
                    {"username": "min", "display_name": "Min", "role": "listener"},
                    {"username": "test_user_1", "display_name": "Test user 1", "role": "listener"}]
        assignments = [{"username": "ash", "pile": "open-v1|1", "position": 0},
                       {"username": "ash", "pile": "sealed-v1|4", "position": 1},
                       {"username": "min", "pile": "sealed-v1|2", "position": 0},
                       {"username": "test_user_1", "pile": "open-v1|2", "position": 0}]
        out = P.piles_data(assignments, accounts, {"sealed-v1"})
        self.assertEqual(out, {"listeners": [{"name": "Ash", "role": "admin", "piles": [{"batch": "open-v1", "pile": 1}]},
                                             {"name": "Min", "role": "listener", "piles": []}]})


class HuggingFace(unittest.TestCase):
    def test_never_uploads_to_a_public_dataset(self):
        from types import SimpleNamespace

        api = mock.MagicMock()
        api.repo_info.return_value = SimpleNamespace(private=False)
        manifest = {"snapshot_id": "a", "ratings_up_to": "b", "data_revision": "c"}
        with mock.patch("huggingface_hub.HfApi", return_value=api):
            with self.assertRaisesRegex(SystemExit, "PUBLIC"):
                P.hf_upload("folder", "team/results", manifest)
            api.upload_folder.assert_not_called()
            api.repo_info.return_value = SimpleNamespace(private=True)
            P.hf_upload("folder", "team/results", manifest)
        kwargs = api.upload_folder.call_args.kwargs
        self.assertEqual((kwargs["repo_type"], kwargs["delete_patterns"]), ("dataset", P.BUNDLE_PATTERNS))


if __name__ == "__main__":
    unittest.main()

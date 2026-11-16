"""Tests of the team's parts merged on 12 Nov 2026, as they now fit the rest (no GPU, no real ratings):
Baesar's preference guidance (preference.py, search.draw_guided) and MOS import (import_mos.py), and
Rama's genome report (report.py), with the rules added since: test accounts are not evidence, and
disabled genes are never proposed again.

    python -m unittest backend.genome.test_guidance_and_report
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))                                    # run as a file by main_backend.py test
sys.path.insert(0, str(ROOT / "backend" / "evaluation"))         # import_mos.py imports checks.py by name

from backend.genome import preference, ratings, report, search  # noqa: E402
from backend.genome.descriptors import SIZE  # noqa: E402

LINE = "Well, you really did it this time, and I honestly didn't think you had it in you, so I'll be waiting."


def scores():
    return {"schema_version": 1, "run": "r1", "scale": [0, 5], "raters": ["Ash", "Test user 3"],
            "scores": {"Ash": {"baseline": 3.0, "genome_01": 4.0, "genome_02": 1.0},
                       "Test user 3": {"baseline": 5.0, "genome_01": 0.0, "genome_02": 5.0}},
            "artifacts": {"Test user 3": {"genome_01": True}}, "picked": [], "updated": None}


class TestAccounts(unittest.TestCase):
    def test_test_accounts_do_not_count_in_the_mos(self):
        self.assertTrue(ratings.is_test_rater("Test user 3") and ratings.is_test_rater("test_user_12"))
        self.assertFalse(ratings.is_test_rater("Tester") or ratings.is_test_rater("Ash"))
        m = ratings.mos(scores(), "genome_01")
        self.assertEqual((m["mos"], m["n"], list(m["scores"])), (4.0, 1, ["Ash"]))
        only = ratings.evidence(scores())
        self.assertEqual((only["raters"], list(only["scores"]), only["artifacts"]), (["Ash"], ["Ash"], {}))
        self.assertIn("Test user 3", scores()["scores"])                      # the file itself keeps them


class Report(unittest.TestCase):
    def test_rated_genomes_mark_what_can_no_longer_be_rendered(self):
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder) / "r1"
            run_dir.mkdir()
            (run_dir / ratings.RATINGS_FILE).write_text(json.dumps(scores()), encoding="utf-8")
            run = {"id": "r1", "voice": "VCTK/p250", "genomes": [
                {"number": 1, "genes": [{"feature": "sigh", "position": 0.5}], "verdict": "fine", "markup": "a"},
                {"number": 2, "genes": [{"feature": "hissing", "position": 0.5}], "verdict": "fine", "markup": "b"}]}
            with mock.patch.object(report.S, "list_runs", lambda: [(run_dir, run)]):
                rows = {r["genome"]: r for r in report.rated_genomes()}
        self.assertEqual((rows[1]["mos"], rows[1]["above_baseline"], rows[1]["unusable"]), (4.0, 1.0, ""))
        self.assertIn("hissing", rows[2]["unusable"])
        self.assertIn("can no longer be rendered", report.describe(rows[2]))
        self.assertEqual(rows[2]["baseline_mos"], 3.0)                         # Ash's plain line only


class Guidance(unittest.TestCase):
    def sessions(self):
        rng = np.random.default_rng(0)
        out = []
        for v in range(6):
            choices = [[{"feature": "sigh", "position": 0.5}], [{"feature": "cough", "position": 0.2}],
                       [{"feature": "mn", "position": 0.8}]]
            out.append(preference.Session(f"VCTK/p{v}", f"voice{v}.wav", [], choices, (0,)))
        self.vectors = {f"voice{v}.wav": rng.normal(size=preference.EMBEDDING_SIZE) for v in range(6)}
        return out

    def test_the_model_trains_saves_and_gives_a_direction(self):
        sessions = self.sessions()
        artifact = preference.train(sessions, validation_fraction=0.34, embedding=self.vectors.__getitem__)
        self.assertEqual(len(artifact["shared"]), SIZE)
        self.assertEqual(artifact["training"]["choices"] + artifact["validation"]["shared"]["choices"], 6)
        self.assertTrue(set(artifact["training"]["train_speakers"]).isdisjoint(artifact["training"]["validation_speakers"]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            loaded = preference.load_model(path)
        self.assertEqual(len(loaded["sha256"]), 64)
        self.assertEqual(preference.direction(loaded, [], conditioned=False).shape, (SIZE,))
        from backend.genome.descriptors import genome_vector
        w = np.asarray(loaded["shared"])
        chosen, other = (np.dot(w, genome_vector(sessions[0].choices[i])) for i in (0, 1))
        self.assertGreater(chosen, other)                                       # the chosen sigh ranks above the cough

    def test_guided_search_never_proposes_a_disabled_gene(self):
        start = [{"feature": "hissing", "position": 0.4}, {"feature": "sigh", "position": 0.8}]
        with mock.patch.object(preference, "load_model", lambda: {"sha256": "x"}), \
                mock.patch.object(preference, "direction", lambda *a, **k: np.zeros(SIZE)):
            found, sha = search.draw_guided(LINE, 2, 6, 0, start, None, conditioned=False)
        self.assertEqual(sha, "x")
        for c in found:
            self.assertFalse({"hissing", "vocalized_noise"} & {g["feature"] for g in c.operations}, c.operations)


class FromTheGenomeBranch(unittest.TestCase):
    """Baesar's genome distance and diverse shortlist, on the current schema."""

    def test_distance_is_symmetric_and_zero_for_the_same_genome(self):
        from backend.genome.descriptors import genome_distance
        a = [{"feature": "breath", "position": 0.0}, {"feature": "speed", "rate": 1.1}]
        b = [{"feature": "sigh", "position": 1.0}]
        self.assertEqual(genome_distance(a, a), 0)
        self.assertAlmostEqual(genome_distance(a, b), genome_distance(b, a))
        self.assertGreater(genome_distance(a, b), 0)

    def test_shortlist_takes_fine_genomes_that_differ(self):
        same = [{"feature": "breath", "position": 0.0}]
        far = [{"feature": "sigh", "position": 1.0}, {"feature": "laughter", "position": 0.5}]
        run = {"genomes": [
            {"number": 1, "verdict": "fine", "genes": same},
            {"number": 2, "verdict": "fine", "genes": list(same)},
            {"number": 3, "verdict": "terrible", "genes": [{"feature": "cough", "position": 0.2}]},
            {"number": 4, "verdict": "fine", "genes": far},
        ]}
        self.assertEqual(search.diverse_shortlist(run, 2), [4, 1])      # the copy (2) and the terrible one (3) wait
        self.assertEqual(sorted(search.diverse_shortlist(run, 9)), [1, 2, 4])
        self.assertEqual(search.diverse_shortlist({"genomes": []}), [])


class ImportMos(unittest.TestCase):
    def test_test_accounts_are_left_out_of_the_mlflow_import(self):
        import import_mos
        with tempfile.TemporaryDirectory() as folder:
            run_dir = Path(folder) / "runs" / "r1"
            run_dir.mkdir(parents=True)
            run = {"id": "r1", "text": LINE, "method": "random", "render_seed": 0,
                   "origin": {"dataset": "VCTK", "speaker": "p250", "clip_id": "p250_001"},
                   "baseline": {"audio": "baseline.wav"},
                   "genomes": [{"number": 1, "audio": "genome_01.wav"}, {"number": 2, "audio": "genome_02.wav"}]}
            (run_dir / "run.json").write_text(json.dumps(run), encoding="utf-8")
            (run_dir / "ratings.json").write_text(json.dumps(scores()), encoding="utf-8")
            out = import_mos.summarize(Path(folder))
        self.assertEqual(out["metrics"]["listeners"], 1)
        self.assertEqual(out["metrics"]["test_account_runs_left_out"], 1)
        self.assertEqual(out["metrics"]["mean_paired_delta"], -0.5)            # Ash: mean(4, 1) - 3


if __name__ == "__main__":
    unittest.main()

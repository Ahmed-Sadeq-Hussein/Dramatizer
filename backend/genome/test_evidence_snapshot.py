"""Tests of the evidence snapshot (evidence_snapshot.py) and the audio artifact check (artifact_check.py),
from the feedback of 16 Nov 2026: reconcile the report with the runs, say what an answer rests on, name no
winner on weak evidence, and check for artifacts before recommending. No models, no real ratings.

    python backend/genome/test_evidence_snapshot.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))    # run as a file by main_backend.py test

from backend import storage  # noqa: E402
from backend.genome import artifact_check as A  # noqa: E402
from backend.genome import evidence_snapshot as E  # noqa: E402
from backend.genome import ratings as R  # noqa: E402
from backend.genome import search as S  # noqa: E402

RATE = 16000


def speech(seconds=2.0, level=0.3, gap=None):
    """A stand-in for speech: a tone; gap=(start, length) in seconds is silenced."""
    t = np.arange(int(seconds * RATE)) / RATE
    x = level * np.sin(2 * np.pi * 220 * t)
    if gap:
        x[int(gap[0] * RATE): int((gap[0] + gap[1]) * RATE)] = 0
    return x


class ArtifactCheck(unittest.TestCase):
    def test_clean_audio_passes(self):
        self.assertEqual(A.check_samples(speech(), RATE)["failed"], [])

    def test_each_fault_is_named(self):
        clipped = np.clip(speech(level=2.0), -1, 1)
        cases = {"clipping": (clipped, None), "gap": (speech(6, gap=(1, 3)), None),
                 "too_short": (speech(0.2), None), "unreadable": (np.array([0.1, np.nan]), None),
                 "burst": (speech(level=0.9), A.check_samples(speech(level=0.3), RATE)["peak_db"])}
        for fault, (x, plain) in cases.items():
            with self.subTest(fault):
                self.assertIn(fault, A.check_samples(x, RATE, plain)["failed"])

    def test_a_sound_as_loud_as_the_plain_line_is_no_burst(self):
        plain = A.check_samples(speech(level=0.3), RATE)["peak_db"]
        self.assertEqual(A.check_samples(speech(level=0.4), RATE, plain)["failed"], [])


class Runs(unittest.TestCase):
    """Fake runs with audio and ratings in a temporary folder and database."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        self.dir = Path(folder.name)
        patch = mock.patch.object(storage, "DB_PATH", self.dir / "test.db")
        patch.start()
        self.addCleanup(patch.stop)

    def run_folder(self, run_id, voice, scores, loud=(), artifacts=None, genomes=3, genes=None):
        folder = self.dir / run_id
        folder.mkdir()
        sf.write(folder / "baseline.wav", speech(), RATE)
        run = {"id": run_id, "voice": voice, "batch": "b", "baseline": {"audio": "baseline.wav"}, "text": "a line", "genomes": []}
        for n in range(1, genomes + 1):
            sf.write(folder / f"genome_{n:02d}.wav", speech(level=0.95 if n in loud else 0.3), RATE)
            run["genomes"].append({"number": n, "verdict": "fine", "markup": f"m{n}", "speed": 1.0, "audio": f"genome_{n:02d}.wav",
                                   "genes": (genes or {}).get(n, [{"feature": "sigh", "position": 0.5}])})
        doc = {"schema_version": 1, "run": run_id, "scale": [0, 5], "raters": sorted(scores), "scores": scores,
               "artifacts": artifacts or {}, "picked": [], "updated": None}
        (folder / R.RATINGS_FILE).write_text(json.dumps(doc), encoding="utf-8")
        return folder, run


class Snapshot(Runs):
    def test_coverage_lists_the_genomes_without_a_score(self):
        runs = [self.run_folder("r1", "VCTK/p1", {"Ash": {"baseline": 3, "genome_01": 4, "genome_02": 2}})]
        snap = E.build(runs)
        c = snap["coverage"]
        self.assertEqual((c["genomes_in_runs"], c["genomes_rated"], c["complete"]), (3, 2, False))
        self.assertEqual(c["genomes_unrated"], [{"voice": "VCTK/p1", "run": "r1", "genome": 3}])
        self.assertTrue(any("1 of 3 genomes have no score" in x for x in snap["caveats"]))

    def test_one_listener_is_not_enough_evidence_for_a_winner(self):
        runs = [self.run_folder("r1", "VCTK/p1", {"Ash": {"baseline": 1, "genome_01": 5, "genome_02": 1, "genome_03": 1}})]
        voice = E.build(runs)["voices"][0]
        self.assertEqual(voice["status"], "not_enough_evidence")
        self.assertNotIn("winner", voice)
        self.assertEqual(voice["candidates"][0]["n"], 1)

    def test_a_clear_winner_with_enough_listeners(self):
        scores = {who: {"baseline": 2, "genome_01": s, "genome_02": 2, "genome_03": 1.5}
                  for who, s in (("Ash", 5), ("Rama", 4.5), ("Julius", 5))}
        voice = E.build([self.run_folder("r1", "VCTK/p1", scores)])["voices"][0]
        self.assertEqual((voice["status"], voice["winner"]["genome"]), ("winner", 1))

    def test_overlapping_listeners_give_no_clear_winner(self):
        scores = {who: {"baseline": 2, "genome_01": a, "genome_02": b, "genome_03": 1}
                  for who, a, b in (("Ash", 5, 4), ("Rama", 2, 4), ("Julius", 4.5, 3.5))}
        voice = E.build([self.run_folder("r1", "VCTK/p1", scores)])["voices"][0]
        self.assertEqual(voice["status"], "no_clear_winner")

    def test_artifact_checks_block_a_genome_before_its_score_is_used(self):
        scores = {who: {"baseline": 2, "genome_01": 5, "genome_02": 5, "genome_03": 4} for who in ("Ash", "Rama")}
        genes = {3: [{"feature": "hissing", "position": 0.5}]}
        snap = E.build([self.run_folder("r1", "VCTK/p1", scores, loud=(1,), artifacts={"Rama": {"genome_02": True}},
                                        genes=genes)])
        rows = {g["genome"]: g for g in snap["genomes"]}
        self.assertIn("audio check: burst", rows[1]["blocked_by"])
        self.assertIn("1 of 1 listeners heard an artifact", rows[2]["blocked_by"])
        self.assertTrue(any("can no longer be rendered" in b for b in rows[3]["blocked_by"]))
        self.assertEqual(snap["voices"][0]["status"], "no_safe_genome")
        self.assertFalse(any(g["recommendable"] for g in snap["genomes"]))

    def test_provenance_says_what_the_answer_rests_on_and_the_id_is_stable(self):
        runs = [self.run_folder("r1", "VCTK/p1", {"Ash": {"baseline": 3, "genome_01": 4},
                                                   "Test user 3": {"genome_01": 0}})]
        first, second = E.build(runs), E.build(runs)
        p = first["provenance"]
        for key in ("snapshot_id", "created", "code_commit", "ratings_up_to", "rating_log_sha256", "rule", "winner_rule"):
            self.assertIn(key, p)
        self.assertEqual(p["snapshot_id"], second["provenance"]["snapshot_id"])
        self.assertEqual(first["coverage"]["test_account_scores_left_out"], 1)
        self.assertEqual(first["coverage"]["listeners"], {"Ash": {"genome_scores": 1, "mean": 4.0}})
        brief = E.summary(first)
        self.assertEqual(brief["provenance"]["runs"], 1)
        self.assertIn("highest_safe", brief["voices"][0])


class SecondListener(Runs):
    def test_the_top_safe_genomes_go_to_a_listener_who_did_not_hear_them_and_their_scores_merge(self):
        from backend.genome import second_listen as SL

        runs = []
        for pile, (who, voice) in enumerate((("Ash", "VCTK/p1"), ("Rama", "VCTK/p2")), start=1):
            folder, run = self.run_folder(f"src_p{pile}", voice, {who: {"baseline": 2, "genome_01": 5, "genome_02": 4,
                                                                        "genome_03": 1}}, loud=(2,))
            run.update(batch=SL.SOURCE_BATCH, pile=pile)
            runs.append((folder, run))
        snap = E.build(runs)
        rows = SL.plan(snap, runs, per_voice=2)
        first = rows[0]
        self.assertEqual((first["genomes"], first["for_listener"]), ([1, 3], "Rama"))   # genome 2 failed the audio check
        made = SL.build(rows, runs, runs_dir=self.dir / "new")
        self.assertEqual(len(made), 2)
        self.assertEqual(SL.build(rows, runs, runs_dir=self.dir / "new"), [])          # made once
        copy = S.load_run(made[0])
        self.assertEqual(([g["number"] for g in copy["genomes"]], copy["blind"], copy["second_listen_of"]),
                         ([1, 3], True, "src_p1"))
        self.assertTrue((made[0] / "genome_03.wav").is_file() and (made[0] / "baseline.wav").is_file())

        R.save_scores(made[0], {"Rama": {"baseline": 2, "genome_01": 5, "genome_03": 1}})
        R.save_scores(made[0], {"Ash": {"genome_01": 0}})                              # already rated it: ignored
        merged = E.build(runs + [(made[0], copy)])
        g1 = next(g for g in merged["genomes"] if g["run"] == "src_p1" and g["genome"] == 1)
        self.assertEqual((g1["n"], g1["listeners"], g1["mos"]), (2, ["Ash", "Rama"], 5.0))
        self.assertEqual(merged["coverage"]["genomes_in_runs"], 6)                     # no new genomes
        self.assertEqual(merged["coverage"]["second_listener_runs"], 1)
        self.assertTrue(any("already rated the original" in c for c in merged["caveats"]))


if __name__ == "__main__":
    unittest.main()

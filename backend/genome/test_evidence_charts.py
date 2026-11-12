"""Tests of the evidence charts: the date range, the per-gene-type rules, and drawing from a small made-up corpus.

    python -m unittest backend.genome.test_evidence_charts
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))    # run as a file by main_backend.py test

from backend.genome import evidence_charts as ec  # noqa: E402


def clip(n, listener, score, above, types, when, voice="VCTK/p250", artifact=None):
    return {"id": f"clip:run/{voice}/{n}", "kind": "clip", "revision": when,
            "fields": {"voice": voice, "run": voice, "genome": n, "listener": listener, "score": score, "above_plain": above,
                       "features": len(types), "gene_types": types, "artifact": artifact}}


def plain(listener, when, voice="VCTK/p250"):
    return {"id": f"clip:run/{voice}/plain/{listener}", "kind": "clip", "revision": when,
            "fields": {"voice": voice, "run": voice, "genome": None, "listener": listener, "score": 3.0, "above_plain": None,
                       "features": 0, "gene_types": [], "artifact": None}}


def sanity(n, types, verdict, when="2026-11-08 03:13:11"):
    return {"id": f"sanity:{n}", "kind": "sanity", "revision": when, "fields": {"gene_types": types, "verdict": verdict}}


def corpus():
    """Ash likes hissing; Bo dislikes speed. Day 1 is Ash, day 2 is Bo."""
    out = [plain("Ash", "2026-11-10 09:00:00"), plain("Bo", "2026-11-11 09:00:00")]
    for i in range(12):
        out.append(clip(i + 1, "Ash", 4.0 if i % 2 else 2.0, 1.0 if i % 2 else -1.0, ["noise"], "2026-11-10 10:00:00"))
        out.append(clip(i + 20, "Bo", 1.0, -2.0, ["speed"], "2026-11-11 10:00:00", artifact=True))
        out.append(clip(i + 40, "Bo", 4.0, 1.0, ["noise", "sigh"], "2026-11-11 11:00:00", artifact=False))
    out += [sanity(i, ["speed"], "failed" if i < 3 else "passed") for i in range(6)]
    return out


class Dates(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(ec.parse_when("2026-11-11"), "2026-11-11 00:00:00")
        self.assertEqual(ec.parse_when("2026-11-11", end=True), "2026-11-11 23:59:59")
        self.assertEqual(ec.parse_when("2026-11-11 16:30"), "2026-11-11 16:30:00")
        self.assertIsNone(ec.parse_when(None))
        with self.assertRaises(ValueError):
            ec.parse_when("11/11/2026")

    def test_range_picks_scores_by_when_they_were_given(self):
        records = corpus()
        self.assertEqual(len(ec.select(records)["genomes"]), 36)
        day2 = ec.select(records, ec.parse_when("2026-11-11"), ec.parse_when("2026-11-11", end=True))
        self.assertEqual({g["listener"] for g in day2["genomes"]}, {"Bo"})
        self.assertEqual(day2["sanity"], [])                       # the sanity check was made on 2026-11-08
        self.assertEqual(day2["listeners"], ["Ash", "Bo"])         # every listener, so colours never change
        evening = ec.select(records, "2026-11-11 10:30:00", None)
        self.assertEqual(len(evening["genomes"]), 12)

    def test_single_gene_sanity_checks_win_when_there_are_any(self):
        records = corpus()
        self.assertEqual(ec.select(records)["sanity_label"], "random-v2, 3 genes per genome")
        alone = [{"id": f"sanity:alone/{i}", "kind": "sanity", "revision": "2026-11-12 10:00:00",
                  "fields": {"gene_types": [t], "genes": 1, "verdict": "passed"}} for i, t in enumerate(["hissing", "speed"])]
        data = ec.select(records + alone)
        self.assertEqual(data["sanity_label"], "each gene type alone")
        self.assertEqual(len(data["sanity"]), 2)
        self.assertEqual(len(ec.select(records + alone, None, "2026-11-11 23:59:59")["sanity"]), 6)   # before it was run

    def test_plain_lines_are_not_genomes(self):
        self.assertTrue(all(g["genome"] is not None for g in ec.select(corpus())["genomes"]))


class Rules(unittest.TestCase):
    def test_gene_type_table(self):
        rows = {r["gene_type"]: r for r in ec.gene_type_table(ec.select(corpus()))}
        self.assertEqual(rows["speed"]["suggestion"], "Lower chance")
        self.assertEqual(rows["speed"]["sanity_failed_pct"], 50)
        self.assertEqual(rows["sigh"]["suggestion"], "Keep")
        self.assertTrue(rows["sigh"]["consistent"])
        self.assertFalse(rows["speed"]["consistent"])
        self.assertIsNone(rows["noise"]["sanity_failed_pct"])     # never sanity-checked

    def test_round_half_up(self):
        self.assertEqual(ec.pct(1, 8), 13)
        self.assertEqual(ec.pct(0, 0), 0)


class Drawing(unittest.TestCase):
    def test_all_six_charts_and_comments(self):
        with tempfile.TemporaryDirectory() as folder:
            summary = ec.make_charts(out_dir=folder, records=corpus())
            self.assertEqual([c["number"] for c in summary["charts"]], ["1", "2", "3", "4", "5", "6"])
            for c in summary["charts"]:
                self.assertTrue(Path(c["file"]).is_file(), c)
            self.assertIn("6. Which genomes can we exclude?", (Path(folder) / "comments.txt").read_text(encoding="utf-8"))
            self.assertEqual(json.loads((Path(folder) / "summary.json").read_text(encoding="utf-8"))["genome_scores"], 36)

    def test_an_empty_range_skips_every_chart(self):
        with tempfile.TemporaryDirectory() as folder:
            summary = ec.make_charts("2030-01-01", "2030-01-02", out_dir=folder, records=corpus())
            self.assertTrue(all("skipped" in c for c in summary["charts"]))

    def test_a_backwards_range_is_refused(self):
        with self.assertRaises(ValueError):
            ec.make_charts("2026-11-12", "2026-11-11", out_dir=tempfile.gettempdir(), records=corpus())

    def test_folder_names(self):
        self.assertEqual(ec.folder_name(None, None), "all dates")
        self.assertEqual(ec.folder_name("2026-11-11 16:00:00", None), "2026-11-11_16-00 to now")


if __name__ == "__main__":
    unittest.main()

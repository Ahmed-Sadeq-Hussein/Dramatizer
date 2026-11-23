"""test_mos_study.py | Only assigned piles count, and the Report evaluation's numbers come out as totals."""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("DATABASE_PATH", str(Path(os.environ.get("TEMP", "/tmp")) / "test_mos_study.db"))

from backend.genome import mos_study as M  # noqa: E402


def row(account, batch, pile, run="r", clip="genome_01", score="3.0", rater=None, **more):
    return {"account": account, "rater": rater or account.title(), "batch": batch, "pile": str(pile), "run": run,
            "clip": clip, "score": score, "artifact": "no", "different_person": "no", "above_plain_line": "",
            **more}


class AssignedOnly(unittest.TestCase):
    def test_a_score_counts_only_in_a_pile_assigned_to_its_listener(self):
        given = {"ash": {"report-evaluation|1"}, "julius": {"report-evaluation|1"}}
        rows = [row("ash", "report-evaluation", 1), row("julius", "report-evaluation", 1),
                row("julius", "second-listener-v1", 1), row("ash", "", ""), row("test_user_3", "mos-diverse-v1", 1),
                row("ash", "report-evaluation", 2)]
        kept, left = M.assigned_scores(rows, given)
        self.assertEqual([(r["account"], r["pile"]) for r in kept], [("ash", "1"), ("julius", "1")])
        self.assertEqual(sum(left.values()), 4)
        self.assertEqual(left["not in a pile"], 1)


class Statistics(unittest.TestCase):
    def records(self):
        out = []
        for i in range(12):
            for who in ("A", "B"):
                for c, s in (("plain", 4.0), ("random", 3.0 + (i % 2)), ("guarded", 3.5)):
                    out.append({"item": f"i{i}", "listener": who, "condition": c, "score": s, "voice": "VCTK/p1",
                                "dataset": "VCTK", "sentence": "s1", "length": "short", "situation": "anger",
                                "genes": 1, "artifact": False, "different": False, "passed_checks": True,
                                "guard_failed": False, "guard_tries": 1, "gene_types": ["sigh"]})
        return out

    def test_paired_differences_and_holm(self):
        import random
        pairs = {(p["a"], p["b"]): p for p in M.paired(self.records(), random.Random(1))}
        g = pairs[("guarded", "plain")]
        self.assertEqual((g["pairs"], g["mean_difference"], g["b_higher"]), (24, -0.5, 24))
        self.assertLessEqual(g["wilcoxon_p"], g["holm_p"])
        self.assertEqual(M.winners(self.records())["plain"], 9)        # half the items tie with random: half a point

    def test_where_the_events_sit(self):
        self.assertEqual(M.tag_position("Don't you dare walk away from me [quick_breath]."), "at a pause")
        self.assertEqual(M.tag_position("Oh [mn], wonderful."), "at a pause")
        self.assertEqual(M.tag_position("Don't you dare walk away from [quick_breath] me."), "before the last word")
        self.assertEqual(M.tag_position("Don't [clucking] you dare walk away."), "inside a phrase")
        self.assertIsNone(M.tag_position("Don't you <strong>dare</strong> walk away."))

    def test_harm_is_totals_only(self):
        records = self.records()
        for r in records:
            r["markup"], r["measures"] = "Hi [sigh], you.", {"wer": 0.0, "similarity": 0.95, "pace": 1.0}
        h = M.harm(M.genome_diffs(records))
        self.assertEqual(h["genomes"], 24)
        self.assertEqual(h["positions"]["at a pause"]["genomes"], 24)
        self.assertNotIn("i0", repr(h))                      # no item names: totals only

    def test_holm_keeps_the_order_and_never_lowers(self):
        self.assertEqual(M._holm([0.01, 0.04, None, 0.03]), [0.03, 0.06, None, 0.06])


if __name__ == "__main__":
    unittest.main()

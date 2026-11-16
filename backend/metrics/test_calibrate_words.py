"""Tests of calibrate_words.py: the four words rules, judged on hand-made transcripts (no audio, no model).

    python -m unittest backend.metrics.test_calibrate_words
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))    # run as a file by main_backend.py test

from backend.metrics import calibrate_words as cw  # noqa: E402

SCRIPT = "We should have left the house an hour ago, but nobody could find the keys, and now the last train is gone."


class Rules(unittest.TestCase):
    def test_identical_clips_pass_every_rule(self):
        v = cw.verdicts(SCRIPT, SCRIPT, SCRIPT)
        self.assertTrue(all(v[r] for r in cw.RULES))

    def test_a_broken_word_fails_the_new_rule_but_not_the_old_one(self):
        broken = SCRIPT.replace("keys", "peas")
        v = cw.verdicts(SCRIPT, SCRIPT, broken)
        self.assertTrue(v["now"])                  # 1 word of 22 is inside 10 %
        self.assertTrue(v["one_word"])
        self.assertFalse(v["same"])
        self.assertFalse(v["no_new"])              # the plain render said "keys" right

    def test_a_word_the_plain_render_also_gets_wrong_is_not_held_against_the_genome(self):
        wrong = SCRIPT.replace("hour", "our")
        v = cw.verdicts(SCRIPT, wrong, wrong)
        self.assertTrue(v["no_new"])
        self.assertEqual((v["_base_wrong"], v["_test_wrong"]), (1, 1))

    def test_fillers_do_not_count(self):
        v = cw.verdicts(SCRIPT, SCRIPT, "Uh, " + SCRIPT)
        self.assertTrue(v["same"] and v["no_new"])

    def test_two_changed_words_on_a_long_line_pass_only_the_current_rule(self):
        two = SCRIPT.replace("keys", "peas").replace("train", "rain")
        v = cw.verdicts(SCRIPT, SCRIPT, two)
        self.assertEqual([r for r in cw.RULES if v[r]], ["now"])

    def test_summary_suggests_the_new_rule_only_when_it_passes_chance(self):
        ok = {"line": 0, "now": True, "one_word": True, "same": True, "no_new": True}
        bad = {**ok, "same": False, "no_new": False}
        genomes = [{"other_checks_ok": True, "_test_wrong": 0, "_base_wrong": 0, "_words": 8, **ok}]
        self.assertEqual(cw.summarise([ok] * 99 + [bad], [], genomes)["suggested"], "no_new")
        self.assertNotEqual(cw.summarise([ok] * 9 + [bad] * 1, [], genomes)["suggested"], "no_new")   # 90 % < 95 %
        lo, hi = cw.interval([ok] * 99 + [bad], "no_new")
        self.assertTrue(lo < 0.99 < hi)


if __name__ == "__main__":
    unittest.main()

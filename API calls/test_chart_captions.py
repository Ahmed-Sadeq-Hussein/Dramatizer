"""Tests of chart_captions.py with a stand-in for Gemma (no GPU needed).

    cd "API calls"; python -m unittest test_chart_captions
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import chart_captions as cc  # noqa: E402
from backend.genome import evidence_charts as ec  # noqa: E402
from backend.genome.test_evidence_charts import corpus  # noqa: E402


class FakeGemma:
    """Replies in turn from a list (step 1 and step 2 alike); records what it was sent."""
    name = "fake-gemma"

    def __init__(self, replies):
        self.replies, self.sent = list(replies), []

    def chat(self, messages):
        self.sent.append(messages)
        return self.replies.pop(0) if self.replies else '{"answer": "Nothing more to say."}'


def facts(*paths):
    return json.dumps({"facts": list(paths)})


def answer(text):
    return json.dumps({"answer": text})


class Paths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.summary = ec.make_charts(out_dir=self.tmp.name, records=corpus())
        self.c = {c["number"]: c for c in self.summary["charts"]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_resolve(self):
        r = lambda p, n="5": cc.resolve(p, self.summary, self.c[n])          # noqa: E731
        self.assertEqual(r("by_features[features=2].genomes"), 12)
        self.assertEqual(r("by_features[0].pct"), r("by_features[features=1].pct"))
        self.assertEqual(r("too_few_to_trust_below"), 20)
        self.assertEqual(r("per_voice[voice=VCTK/p250].genomes_scored", "1"), 36)
        self.assertEqual(r("chart1:voices"), 1)
        self.assertEqual(r("set:genome_scores"), 36)
        for bad in ("nope", "by_features[features=9].pct", "by_features[7].pct", "chart9:x", "voices.x"):
            with self.assertRaises(cc.PathError):
                r(bad)


class Captions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.summary = ec.make_charts(out_dir=self.folder, records=corpus())
        self.q1 = next(c for c in self.summary["charts"] if c["number"] == "1")
        self.head = [p for p, _ in cc.headline_facts(self.q1)]

    def picked(self, out):
        """The facts Gemma chose, after the chart's headline numbers that are always given."""
        return [f for f in out["facts"] if f["path"] not in self.head]

    def test_the_headline_numbers_are_always_facts(self):
        """Chart 1 once got a text about one voice only: the counts over all voices are now always given."""
        self.assertEqual(self.head, ["voices", "voices_with_a_genome_above_plain", "voices_with_a_tie_for_first",
                                     "genomes", "genomes_scored_by_one_listener_only"])
        model = FakeGemma([facts("per_voice[voice=VCTK/p250].best_above_plain"), answer("1 voice was scored, 36 genomes.")])
        out = cc.write_one(model, cc.Tools(self.summary), self.summary, self.q1)
        self.assertEqual([f["path"] for f in out["facts"]], self.head + ["per_voice[voice=VCTK/p250].best_above_plain"])
        self.assertIn("Always included", model.sent[0][1]["content"])
        step2 = model.sent[1][1]["content"]
        self.assertIn("- genomes_scored_by_one_listener_only = 36 (headline)", step2)
        self.assertNotIn("best_above_plain = 1.0 (headline)", step2)
        nothing_more = FakeGemma([facts(), answer("1 voice was scored, 36 genomes.")])            # headline alone is enough
        self.assertEqual(len(cc.write_one(nothing_more, cc.Tools(self.summary), self.summary, self.q1)["facts"]),
                         len(self.head))

    def tearDown(self):
        self.tmp.cleanup()

    def test_facts_are_looked_up_by_code_and_the_text_is_written_from_them(self):
        model = FakeGemma([facts("voices", "per_voice[voice=VCTK/p250].best_above_plain"),
                           answer("Only 1 voice was scored; its best genome was 1 point above its plain line.")])
        out = cc.write_one(model, cc.Tools(self.summary), self.summary, self.q1)
        self.assertEqual([f["path"] for f in self.picked(out)], ["per_voice[voice=VCTK/p250].best_above_plain"])
        self.assertEqual(self.picked(out)[0]["value"], 1.0)                       # the value from the data, not from Gemma
        step2 = model.sent[1][1]["content"]
        self.assertIn("- per_voice[voice=VCTK/p250].best_above_plain = 1.0", step2)
        self.assertNotIn(self.q1["claim"], step2)                             # never our rule-based sentence
        self.assertNotIn("unverified_numbers", out)

    def test_a_path_not_in_the_numbers_is_sent_back_once(self):
        model = FakeGemma([facts("voices", "best_voice"), facts("voices_with_a_genome_above_plain"),
                           answer("1 voice.")])
        out = cc.write_one(model, cc.Tools(self.summary), self.summary, self.q1)
        self.assertIn("best_voice", model.sent[1][-1]["content"])
        self.assertIn("the keys are", model.sent[1][-1]["content"])
        self.assertEqual([f["path"] for f in out["facts"]], self.head)          # the wrong path is left out

    def test_a_number_that_is_not_a_fact_is_sent_back_once(self):
        model = FakeGemma([facts("voices"), answer("97% of voices improved."), answer("1 voice was scored, 36 genomes.")])
        out = cc.write_one(model, cc.Tools(self.summary), self.summary, self.q1)
        self.assertEqual(out["text"], "1 voice was scored, 36 genomes.")
        self.assertIn("97", model.sent[-1][-1]["content"])
        stubborn = FakeGemma([facts("voices"), answer("97% improved."), answer("Still 97%.")])
        self.assertEqual(cc.write_one(stubborn, cc.Tools(self.summary), self.summary, self.q1)["unverified_numbers"], [97.0])

    def test_a_number_from_the_whole_table_is_not_enough(self):
        """Only the chosen facts count: 12 is in chart 1's numbers, but not among the facts picked."""
        model = FakeGemma([facts("voices"), answer("12 genomes stood out."), answer("12 genomes stood out.")])
        out = cc.write_one(model, cc.Tools(self.summary), self.summary, self.q1)
        self.assertEqual(out["unverified_numbers"], [12.0])

    def test_tools_then_facts(self):
        model = FakeGemma(['{"tool": "chart_numbers", "arguments": {"number": "5"}}',
                           facts("chart5:by_features[features=1].genomes"), answer("24 genomes had one feature.")])
        out = cc.write_one(model, cc.Tools(self.summary), self.summary, self.q1)
        self.assertEqual(out["tools"], ["chart_numbers"])
        self.assertEqual(self.picked(out)[0]["value"], 24)
        self.assertNotIn("unverified_numbers", out)

    def test_a_whole_table_is_not_a_fact(self):
        """Gemma once picked chart 1's whole per-voice table and wrote it out voice by voice."""
        cc_max = cc.MAX_FACT_CHARS
        cc.MAX_FACT_CHARS = 20
        try:
            model = FakeGemma([facts("per_voice"), facts(), answer("1 voice was scored, 36 genomes.")])
            out = cc.write_one(model, cc.Tools(self.summary), self.summary, self.q1)
        finally:
            cc.MAX_FACT_CHARS = cc_max
        self.assertNotIn("per_voice", [f["path"] for f in out["facts"]])
        self.assertIn("a whole table", model.sent[1][-1]["content"])

    def test_a_text_about_one_example_only_is_sent_back_once(self):
        """Chart 1 kept getting a text about one voice; the counts over all voices must be in it."""
        model = FakeGemma([facts("per_voice[voice=VCTK/p250].best_above_plain"),
                           answer("For VCTK/p250 the best genome was 1.0 above plain."),
                           answer("Of 1 voice, 1 had a genome above its plain line; all 36 genomes had one listener.")])
        out = cc.write_one(model, cc.Tools(self.summary), self.summary, self.q1)
        self.assertIn("all 36 genomes", out["text"])
        self.assertIn("one example only", model.sent[-1][-1]["content"])
        self.assertTrue(cc.misses_headline("Nothing here.", [("lower_chance", ["speed", "hissing"])]))
        self.assertFalse(cc.misses_headline("Give speed a lower chance.", [("lower_chance", ["speed", "hissing"])]))

    def test_a_cut_off_answer_is_asked_for_again(self):
        model = FakeGemma([facts(), '```json { "answer": "For the voice CREMA-D/1014, the best', answer("1 voice was scored, 36 genomes.")])
        out = cc.write_one(model, cc.Tools(self.summary), self.summary, self.q1)
        self.assertEqual(out["text"], "1 voice was scored, 36 genomes.")
        self.assertIn("not one whole JSON object", model.sent[-1][-1]["content"])

    def test_whole_run_keeps_texts_and_skips_charts_without_data(self):
        replies = []
        for _ in range(6):
            replies += [facts("set:genome_scores"), answer("36 genome scores so far.")]
        written = cc.write_captions(self.folder, FakeGemma(replies), log=lambda *_: None)
        self.assertEqual(written, 6)
        shown = json.loads((self.folder / "summary.json").read_text(encoding="utf-8"))
        cap = next(c for c in shown["charts"] if c["number"] == "1")["caption"]
        self.assertEqual((cap["model"], cap["facts"][-1]["value"]), ("fake-gemma", 36))
        self.assertEqual(cc.write_captions(self.folder, FakeGemma([]), log=lambda *_: None), 0)   # current: kept
        ec.make_charts("2030-01-01", "2030-01-02", out_dir=self.folder, records=corpus())
        self.assertEqual(cc.write_captions(self.folder, FakeGemma([]), log=lambda *_: None), 0)


if __name__ == "__main__":
    unittest.main()

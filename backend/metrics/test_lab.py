"""Tests of the Lab's measuring (lab.py), with stand-in models: no Whisper, no GPU.

    python -m unittest backend.metrics.test_lab
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))    # run as a file by main_backend.py test

from backend.metrics import lab  # noqa: E402


class StandIn(lab.Measurer):
    """Hears and embeds each file by what is written in it: "heard|x,y|seconds"."""

    def __init__(self):
        self.listened = 0

    def _parts(self, path):
        return Path(path).read_text(encoding="utf-8").split("|")

    def transcribe(self, path):
        self.listened += 1
        return self._parts(path)[0]

    def embedding(self, path):
        return [float(v) for v in self._parts(path)[1].split(",")]

    def qualities(self, path):
        return {"stoi": 0.9, "pesq": 3.21234, "si_sdr": 12.0}

    def seconds(self, path):
        return float(self._parts(path)[2])

    def speech_seconds(self, path):
        return float(self._parts(path)[2]) - 0.5

    def models(self):
        return {"wer": "stand-in"}


class Compare(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.dir = Path(folder.name)
        lab._kept.clear()

    def file(self, name, content):
        path = self.dir / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_wer_similarity_and_the_matrix(self):
        original = self.file("o.wav", "the cat sat on the mat|1,0|3.5")
        good = self.file("g.wav", "the cat sat on the mat|0.9,0.1|2.5")
        bad = self.file("b.wav", "uh the bat sat on mat now|0,1|2.5")
        result = lab.compare([{"label": "Original", "path": original, "text": "The cat sat on the mat."},
                              {"label": "Good", "path": good, "text": "The cat [sigh] sat on the <emphasis>mat</emphasis>."},
                              {"label": "Bad", "path": bad, "text": "The cat sat on the mat."}],
                             reference=0, measurer=StandIn())
        o, g, b = result["recordings"]
        self.assertTrue(o["reference"])
        self.assertIsNone(o["similarity"])                          # the reference is not compared with itself
        self.assertEqual(g["text"], "The cat sat on the mat.")      # gene tags are not words
        self.assertEqual(g["wer"], 0.0)
        self.assertEqual((b["errors"], b["word_count"]), (3, 6))    # cat->bat, "the" missing, "now" extra; "uh" ignored
        ops = [w["op"] for w in b["words"]]
        self.assertIn("substitute", ops)
        self.assertGreater(g["similarity"], 0.9)
        self.assertAlmostEqual(b["similarity"], 0.0)
        self.assertEqual(len(result["matrix"]), 3)
        self.assertEqual(result["matrix"][1][2], result["matrix"][2][1])
        self.assertEqual(result["matrix"][0][0], 1.0)
        self.assertEqual(g["quality"]["pesq"], 3.212)
        self.assertEqual(g["words_per_second"], round(6 / 2.0, 2))

    def test_a_file_is_heard_once_and_only_what_is_asked_is_measured(self):
        clip = self.file("c.wav", "hello there|1,1|2")
        standin = StandIn()
        for _ in range(2):
            lab.compare([{"label": "A", "path": clip, "text": "Hello there"}], measurer=standin)
        self.assertEqual(standin.listened, 1)
        only = lab.compare([{"label": "A", "path": clip}], measure=["similarity"], measurer=standin)
        row = only["recordings"][0]
        self.assertNotIn("wer", row)
        self.assertNotIn("quality", row)
        self.assertEqual(only["matrix"], [[1.0]])
        no_script = lab.compare([{"label": "A", "path": clip}], measurer=standin)["recordings"][0]
        self.assertIsNone(no_script["wer"])
        self.assertEqual(no_script["heard"], "hello there")

    def test_refuses_too_many_or_a_wrong_reference(self):
        clip = self.file("c.wav", "hi|1,0|1")
        with self.assertRaises(ValueError):
            lab.compare([], measurer=StandIn())
        with self.assertRaises(ValueError):
            lab.compare([{"label": str(i), "path": clip} for i in range(lab.MAX_RECORDINGS + 1)], measurer=StandIn())
        with self.assertRaises(ValueError):
            lab.compare([{"label": "A", "path": clip}], reference=1, measurer=StandIn())

    def test_alignment(self):
        self.assertEqual(lab.alignment("a b", ""), [{"op": "delete", "expected": ["a", "b"], "heard": []}])
        self.assertEqual(lab.alignment("", "x"), [{"op": "insert", "expected": [], "heard": ["x"]}])
        self.assertEqual([w["op"] for w in lab.alignment("one two three", "one too three")], ["equal", "substitute", "equal"])


if __name__ == "__main__":
    unittest.main()

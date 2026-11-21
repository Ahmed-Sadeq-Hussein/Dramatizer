"""Tests of the glossary (glossary.py): which term a question asks about. No models.

    python -m unittest backend.test_glossary
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # run as a file by main_backend.py test

from backend import glossary  # noqa: E402


class Asked(unittest.TestCase):
    def test_a_plain_definition_names_its_term(self):
        for question, term in (("what is STOI", "STOI"), ("What does WER mean?", "WER"),
                               ("explain the sanity check", "Sanity check"), ("what is a genome", "Genome"),
                               ("what does PESQ stand for", "PESQ"), ("what is si-sdr", "SI-SDR"),
                               ("what is the plain line", "Plain line"), ("define speaker similarity", "Speaker similarity"),
                               ("what does laughing speech do", "Laughing speech (gene)")):
            entry, follow_up = glossary.asked(question)
            self.assertEqual((entry and entry["term"], follow_up), (term, None), question)

    def test_a_follow_up_names_nothing(self):
        self.assertEqual(glossary.asked("but specifically what does it measure?"), (None, "follow-up"))

    def test_other_questions_are_not_definitions(self):
        for question in ("what is the best genome", "which genomes did Rama rate 5", "how many genomes are there",
                         "what is the MOS of genome 3 of VCTK/p250"):
            self.assertEqual(glossary.asked(question), (None, None), question)

    def test_every_entry_is_complete_and_named_once(self):
        names = [n for e in glossary.entries() for n in [e["term"].lower()]]
        self.assertEqual(len(names), len(set(names)))
        for e in glossary.entries():
            self.assertTrue(e["meaning"] and e["kind"], e["term"])
        self.assertEqual(glossary.term("word error rate")["term"], "WER")


if __name__ == "__main__":
    unittest.main()

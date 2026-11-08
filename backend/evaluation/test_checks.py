"""test_checks.py | Known-answer tests for checks.py.

    python backend/evaluation/test_checks.py

A small made-up manifest where the right answer is known in advance. A correct
split must pass every check. Each single fault must fail exactly the check it
belongs to. No files, no models: it runs anywhere.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checks import FIELDS, contract_checks, run_checks, speaker_key  # noqa: E402

CHECK_NAMES = [
    "validation and test are both present",
    "validation and test share no speaker",
    "every clip belongs to a speaker in the split",
    "no clip has an empty script",
]


def rows_for(speakers):
    return [
        {"dataset": "VCTK", "speaker": s, "clip_id": f"{s}_{i}", "text": f"sentence {i}"}
        for s in speakers
        for i in range(3)
    ]


def good_setup():
    rows = rows_for(["p225", "p226", "p227"])
    rows += [{"dataset": "CREMA-D", "speaker": s, "clip_id": f"{s}_x", "text": "hi"} for s in ["1001", "1002"]]
    split = {
        "validation": ["VCTK/p225", "VCTK/p226", "CREMA-D/1001"],
        "test": ["VCTK/p227", "CREMA-D/1002"],
    }
    return rows, split


def failed(results):
    return [name for name, ok in results if not ok]


class KnownAnswers(unittest.TestCase):
    def test_correct_setup_passes_every_check(self):
        rows, split = good_setup()
        results = run_checks(rows, split)
        self.assertEqual([name for name, _ in results], CHECK_NAMES)
        self.assertEqual(failed(results), [])

    def test_missing_test_set_fails_only_the_presence_check(self):
        rows, split = good_setup()
        split = {"validation": split["validation"] + split["test"], "test": []}
        self.assertEqual(failed(run_checks(rows, split)), [CHECK_NAMES[0]])

    def test_speaker_in_both_sets_fails_only_the_overlap_check(self):
        rows, split = good_setup()
        split["test"] = split["test"] + [split["validation"][0]]
        self.assertEqual(failed(run_checks(rows, split)), [CHECK_NAMES[1]])

    def test_clip_from_unknown_speaker_fails_only_the_membership_check(self):
        rows, split = good_setup()
        rows.append({"dataset": "VCTK", "speaker": "p999", "clip_id": "x", "text": "hi"})
        self.assertEqual(failed(run_checks(rows, split)), [CHECK_NAMES[2]])

    def test_empty_script_fails_only_the_script_check(self):
        rows, split = good_setup()
        rows[0]["text"] = "   "
        self.assertEqual(failed(run_checks(rows, split)), [CHECK_NAMES[3]])

    def test_missing_script_counts_as_empty(self):
        rows, split = good_setup()
        rows[0]["text"] = None
        self.assertEqual(failed(run_checks(rows, split)), [CHECK_NAMES[3]])

    def test_same_speaker_id_in_two_datasets_is_not_a_leak(self):
        # "1001" can exist in CREMA-D and in VCTK: the dataset is part of the key
        self.assertNotEqual(
            speaker_key({"dataset": "VCTK", "speaker": "1001"}),
            speaker_key({"dataset": "CREMA-D", "speaker": "1001"}),
        )


def contract_rows():
    """One clip of each dataset, with the labels the contract allows."""
    base = dict.fromkeys(FIELDS, "")
    return [
        {**base, "dataset": "VCTK", "clip_id": "p225_001", "speaker": "p225", "text": "hi", "mic": "mic1",
         "licence": "CC BY 4.0", "source": "a.wav"},
        {**base, "dataset": "CREMA-D", "clip_id": "1001_DFA_ANG_XX", "speaker": "1001", "text": "hi",
         "emotion": "anger", "intensity": "unspecified", "licence": "ODbL", "source": "b.wav"},
        {**base, "dataset": "LibriTTS-R", "clip_id": "19_198_000000_000000", "speaker": "19", "text": "hi",
         "licence": "CC BY 4.0", "source": "c.wav"},
    ]


class Contract(unittest.TestCase):
    def test_good_rows_pass_every_rule(self):
        self.assertEqual(failed(contract_checks(contract_rows(), audio_exists=lambda p: True)), [])

    def test_each_fault_fails_only_its_rule(self):
        cases = [
            (lambda rows: rows[0].update(licence="ODbL"), "every clip carries its dataset's licence"),
            (lambda rows: rows.append(dict(rows[0])), "no clip is listed twice"),
            (lambda rows: rows[1].update(emotion="joy"), "emotion is an allowed label (CREMA-D only)"),
            (lambda rows: rows[0].update(emotion="anger"), "emotion is an allowed label (CREMA-D only)"),
            (lambda rows: rows[1].update(intensity="extreme"), "intensity is an allowed label (CREMA-D only)"),
            (lambda rows: rows[2].update(mic="mic1"), "mic is an allowed label (VCTK only)"),
            (lambda rows: rows[2].update(speaker=" "), "every clip has a speaker"),
        ]
        for fault, rule in cases:
            rows = contract_rows()
            fault(rows)
            with self.subTest(rule=rule):
                self.assertEqual(failed(contract_checks(rows)), [rule])

    def test_unknown_dataset_fails_dataset_and_licence(self):
        rows = contract_rows()
        rows[0]["dataset"] = "Other"
        self.assertIn("every clip comes from a known dataset", failed(contract_checks(rows)))

    def test_columns_must_match(self):
        rows = [{k: v for k, v in r.items() if k != "mic"} for r in contract_rows()]
        self.assertIn("the manifest has exactly the contract's columns", failed(contract_checks(rows)))

    def test_missing_audio_fails_only_when_checked(self):
        self.assertEqual(failed(contract_checks(contract_rows())), [])
        self.assertEqual(failed(contract_checks(contract_rows(), audio_exists=lambda p: p != "b.wav")),
                         ["every clip's audio file exists"])


if __name__ == "__main__":
    unittest.main()

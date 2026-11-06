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

from checks import run_checks, speaker_key  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()

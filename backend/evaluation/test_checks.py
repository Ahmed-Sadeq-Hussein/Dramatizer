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

from checks import FIELDS, contract_checks, run_checks, speaker_key, voice_checks  # noqa: E402

CHECK_NAMES = [
    "validation and test are both present",
    "validation and test share no speaker",
    "every clip's speaker is a validation voice, a test voice or listed as unused",
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

    def test_unused_speaker_is_accounted_for(self):
        rows, split = good_setup()
        rows.append({"dataset": "CREMA-D", "speaker": "1003", "clip_id": "1003_x", "text": "hi"})
        self.assertEqual(failed(run_checks(rows, split)), [CHECK_NAMES[2]])
        split["unused"] = [{"speaker": "CREMA-D/1003", "why": "no neutral clip"}]
        self.assertEqual(failed(run_checks(rows, split)), [])

    def test_same_speaker_id_in_two_datasets_is_not_a_leak(self):
        # "1001" can exist in CREMA-D and in VCTK: the dataset is part of the key
        self.assertNotEqual(
            speaker_key({"dataset": "VCTK", "speaker": "1001"}),
            speaker_key({"dataset": "CREMA-D", "speaker": "1001"}),
        )


VOICE_CHECKS = [
    "every voice is listed once, with one clip",
    "the split's lists hold exactly the voices, each in its own split",
    "no unused speaker is also a voice",
    "every voice's clip is in the manifest and is its own speaker's",
    "every voice's clip is neutral, within the length limits",
    "each dataset's test share is 20 %, rounded, at least one voice",
]


def voice_setup():
    """Five VCTK voices (1 test) and one CREMA-D voice (1 test), each with one clip, plus one unused speaker."""
    speakers = ["p225", "p226", "p227", "p228", "p229"]
    rows = [{"dataset": "VCTK", "speaker": s, "clip_id": f"{s}_1", "text": "hi", "emotion": ""} for s in speakers]
    rows.append({"dataset": "CREMA-D", "speaker": "1001", "clip_id": "1001_NEU", "text": "hi", "emotion": "neutral"})
    rows.append({"dataset": "CREMA-D", "speaker": "1002", "clip_id": "1002_ANG", "text": "hi", "emotion": "anger"})
    voices = [{"voice": f"VCTK/{s}", "dataset": "VCTK", "clip_id": f"{s}_1", "speech_seconds": 9.0,
               "split": "test" if s == "p229" else "validation"} for s in speakers]
    voices.append({"voice": "CREMA-D/1001", "dataset": "CREMA-D", "clip_id": "1001_NEU", "speech_seconds": 3.5,
                   "split": "test"})
    split = {
        "rules": {"min_seconds": 3.0, "max_seconds": 25.0, "test_fraction": 0.2},
        "validation": [v["voice"] for v in voices if v["split"] == "validation"],
        "test": [v["voice"] for v in voices if v["split"] == "test"],
        "unused": [{"speaker": "CREMA-D/1002", "why": "no neutral clip"}],
        "voices": voices,
    }
    return rows, split


class Voices(unittest.TestCase):
    def test_correct_voices_pass_every_check(self):
        rows, split = voice_setup()
        results = voice_checks(rows, split)
        self.assertEqual([name for name, _ in results], VOICE_CHECKS)
        self.assertEqual(failed(results), [])
        self.assertEqual(failed(run_checks(rows, split)), [])

    def test_each_fault_fails_only_its_rule(self):
        def listed_twice(split):
            split["voices"].append(dict(split["voices"][0]))

        def wrong_list(split):
            split["validation"].remove("VCTK/p225")
            split["test"].append("VCTK/p225")

        def other_speakers_clip(split):
            split["voices"][0]["clip_id"] = "p226_1"

        def emotional_clip(split):
            split["voices"][5]["clip_id"] = "1002_ANG"
            split["voices"][5]["voice"] = "CREMA-D/1002"
            split["test"] = ["VCTK/p229", "CREMA-D/1002"]
            split["unused"] = []

        def too_short(split):
            split["voices"][0]["speech_seconds"] = 2.0

        def test_share(split):
            split["voices"][0]["split"] = "test"
            split["validation"].remove("VCTK/p225")
            split["test"].append("VCTK/p225")

        def unused_voice(split):
            split["unused"].append({"speaker": "VCTK/p225", "why": "x"})

        cases = [
            (listed_twice, VOICE_CHECKS[0]),
            (wrong_list, VOICE_CHECKS[1]),
            (unused_voice, VOICE_CHECKS[2]),
            (other_speakers_clip, VOICE_CHECKS[3]),
            (emotional_clip, VOICE_CHECKS[4]),
            (too_short, VOICE_CHECKS[4]),
            (test_share, VOICE_CHECKS[5]),
        ]
        for fault, rule in cases:
            rows, split = voice_setup()
            fault(split)
            with self.subTest(rule=rule, fault=fault.__name__):
                self.assertEqual(failed(voice_checks(rows, split)), [rule])


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

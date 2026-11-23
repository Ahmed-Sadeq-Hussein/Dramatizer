"""Tests of pacing.py and the natural speed in tts_formatter.synthesize (no model: a fake CosyVoice).

    python -m unittest backend.speech.test_pacing
"""
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))    # run as a file by main_backend.py test

from backend.speech import pacing, tts_formatter  # noqa: E402

SR = 16000


def tone(seconds, hz=220.0):
    t = np.arange(int(SR * seconds)) / SR
    return 0.5 * np.sin(2 * np.pi * hz * t)


def quiet(seconds):
    return np.random.default_rng(0).normal(0, 1e-4, int(SR * seconds))


def line():
    """Three "words" with two pauses inside (0.2 s and 0.3 s), and silence before and after."""
    return np.concatenate([quiet(0.3), tone(0.5), quiet(0.2), tone(0.4), quiet(0.3), tone(0.6), quiet(0.3)])


def main_hz(x):
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / SR)[np.argmax(spectrum)]


class Pauses(unittest.TestCase):
    def test_finds_only_the_pauses_inside_the_speech(self):
        found = [(round(a / SR, 2), round((b - a) / SR, 2)) for a, b in pacing.silences(line(), SR)]
        self.assertEqual([d for _, d in found], [0.2, 0.3])
        self.assertEqual([s for s, _ in found], [0.8, 1.4])

    def test_slower_lengthens_the_pauses_and_keeps_the_words(self):
        x = line()
        y = pacing.scale_pauses(x, SR, 1.5)
        self.assertAlmostEqual(len(y) / SR, len(x) / SR + 0.25, places=2)      # 0.5 s of pauses -> 0.75 s
        np.testing.assert_allclose(y[:int(0.8 * SR)], x[:int(0.8 * SR)])        # before the first pause: untouched
        np.testing.assert_allclose(y[-int(0.9 * SR):], x[-int(0.9 * SR):])      # the last word and the end: untouched
        found = [round((b - a) / SR, 2) for a, b in pacing.silences(y, SR)]
        self.assertEqual(found, [0.3, 0.45])

    def test_faster_shortens_them_but_not_below_the_floor(self):
        x = line()
        y = pacing.scale_pauses(x, SR, 0.1)
        found = [round((b - a) / SR, 2) for a, b in pacing.silences(np.concatenate([y, quiet(0.01)]), SR)]
        self.assertTrue(all(abs(d - pacing.MIN_PAUSE_MS / 1000) <= 0.02 for d in found))
        self.assertEqual(len(pacing.scale_pauses(x, SR, 1.0)), len(x))


class Wsola(unittest.TestCase):
    def test_length_changes_and_pitch_stays(self):
        x = tone(1.0, 220)
        for rate in (0.85, 1.15):
            y = pacing.wsola(x, rate, SR)
            self.assertEqual(len(y), int(len(x) / rate))
            self.assertAlmostEqual(main_hz(y[2000:-2000]), 220, delta=5)       # a resample would move it to 187 or 253

    def test_paced_prompt_is_made_once(self):
        import soundfile
        with tempfile.TemporaryDirectory() as folder:
            src = Path(folder) / "ref.wav"
            soundfile.write(str(src), tone(1.0).astype(np.float32), SR)
            out = pacing.paced_prompt(src, 0.85, Path(folder) / "cache")
            self.assertAlmostEqual(soundfile.info(out).duration, 1 / 0.85, places=2)
            with mock.patch.object(pacing, "wsola", side_effect=AssertionError("made again")):
                self.assertEqual(pacing.paced_prompt(src, 0.85, Path(folder) / "cache"), out)


class FakeCosyVoice:
    sample_rate = SR

    def __init__(self):
        self.calls = []

    def inference_zero_shot(self, text, prompt_text, prompt_wav, **kwargs):
        import torch
        self.calls.append({"prompt_wav": prompt_wav, "speed": kwargs.get("speed")})
        yield {"tts_speech": torch.from_numpy(line().astype(np.float32)).unsqueeze(0)}


@unittest.skipUnless(importlib.util.find_spec("torch"), "needs PyTorch (not installed for CI: requirements-ci.txt)")
class Synthesize(unittest.TestCase):
    def setUp(self):
        import soundfile
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.dir = Path(folder.name)
        self.ref = self.dir / "ref.wav"
        soundfile.write(str(self.ref), tone(1.0).astype(np.float32), SR)
        p = mock.patch.object(tts_formatter, "_paced_dir", lambda: self.dir / "paced")
        p.start()
        self.addCleanup(p.stop)

    def speak(self, method, rate):
        plan = tts_formatter.process_tts_string("Well, you did it.", [{"feature": "speed", "kwargs": {"rate": rate}}])
        model = FakeCosyVoice()
        with mock.patch.dict(os.environ, {"SPEED_METHOD": method}):
            audio = tts_formatter.synthesize(model, plan, str(self.ref), "Hello there.")
        return model.calls[0], audio.shape[1] / SR

    def test_natural_speaks_from_a_paced_reference_and_scales_the_pauses(self):
        call, seconds = self.speak("natural", 0.85)
        self.assertEqual(call["speed"], 1.0)                                    # no spectrogram stretch
        self.assertNotEqual(call["prompt_wav"], str(self.ref))
        self.assertIn("x0.850", call["prompt_wav"])
        self.assertAlmostEqual(seconds, len(line()) / SR + 0.5 * (1 / 0.85 - 1), places=2)

    def test_stretch_is_the_old_way(self):
        call, seconds = self.speak("stretch", 0.85)
        self.assertEqual((call["speed"], call["prompt_wav"]), (0.85, str(self.ref)))
        self.assertAlmostEqual(seconds, len(line()) / SR, places=3)            # the fake does not stretch

    def test_normal_speed_changes_nothing(self):
        call, seconds = self.speak("natural", 1.0)
        self.assertEqual((call["speed"], call["prompt_wav"]), (1.0, str(self.ref)))

    def test_unknown_method_refused(self):
        with mock.patch.dict(os.environ, {"SPEED_METHOD": "warp"}):
            with self.assertRaises(ValueError):
                tts_formatter.speed_method()


if __name__ == "__main__":
    unittest.main()

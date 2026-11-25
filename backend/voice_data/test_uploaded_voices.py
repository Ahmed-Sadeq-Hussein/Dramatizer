"""test_uploaded_voices.py | The words typed for an uploaded or recorded voice must match the recording (WER check).

    python backend/voice_data/test_uploaded_voices.py

Whisper is replaced by a stand-in that "hears" a fixed text, so this runs without it (and on CI).
"""
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.voice_data import uploaded_voices as uv  # noqa: E402

SCRIPT = "The old lighthouse keeper climbed the stairs every evening."


def recording(seconds=4.0):
    buffer = io.BytesIO()
    t = np.arange(int(16000 * seconds)) / 16000
    sf.write(buffer, (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32"), 16000, format="WAV")
    return buffer.getvalue()


def hears(text):
    return lambda mono, rate: text


class ScriptCheck(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.dict(os.environ, {"UPLOAD_MAX_WER": "0.35"})
        patch.start()
        self.addCleanup(patch.stop)
        self.mono = np.zeros(16000, dtype="float32")

    def check(self, heard, script=SCRIPT):
        return uv.check_script(self.mono, 16000, script, hear=hears(heard))

    def test_the_same_words_pass_whatever_the_case_and_punctuation(self):
        self.assertEqual(self.check("the old lighthouse keeper climbed the stairs every evening"), 0.0)

    def test_a_few_slips_by_whisper_pass(self):
        self.assertLessEqual(self.check("The old light house keeper climbed the stairs every evening."), 0.35)

    def test_unrelated_text_is_refused_and_says_what_was_heard(self):
        with self.assertRaises(uv.UploadRefused) as caught:
            self.check("Buy cheap watches now at our online shop")
        self.assertIn("do not match", str(caught.exception))
        self.assertIn("Buy cheap watches", str(caught.exception))
        self.assertIn("35%", str(caught.exception))

    def test_a_silent_or_wordless_recording_is_refused(self):
        with self.assertRaises(uv.UploadRefused):
            self.check("")

    def test_a_long_transcript_is_cut_in_the_message(self):
        with self.assertRaises(uv.UploadRefused) as caught:
            self.check("word " * 200)
        self.assertLess(len(str(caught.exception)), 500)

    def test_the_limit_is_a_setting_and_can_be_turned_off(self):
        with mock.patch.dict(os.environ, {"UPLOAD_MAX_WER": "0.9"}):
            self.assertGreater(self.check("the keeper climbed every evening"), 0.35)       # allowed at 0.9
        with mock.patch.dict(os.environ, {"UPLOAD_MAX_WER": "off"}):
            self.assertIsNone(self.check("nothing alike at all"))

    def test_a_broken_listener_refuses_instead_of_letting_it_through(self):
        def broken(mono, rate):
            raise RuntimeError("no whisper here")

        with self.assertRaises(uv.UploadRefused) as caught:
            uv.check_script(self.mono, 16000, SCRIPT, hear=broken)
        self.assertIn("could not be checked", str(caught.exception))


class Decode(unittest.TestCase):
    """An upload's length is checked before it is decoded (Vunrabilities.txt, finding 4)."""

    @staticmethod
    def flac(seconds, rate=16000):
        buffer = io.BytesIO()
        sf.write(buffer, np.zeros(int(rate * seconds), dtype="float32") + 0.2, rate, format="FLAC")
        return buffer.getvalue()

    def test_a_good_recording_decodes(self):
        mono, rate, seconds = uv._decode(recording(4.0), "voice.wav")
        self.assertEqual((rate, round(seconds, 1)), (16000, 4.0))

    def test_hours_packed_into_a_small_flac_are_refused_from_the_header(self):
        data = self.flac(2 * 3600)                     # two hours of a constant: well under the 8 MB cap
        self.assertLess(len(data), uv.MAX_BYTES)
        with mock.patch.object(uv.sf, "read", side_effect=AssertionError("decoded")):
            with self.assertRaisesRegex(uv.UploadRefused, "must be 3 to 30 s"):
                uv._decode(data, "voice.flac")

    def test_decoding_never_goes_past_the_limit_even_if_the_header_lies(self):
        data = self.flac(60)
        real_info = uv.sf.info

        def lying_info(file):
            about = real_info(file)
            about.frames = 16000 * 10                  # claims 10 s
            return about

        read = mock.Mock(wraps=uv.sf.read)
        with mock.patch.object(uv.sf, "info", lying_info), mock.patch.object(uv.sf, "read", read):
            with self.assertRaisesRegex(uv.UploadRefused, "longer than 30 s"):
                uv._decode(data, "voice.flac")
        self.assertEqual(read.call_args.kwargs["frames"], int(uv.MAX_SECONDS * 16000) + 1)

    def test_not_audio_is_refused(self):
        with self.assertRaisesRegex(uv.UploadRefused, "could not be read"):
            uv._decode(b"not audio at all" * 100, "voice.wav")


class Save(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp())

    def test_a_matching_upload_is_kept_with_its_wer(self):
        about = uv.save("ana", "Calm", SCRIPT, recording(), "rec.wav", base=self.base,
                        check=lambda mono, rate, script: uv.check_script(mono, rate, script, hear=hears(SCRIPT)))
        self.assertEqual(about["script_wer"], 0.0)
        stored = json.loads((self.base / "ana" / "calm" / "about.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["script_wer"], 0.0)

    def test_a_mismatch_keeps_nothing(self):
        with self.assertRaises(uv.UploadRefused):
            uv.save("ana", "Calm", SCRIPT, recording(), "rec.wav", base=self.base,
                    check=lambda mono, rate, script: uv.check_script(mono, rate, script, hear=hears("spam spam")))
        self.assertEqual(list((self.base / "ana").glob("*")) if (self.base / "ana").exists() else [], [])

    def test_the_real_check_is_used_by_default(self):
        with mock.patch.object(uv, "check_script", lambda mono, rate, script: 0.123):
            about = uv.save("ana", "Warm", SCRIPT, recording(), "rec.wav", base=self.base)
        self.assertEqual(about["script_wer"], 0.123)


if __name__ == "__main__":
    unittest.main()

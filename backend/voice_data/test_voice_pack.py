"""test_voice_pack.py | The serving voice pack: only validation voices, checked file by file, read by the site.

    python backend/voice_data/test_voice_pack.py

Uses a tiny made-up library and manifest in a temporary folder, so it runs without the voice library (and on CI).
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.voice_data import voice_pack  # noqa: E402
from backend.voice_data.Voice_Importer import Clip, get_clip  # noqa: E402

VOICES = [  # (dataset, speaker, clip id, split, file format)
    ("CREMA-D", "1003", "1003_ITH_NEU_XX", "validation", "PCM_16"),
    ("LibriTTS-R", "84", "84_121550_000173_000000", "validation", "PCM_24"),     # converted to 16-bit
    ("VCTK", "p225", "p225_023_mic1", "validation", "PCM_16"),     # as get_clip writes it out of its parquet
    ("CREMA-D", "1008", "1008_ITH_NEU_XX", "test", "PCM_16"),
]


def tone(path, fmt):
    t = np.arange(16000) / 16000
    samples = (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32")
    sf.write(str(path), samples, 16000, subtype=fmt)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.clips, voices = [], []
        for dataset, speaker, clip_id, split, fmt in VOICES:
            source = self.root / "library" / dataset / f"{clip_id}.wav"
            source.parent.mkdir(parents=True, exist_ok=True)
            tone(source, fmt)
            self.clips.append(Clip(dataset=dataset, clip_id=clip_id, speaker=speaker, text=f"Words of {speaker}.",
                                   source=source, details={"note": "x"}))
            voices.append({"voice": f"{dataset}/{speaker}", "dataset": dataset, "speaker": speaker,
                           "clip_id": clip_id, "split": split, "sha256": "not-known"})
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({
            "name": "voice-split-test", "voices": voices,
            "validation": [v["voice"] for v in voices if v["split"] == "validation"],
            "test": [v["voice"] for v in voices if v["split"] == "test"]}), encoding="utf-8")
        self.pack = self.root / "pack"
        imports = mock.patch("backend.voice_data.Voice_Importer.IMPORT_DIR", self.root / "imported")
        imports.start()
        self.addCleanup(imports.stop)

    def build(self):
        return voice_pack.build(self.pack, clips=self.clips, manifest_path=self.manifest)


class Build(Fixture):
    def test_only_the_validation_voices_go_in(self):
        pack = self.build()
        self.assertEqual(pack["count"], 3)
        self.assertEqual({e["voice"] for e in pack["clips"]}, {"CREMA-D/1003", "LibriTTS-R/84", "VCTK/p225"})
        self.assertFalse((self.pack / "CREMA-D" / "1008").exists())
        self.assertNotIn("1008", (self.pack / "pack.json").read_text(encoding="utf-8"))

    def test_the_pack_is_whole_and_says_where_it_came_from(self):
        pack = self.build()
        self.assertEqual(voice_pack.check(self.pack), [])
        self.assertEqual(len(pack["manifest_sha256"]), 64)
        self.assertEqual(set(pack["licences"]), {"CREMA-D", "LibriTTS-R", "VCTK"})
        self.assertIn("ODbL", (self.pack / "README.md").read_text(encoding="utf-8"))

    def test_every_clip_is_a_16_bit_wav_and_wavs_keep_their_bytes(self):
        self.build()
        for entry in voice_pack.load(self.pack)["clips"]:
            self.assertEqual(sf.info(str(self.pack / entry["audio"])).subtype, "PCM_16")
        self.assertEqual([e["voice"] for e in voice_pack.load(self.pack)["clips"] if e["same_bytes_as_source"]], [])
        original = self.clips[0].source.read_bytes()
        self.assertEqual((self.pack / "CREMA-D" / "1003" / "reference.wav").read_bytes(), original)

    def test_a_missing_clip_stops_the_build(self):
        with self.assertRaises(voice_pack.PackError):
            voice_pack.build(self.pack, clips=self.clips[1:], manifest_path=self.manifest)

    def test_a_rebuild_replaces_the_old_pack(self):
        self.build()
        self.build()
        self.assertEqual(voice_pack.check(self.pack), [])


class Check(Fixture):
    def test_a_changed_or_missing_file_is_found(self):
        self.build()
        (self.pack / "VCTK" / "p225" / "reference.txt").write_text("other words", encoding="utf-8")
        (self.pack / "LibriTTS-R" / "84" / "reference.wav").unlink()
        problems = voice_pack.check(self.pack)
        self.assertEqual(len(problems), 2)
        self.assertTrue(any("changed" in p for p in problems) and any("missing" in p for p in problems))

    def test_no_pack_is_a_problem_not_a_crash(self):
        self.assertEqual(len(voice_pack.check(self.root / "nothing")), 1)

    def test_the_revision_can_be_recorded_without_breaking_the_check(self):
        self.build()
        voice_pack.set_revision(self.pack, "abc1234")
        self.assertEqual(voice_pack.load(self.pack)["revision"], "abc1234")
        self.assertEqual(voice_pack.check(self.pack), [])


class Use(Fixture):
    def test_the_packs_clips_are_used_as_they_are(self):
        self.build()
        clips = voice_pack.read(self.pack)
        self.assertEqual(len(clips), 3)
        for clip in clips:
            audio, text = get_clip(clip)
            self.assertEqual(Path(audio).parent.parent.parent, self.pack)     # nothing written elsewhere
            self.assertTrue(text.startswith("Words of"))
        self.assertFalse((self.root / "imported").exists())

    def test_a_run_finds_its_reference_by_clip_id_in_the_pack(self):
        self.build()
        run = {"reference_audio": "D:\\Voices library\\VCTK\\gone.wav",
               "origin": {"dataset": "VCTK", "speaker": "p225", "clip_id": "p225_023_mic1"}}
        with mock.patch.dict(os.environ, {"VOICE_PACK_DIR": str(self.pack)}):
            self.assertEqual(Path(voice_pack.reference_of(run)), self.pack / "VCTK" / "p225" / "reference.wav")
        with mock.patch.dict(os.environ, {"VOICE_PACK_DIR": ""}):
            self.assertEqual(voice_pack.reference_of(run), run["reference_audio"])
        other = {**run, "origin": {"dataset": "VCTK", "speaker": "p999", "clip_id": "p999_001_mic1"}}
        with mock.patch.dict(os.environ, {"VOICE_PACK_DIR": str(self.pack)}):
            self.assertEqual(voice_pack.reference_of(other), run["reference_audio"])

    def test_the_site_lists_the_packs_voices(self):
        from backend.api import server

        self.build()
        chosen = {f"{d}/{s}": c for d, s, c, split, _ in VOICES if split == "validation"}
        saved = server._library["clips"]
        self.addCleanup(server._library.__setitem__, "clips", saved)
        server._library["clips"] = None
        with mock.patch.dict(os.environ, {"VOICE_PACK_DIR": str(self.pack), "VOICE_LIBRARY_DIR": "Z:/nowhere"}), \
                mock.patch("backend.voice_data.split_voice_data.voice_clips", lambda split: chosen):
            library = server.library()
        self.assertEqual(sum(len(d["clips"]) for d in library.values()), 3)
        self.assertIn("p225_023_mic1", library["VCTK"]["by_id"])


class Upload(Fixture):
    def api(self, private):
        upload = mock.Mock(return_value=SimpleNamespace(oid="f00dfeed" * 5))
        return mock.Mock(repo_info=mock.Mock(return_value=SimpleNamespace(private=private)), upload_folder=upload)

    def test_a_public_dataset_is_refused(self):
        self.build()
        api = self.api(private=False)
        with mock.patch("huggingface_hub.HfApi", return_value=api), self.assertRaises(voice_pack.PackError) as caught:
            voice_pack.upload(self.pack, "Dramatizer/Serving_Voices")
        self.assertIn("PUBLIC", str(caught.exception))
        api.upload_folder.assert_not_called()

    def test_a_broken_pack_is_not_uploaded(self):
        self.build()
        (self.pack / "VCTK" / "p225" / "reference.wav").unlink()
        api = self.api(private=True)
        with mock.patch("huggingface_hub.HfApi", return_value=api), self.assertRaises(voice_pack.PackError):
            voice_pack.upload(self.pack, "Dramatizer/Serving_Voices")
        api.upload_folder.assert_not_called()

    def test_an_upload_records_its_commit_as_the_revision(self):
        self.build()
        with mock.patch("huggingface_hub.HfApi", return_value=self.api(private=True)):
            revision = voice_pack.upload(self.pack, "Dramatizer/Serving_Voices")
        self.assertEqual(revision, "f00dfeed" * 5)
        self.assertEqual(voice_pack.load(self.pack)["revision"], revision)

    def test_a_pull_needs_a_revision(self):
        with self.assertRaises(voice_pack.PackError):
            voice_pack.pull("", "Dramatizer/Serving_Voices", self.root / "pulled")


if __name__ == "__main__":
    unittest.main()

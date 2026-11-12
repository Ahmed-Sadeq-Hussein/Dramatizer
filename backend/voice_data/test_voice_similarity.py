"""Tests of the voice similarity map: which clips, the GPU steps against Resemblyzer's own code (run on the CPU
here), and picking the closest voice with a genome. No GPU needed.

    python -m unittest backend.voice_data.test_voice_similarity
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))    # run as a file by main_backend.py test

from backend.voice_data import voice_similarity as vs  # noqa: E402


def clip(dataset, speaker, n, seconds=4.0, source="a.wav", row_group=None):
    return {"dataset": dataset, "speaker": speaker, "clip_id": f"{speaker}_{n:03d}", "speech_seconds": seconds,
            "source": source, "row_group": row_group, "row": n}


def cpu_embedder():
    """The GPU embedder's own steps, on the CPU, without loading the encoder."""
    import torch
    import torchaudio
    e = vs.GpuEmbedder.__new__(vs.GpuEmbedder)
    e.torch, e.F, e.device = torch, torchaudio.functional, "cpu"
    e.mel = torchaudio.transforms.MelSpectrogram(sample_rate=vs.SR, n_fft=400, hop_length=160, n_mels=40, power=2.0,
                                                 center=True, pad_mode="constant", norm="slaney", mel_scale="slaney")
    return e


def bursts(seconds=5.0, seed=0):
    """Speech-like bursts of a tone with noise, and silences between them."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * vs.SR)) / vs.SR
    on = (np.sin(2 * np.pi * 0.7 * t) > 0).astype(np.float32)
    return (on * (0.3 * np.sin(2 * np.pi * 180 * t) + 0.05 * rng.standard_normal(len(t)))).astype(np.float32)


class Clips(unittest.TestCase):
    def test_over_three_seconds_and_no_test_voices(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "m.json"
            path.write_text(json.dumps({"clips": [clip("VCTK", "p1", 1, 3.0), clip("VCTK", "p1", 2, 3.5),
                                                  clip("VCTK", "p2", 1, 9.0), clip("CREMA-D", "1001", 1, 4.0)]}))
            kept = vs.manifest_clips(3.0, path, locked={"VCTK/p2"})
        self.assertEqual([c["clip_id"] for c in kept], ["p1_002", "1001_001"])

    def test_cap_per_voice_is_fixed_and_keeps_done_clips(self):
        clips = [clip("VCTK", "p1", i) for i in range(10)] + [clip("VCTK", "p2", i) for i in range(3)]
        a, b = vs.cap_per_voice(clips, 4), vs.cap_per_voice(clips, 4)
        self.assertEqual([c["clip_id"] for c in a], [c["clip_id"] for c in b])
        self.assertEqual(sum(c["speaker"] == "p1" for c in a), 4)
        self.assertEqual(sum(c["speaker"] == "p2" for c in a), 3)
        done = {vs.clip_key("VCTK", f"p1_00{i}") for i in range(6)}
        kept = vs.cap_per_voice(clips, 4, keep=done)
        self.assertTrue(done <= {vs.clip_key(c["dataset"], c["clip_id"]) for c in kept})

    def test_each_file_or_row_group_is_read_once(self):
        clips = [clip("VCTK", "p1", 1, source="x.parquet", row_group=2), clip("VCTK", "p1", 2, source="x.parquet", row_group=2),
                 clip("VCTK", "p2", 1, source="x.parquet", row_group=0), clip("LibriTTS-R", "9", 1, source="b.wav")]
        units = vs.work_units(clips)
        self.assertEqual(len(units), 3)
        self.assertEqual(len([u for u in units if len(u) == 2]), 1)


class SameAsResemblyzer(unittest.TestCase):
    def test_trimming_silences(self):
        from resemblyzer.audio import trim_long_silences
        e = cpu_embedder()
        for seed in range(3):
            wav = bursts(4.0 + seed, seed)
            ours = e._trim(e.torch.from_numpy(wav), vs.vad_flags(wav)).numpy()
            theirs = trim_long_silences(wav)
            if len(theirs) >= vs.SR // 10:
                np.testing.assert_array_equal(ours, theirs)

    def test_partial_windows(self):
        from resemblyzer import VoiceEncoder
        e = cpu_embedder()
        for seconds in (0.5, 1.6, 2.0, 3.3, 7.9, 12.0):
            n = int(seconds * vs.SR)
            _, mel_slices = VoiceEncoder.compute_partial_slices(n, 1.3, 0.75)
            parts = e._partials(e.torch.zeros(n))
            self.assertEqual(tuple(parts.shape), (len(mel_slices), 160, 40), seconds)

    def test_mel_spectrogram(self):
        from resemblyzer.audio import wav_to_mel_spectrogram
        e = cpu_embedder()
        wav = bursts(2.0)
        ours = e.mel(e.torch.from_numpy(wav)).T.numpy()
        theirs = wav_to_mel_spectrogram(wav)
        self.assertEqual(ours.shape, theirs.shape)
        np.testing.assert_allclose(ours, theirs, rtol=1e-3, atol=1e-5)


class RenderedLine(unittest.TestCase):
    def test_reference_clip(self):
        clips = [dict(clip("CREMA-D", "1", 1, 9.0), emotion="angry"), dict(clip("CREMA-D", "1", 2, 4.0), emotion="neutral"),
                 dict(clip("CREMA-D", "1", 3, 30.0), emotion="neutral")]
        self.assertEqual(vs.reference_clip(clips, "1_001")["clip_id"], "1_001")          # the split's clip wins
        self.assertEqual(vs.reference_clip(clips)["clip_id"], "1_002")                   # longest neutral of 3-25 s
        self.assertEqual(vs.reference_clip(clips[:1])["clip_id"], "1_001")               # else any emotion

    def test_compare_maps(self):
        def write(folder, voices, emb):
            Path(folder).mkdir(parents=True, exist_ok=True)
            Path(folder, "voices.json").write_text(json.dumps({"voices": [{"voice": v} for v in voices]}))
            emb = np.asarray(emb, dtype=np.float32)
            np.save(Path(folder) / "voice_embeddings.npy", emb)
            np.save(Path(folder) / "similarity.npy", emb @ emb.T)
        with tempfile.TemporaryDirectory() as root:
            write(Path(root, "a"), ["V/1", "V/2", "V/3", "V/4"], np.eye(4))
            near = np.eye(4)[[2, 0, 1]] * 0.99 + 0.01                      # renders of V/3, V/1, V/2, a little off
            write(Path(root, "b"), ["V/3", "V/1", "V/2"], near / np.linalg.norm(near, axis=1, keepdims=True))
            result = vs.compare_maps(Path(root, "a"), Path(root, "b"))
        self.assertEqual(result["voices"], 3)
        self.assertEqual(result["own_voice_first"], 3)
        self.assertGreater(result["render_vs_own_mean"], 0.99)


class Closest(unittest.TestCase):
    def test_suggests_the_genome_of_the_closest_voice_that_has_one(self):
        voices = ["VCTK/p1", "VCTK/p2", "LibriTTS-R/9"]
        emb = np.array([[1, 0, 0], [0.8, 0.6, 0], [0, 0, 1]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "voices.json").write_text(json.dumps({"voices": [{"voice": v} for v in voices]}))
            np.save(Path(folder) / "voice_embeddings.npy", emb)
            np.save(Path(folder) / "similarity.npy", emb @ emb.T)
            fake = mock.Mock()
            fake.embed_files.return_value = np.array([[1, 0, 0]], dtype=np.float32)    # sounds exactly like p1
            genome = {"run": "r", "genome": 3, "name": None, "mos": 4.0, "above_plain": 1.0, "described": ["a pause"],
                      "markup": "x", "raters": ["Ash"], "rated_genomes": 10}
            with mock.patch.object(vs, "genomes_by_voice", return_value={"VCTK/p2": genome, "VCTK/p99": genome}):
                result = vs.closest("me.wav", top=2, folder=folder, embedder=fake)
        self.assertEqual(result["closest"][0], {"voice": "VCTK/p1", "similarity": 1.0, "has_genome": False})
        self.assertEqual(result["suggestion"]["voice"], "VCTK/p2")              # p1 has no genome: the next one
        self.assertEqual(result["suggestion"]["how_close"], "sounds like the same person")
        self.assertEqual(result["suggestion"]["genome"]["genome"], 3)
        self.assertEqual(result["genome_voices_not_in_map"], ["VCTK/p99"])


if __name__ == "__main__":
    unittest.main()

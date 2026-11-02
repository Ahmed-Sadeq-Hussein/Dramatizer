"""Run from Dramatizer/: python -m unittest discover -s tests -v"""
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from genome import Gene, Genome, to_cosyvoice


class GenomeTests(unittest.TestCase):
    def test_editing_does_not_change_parent(self):
        parent = Genome([Gene("emphasis", position=0)])
        child = parent.add(Gene("speed", rate=1.2)).modify(0, position=1)
        self.assertEqual(parent.genes[0].position, 0)
        self.assertEqual(child.genes[0].position, 1)
        self.assertEqual(child.remove(0), Genome([Gene("speed", rate=1.2)]))
        self.assertEqual(parent.replace(0, Gene("breath", position=0)).features, ["breath"])
        with self.assertRaises(IndexError):
            parent.remove(-1)

    def test_json_round_trip(self):
        genome = Genome([Gene("pause", position=0.3, kind="long"), Gene("speed", rate=0.8)])
        self.assertEqual(Genome.from_dict(json.loads(json.dumps(genome.to_dict()))), genome)
        with self.assertRaises(ValueError):
            Genome.from_dict({"schema_version": 2, "genes": []})
        with self.assertRaises(ValueError):
            Gene.from_dict({"feature": "breath", "position": 0, "typo": 1})

    def test_invalid_genes(self):
        for data in [
            {"feature": "made_up", "position": 0},
            {"feature": "emphasis", "position": float("nan")},
            {"feature": "breath", "position": True},
            {"feature": "breath", "position": 1.1},
            {"feature": "breath"},
            {"feature": "speed", "rate": 0},
            {"feature": "speed", "rate": float("inf")},
            {"feature": "speed", "position": 0, "rate": 1},
            {"feature": "pause", "position": 0, "kind": "exactly_1_second"},
        ]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                Gene.from_dict(data)
        with self.assertRaises(ValueError):
            Genome([Gene("speed", rate=1), Gene("speed", rate=1.2)])
        with self.assertRaises(ValueError):
            Genome([Gene("breath", position=0)] * 2)
        with self.assertRaises(ValueError):
            Genome([Gene("breath", position=i / 7) for i in range(7)])

    def test_neutral_preserves_original(self):
        text = "  I can't\tbelieve it!\n"
        self.assertEqual(to_cosyvoice(text).markup, text)
        self.assertEqual(to_cosyvoice(text, Genome()).markup, text)
        self.assertEqual(to_cosyvoice(text).speed, 1)

    def test_positions_transfer_and_keep_punctuation(self):
        genome = Genome([Gene("emphasis", position=1), Gene("breath", position=0)])
        self.assertEqual(to_cosyvoice("Hello, world!", genome).markup,
                         "[breath] Hello, <strong>world</strong>!")
        self.assertEqual(to_cosyvoice("I can't believe it.", genome).markup,
                         "[breath] I can't believe <strong>it</strong>.")
        self.assertEqual(to_cosyvoice("Hi!", genome).markup,
                         "[breath] <strong>Hi</strong>!")
        self.assertEqual(to_cosyvoice("one two three", Genome([Gene("emphasis", position=.5)])).markup,
                         "one <strong>two</strong> three")

    def test_overlapping_edits_and_pause_hints(self):
        genome = Genome([Gene("laughing_speech", position=0), Gene("emphasis", position=0),
                         Gene("pause", position=0, kind="long"), Gene("speed", rate=1.2)])
        delivery = to_cosyvoice("Hello there", genome)
        self.assertEqual(delivery.markup, "<strong><laughter>Hello</laughter></strong>... there")
        self.assertEqual(delivery.text, "Hello there")
        self.assertEqual(delivery.speed, 1.2)
        self.assertEqual(to_cosyvoice("Hello, there", genome).markup,
                         "<strong><laughter>Hello</laughter></strong>, there")

    def test_reject_empty_and_preformatted_input(self):
        genome = Genome([Gene("breath", position=0)])
        for text in ["", "   ", "...", "<strong>Hi</strong>", "[breath] Hi"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                to_cosyvoice(text, genome)


class GenerationTests(unittest.TestCase):
    def test_generation_passes_markup_and_rate_without_changing_reference(self):
        # Exercise the real generate_speech wrapper, replacing only heavyweight
        # audio/model dependencies. No weights, GPU or real audio are required.
        # Keep personal .env settings out of this test, and allow it to run
        # with the standard library alone.
        dotenv = types.ModuleType("dotenv")
        dotenv.load_dotenv = Mock()
        with patch.dict(sys.modules, {"dotenv": dotenv}):
            import model
        torch = types.ModuleType("torch")
        torch.cat = Mock(return_value=Mock(cpu=Mock(return_value="audio")))
        torchaudio = types.ModuleType("torchaudio")
        torchaudio.save = Mock()
        backend = Mock(sample_rate=24000)
        backend.inference_zero_shot.return_value = [{"tts_speech": "chunk"}]
        previous = sys.modules.pop("Voice_model", None)
        try:
            with patch.dict(sys.modules, {"torch": torch, "torchaudio": torchaudio, "model": model}):
                voice = importlib.import_module("Voice_model")
                with tempfile.TemporaryDirectory() as tmp, patch.object(model, "voice_model", return_value=backend):
                    reference = Path(tmp) / "reference.wav"
                    reference.write_bytes(b"fake reference")
                    with patch.object(voice, "OUTPUT_DIR", Path(tmp) / "output"):
                        genome = Genome([Gene("emphasis", position=1), Gene("speed", rate=1.2)])
                        voice.generate_speech(reference, "Reference words.", "Hello world!", genome=genome)
                        args, kwargs = backend.inference_zero_shot.call_args
                        self.assertEqual(args[0], "Hello <strong>world</strong>!")
                        self.assertEqual(args[1], voice.COSYVOICE3_PROMPT + "Reference words.")
                        self.assertEqual(kwargs, {"stream": False, "speed": 1.2, "text_frontend": False})
                        voice.generate_speech(reference, "Reference words.", "Hello world!")
                        self.assertEqual(backend.inference_zero_shot.call_args.kwargs, {"stream": False})
                        self.assertEqual(torchaudio.save.call_count, 2)
                        backend.inference_zero_shot.reset_mock()
                        with self.assertRaises(ValueError):
                            voice.generate_speech(reference, "Reference words.", "[breath] Hello", genome=genome)
                        backend.inference_zero_shot.assert_not_called()
        finally:
            sys.modules.pop("Voice_model", None)
            if previous is not None:
                sys.modules["Voice_model"] = previous


if __name__ == "__main__":
    unittest.main()

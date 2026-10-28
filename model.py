"""model.py | Every model Dramatizer uses, in one place.

Other files never load a model themselves. They ask this file:

    import model

    cosyvoice = model.voice_model()        # speaks in a cloned voice
    whisper_model = model.recogniser()     # turns speech into text (for WER)
    encoder = model.speaker_encoder()      # turns a voice into numbers (similarity)
    squim = model.quality_model()          # estimates how clean a recording is

Why one file: to change a model (a bigger Whisper, another speaker encoder, a
new TTS model) you edit .env, or the one loader below, and nothing else.

Each model is loaded the first time it is asked for and then kept, so asking
again is free. Loading is thread-safe: the web page and a background job can
ask at the same time. Importing this file loads nothing and imports no model
library, so it is fast.

Settings, from .env (a missing one falls back to the default shown):

    DEVICE                cuda or cpu (cuda falls back to cpu if there is no GPU)
    VOICE_MODEL_DIR       the CosyVoice repository checkout (its code)
    COSYVOICE_MODEL_DIR   the CosyVoice 3 weights, Fun-CosyVoice3-0.5B
    WER_MODEL             whisper-base   (whisper-tiny, whisper-small, ...)
    SIMILARITY_MODEL      resemblyzer
    PERQ_MODEL            squim
"""
import logging
import os
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent

# The settings of this machine
load_dotenv(PROJECT_DIR / ".env")

# CosyVoice turns on debug logging for everything, and numba (used inside
# librosa) then prints pages of compiler internals. Keep those quiet.
for noisy in ("numba", "matplotlib", "PIL", "urllib3", "filelock"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

_loaded = {}
_lock = threading.RLock()


def setting(name, default=None):
    """A value from .env (or the environment). Without a default, it must exist."""
    value = os.environ.get(name, default)

    if value is None or not str(value).strip():
        raise KeyError(f"{name} is not set. Add it to .env (see requirements.txt and README.md).")

    return str(value).strip()


def device():
    """cuda if .env asks for it and a GPU is there, otherwise cpu."""
    wanted = setting("DEVICE", "cpu").lower()

    if wanted == "cuda":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    return "cpu"


def _load_once(key, loader):
    """Run loader the first time `key` is asked for, and keep what it returns."""
    with _lock:
        if key not in _loaded:
            _loaded[key] = loader()
        return _loaded[key]


def is_loaded(key):
    """True once the model is in memory. Keys: voice, recogniser, speaker, quality."""
    return key in _loaded


# --------------------------------------------------------------------------- #
# The voice model: CosyVoice 3
# --------------------------------------------------------------------------- #


def voice_model():
    """
    CosyVoice 3 with the Fun-CosyVoice3-0.5B-2512 weights. Takes about a minute
    the first time. Its methods (inference_zero_shot, ...) and sample_rate are
    CosyVoice's own; Voice_model.generate_speech is the way to use them.
    """

    def load():
        code = Path(setting("VOICE_MODEL_DIR"))
        weights = Path(setting("COSYVOICE_MODEL_DIR"))

        if not weights.is_dir():
            raise FileNotFoundError(f"CosyVoice weights not found: {weights} (COSYVOICE_MODEL_DIR in .env)")

        # CosyVoice is not installed with pip: its code is imported from the
        # checkout, together with Matcha-TTS, which it ships as a submodule
        for folder in (code, code / "third_party" / "Matcha-TTS"):
            if str(folder) not in sys.path:
                sys.path.insert(0, str(folder))

        from cosyvoice.cli.cosyvoice import AutoModel

        print("Loading CosyVoice 3...")
        return AutoModel(model_dir=str(weights))

    return _load_once("voice", load)


# --------------------------------------------------------------------------- #
# The speech recogniser: Whisper (for the word error rate)
# --------------------------------------------------------------------------- #


def recogniser():
    """A Whisper model. WER_MODEL=whisper-base picks the size (tiny, base, small, medium, ...)."""

    def load():
        name = setting("WER_MODEL", "whisper-base").lower()
        family, _, size = name.partition("-")

        if family != "whisper" or not size:
            raise ValueError(f"Unknown WER_MODEL {name!r}. Use whisper-<size>, e.g. whisper-base.")

        import whisper

        print(f"Loading Whisper ({size})...")
        return whisper.load_model(size, device=device())

    return _load_once("recogniser", load)


# --------------------------------------------------------------------------- #
# The speaker encoder: Resemblyzer (for the speaker similarity)
# --------------------------------------------------------------------------- #


def speaker_encoder():
    """A speaker encoder: embed_utterance(wav) -> 256 numbers that describe a voice."""

    def load():
        name = setting("SIMILARITY_MODEL", "resemblyzer").lower()

        if name != "resemblyzer":
            raise ValueError(f"Unknown SIMILARITY_MODEL {name!r}. Supported: resemblyzer.")

        from resemblyzer import VoiceEncoder

        print("Loading the speaker encoder (Resemblyzer)...")
        return VoiceEncoder(device=device(), verbose=False)

    return _load_once("speaker", load)


# --------------------------------------------------------------------------- #
# The quality model: SQUIM (for the perceptual quality)
# --------------------------------------------------------------------------- #


def quality_model():
    """
    torchaudio's SQUIM objective model. From one 16 kHz recording it estimates
    three scores with no reference: STOI, PESQ and SI-SDR (see PERQ.py).
    """

    def load():
        name = setting("PERQ_MODEL", "squim").lower()

        if name != "squim":
            raise ValueError(f"Unknown PERQ_MODEL {name!r}. Supported: squim.")

        from torchaudio.pipelines import SQUIM_OBJECTIVE

        print("Loading the quality model (SQUIM)...")
        return SQUIM_OBJECTIVE.get_model().to(device()).eval()

    return _load_once("quality", load)

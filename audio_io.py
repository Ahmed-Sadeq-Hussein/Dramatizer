"""audio_io.py | Read a recording the way the measuring files need it.

One small function, shared so that Similarity.py, WER.py and PERQ.py all read
audio the same way: any wav, flac or mp3, as one channel at the sample rate
the model wants. It does not need ffmpeg.
"""
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def load_mono(path, sample_rate=16000):
    """
    The recording at `path` as a float32 numpy array in [-1, 1]: one channel,
    resampled to `sample_rate`. Raises FileNotFoundError or ValueError with a
    message that names the file.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Recording not found: {path}")

    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)

    if audio.size == 0:
        raise ValueError(f"Recording is empty: {path}")

    if rate != sample_rate:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=sample_rate)

    return np.ascontiguousarray(audio, dtype=np.float32)

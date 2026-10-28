"""Similarity.py | How much does a recording sound like the same person as another?

    from Similarity import speaker_similarity

    score = speaker_similarity("clone.wav", "reference.wav")    # about 0.85

The score is the cosine similarity of the two voices' speaker embeddings (the
model comes from model.py, SIMILARITY_MODEL in .env):

    1.0     identical voice
    ~0.75+  the same person, in the project's rule of thumb (Explorer min_similarity)
    ~0.5    different people
    can fall a little below 0 for very different voices

It needs TWO recordings because similarity compares: the one to judge, and the
voice it should sound like.

    python Similarity.py clone.wav reference.wav
"""
import sys
from pathlib import Path

import numpy as np

import model
from audio_io import load_mono

SAMPLE_RATE = 16000    # what the speaker encoder expects

# Reference voices are compared against many clips: embed each file once
_embeddings = {}


def _embedding(path):
    """The voice in a recording as 256 numbers. Cached by file and modification time."""
    path = Path(path)
    key = (str(path.resolve()), path.stat().st_mtime if path.exists() else None)

    if key not in _embeddings:
        from resemblyzer import preprocess_wav

        # Resemblyzer trims silence and evens out the volume before embedding
        wav = preprocess_wav(load_mono(path, SAMPLE_RATE), source_sr=SAMPLE_RATE)

        if len(wav) == 0:
            raise ValueError(f"No speech found in {path}")

        _embeddings[key] = model.speaker_encoder().embed_utterance(wav)

    return _embeddings[key]


def speaker_similarity(audio, reference_audio):
    """
    How alike the voices in two recordings are, from about 0 (different) to 1
    (identical). `audio` is the recording to judge, `reference_audio` the voice
    it should sound like. Both are paths to wav, flac or mp3 files.
    """
    a = _embedding(audio)
    b = _embedding(reference_audio)

    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python Similarity.py AUDIO REFERENCE_AUDIO")

    print(f"{speaker_similarity(sys.argv[1], sys.argv[2]):.3f}")

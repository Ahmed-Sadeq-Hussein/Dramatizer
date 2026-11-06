"""PERQ.py | How good does a recording sound, by itself?

PERQ here means perceptual quality: an estimate of how clean and listenable a
recording is, from the recording alone (no reference to compare with).

    from backend.metrics.PERQ import perceptual_quality

    score = perceptual_quality("clone.wav")            # about 3.9
    stoi = perceptual_quality("clone.wav", "stoi")

The model is torchaudio's SQUIM (model.py, PERQ_MODEL in .env). It estimates
three standard measures, and you pick one:

    pesq    (default) Perceptual Evaluation of Speech Quality, 1.0 to 4.5.
            About 4+ is clean, below 2.5 is noisy or distorted
    stoi    Short-Time Objective Intelligibility, 0 to 1. How easy the speech is to understand
    si_sdr  Scale-invariant signal-to-distortion ratio, in dB. Higher is cleaner

These are ESTIMATES of how clean the signal is: noise, hiss, clipping and
muffling. They do not judge whether the voice sounds natural or human, or
whether it matches a person. Use Similarity.py and WER.py for those, and
people listening for the rest.

    python -m backend.metrics.PERQ clone.wav [pesq|stoi|si_sdr]
"""
import sys

import torch

from backend import model
from backend.audio_io import load_mono

SAMPLE_RATE = 16000    # what SQUIM expects

# SQUIM returns its three estimates in this order
METRICS = ("stoi", "pesq", "si_sdr")


def perceptual_quality(audio, metric="pesq"):
    """
    The estimated quality of a recording: one number for the chosen metric
    (pesq, stoi or si_sdr, see above). `audio` is a path to a wav, flac or
    mp3 file. Higher is better for all three.
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}. Use one of: {', '.join(METRICS)}.")

    return perceptual_qualities(audio)[metric]


def perceptual_qualities(audio):
    """All three estimates at once: {"stoi": ..., "pesq": ..., "si_sdr": ...}."""
    wave = torch.from_numpy(load_mono(audio, SAMPLE_RATE)).unsqueeze(0)

    with torch.no_grad():
        estimates = model.quality_model()(wave.to(model.device()))

    return {name: float(value.item()) for name, value in zip(METRICS, estimates)}


if __name__ == "__main__":
    if not 2 <= len(sys.argv) <= 3:
        sys.exit("usage: python -m backend.metrics.PERQ AUDIO [pesq|stoi|si_sdr]")

    metric = sys.argv[2] if len(sys.argv) == 3 else "pesq"
    print(f"{metric}: {perceptual_quality(sys.argv[1], metric):.3f}")

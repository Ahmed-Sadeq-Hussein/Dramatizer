"""Clean_Audio_Distortion.py | Remove background noise from a speech recording.

A noisy reference voice gets cloned noise and all: CosyVoice copies the room,
the hiss and the hum along with the voice. Cleaning a reference first gives a
cleaner clone. One function, two tools to choose from:

    clean_audio(audio, method="deepfilter", strength=1.0, output=None)

    method="deepfilter"  DeepFilterNet 3 (MIT / Apache-2.0), a small neural
                         network trained to keep speech and drop everything
                         else: traffic, chatter, fans, music, room echo. The
                         best choice for most recordings. Runs at 48 kHz, on
                         the GPU when there is one. Loaded through model.py.
    method="spectral"    noisereduce (MIT), spectral gating: learns the noise
                         from the quiet parts and turns it down everywhere.
                         No model, very fast, good for steady noise (hiss,
                         hum, fan), weak on noise that changes (voices, music).
                         WARNING: it also changes the voice. Tested on a VCTK
                         clip with hiss and hum (28 Oct 2026), it scored best
                         on estimated quality but dropped speaker similarity
                         to 0.63, below the 0.70 limit in reward.py. Do not
                         use it on reference voices; DeepFilterNet raised
                         similarity on the same clip (0.82 -> 0.85).

    strength  1.0 removes as much as the tool can; lower keeps some background
              and sounds more natural (0.5 = about half). Strong cleaning can
              make a voice sound thin or "underwater"; listen to the result.

What it cannot fix: clipping (a voice recorded too loud, flat-topped waves),
low-bitrate artefacts, or a voice that is quieter than the noise.

Use it from Python:

    from Clean_Audio_Distortion import clean_audio, compare

    path = clean_audio("noisy.wav")                       # output/cleaned/noisy_clean.wav
    path = clean_audio("noisy.wav", method="spectral", strength=0.8)
    compare("noisy.wav", path)                            # estimated quality before and after

or from the command line:

    python Clean_Audio_Distortion.py noisy.wav
    python Clean_Audio_Distortion.py noisy.wav --method spectral --strength 0.8 --compare

The result keeps the sample rate of the original and is saved as a wav.
"""
import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

import model

OUTPUT_DIR = model.PROJECT_DIR / "output" / "cleaned"
METHODS = ("deepfilter", "spectral")

# At strength 1.0 DeepFilterNet removes noise without limit; lower strengths
# cap how many dB it may take away: strength 0.5 -> at most 30 dB, 0.25 -> 15 dB
DEEPFILTER_FULL_DB = 60.0


def _read(audio):
    """The recording as mono float32 samples, and its sample rate."""
    samples, rate = sf.read(str(audio), dtype="float32", always_2d=True)
    return samples.mean(axis=1), rate


def _deepfilter(samples, rate, strength):
    import torch

    # Loads DeepFilterNet (or says how to install it), before its enhance is imported
    network, df_state = model.denoiser()
    from df.enhance import enhance

    model_rate = df_state.sr()

    audio = librosa.resample(samples, orig_sr=rate, target_sr=model_rate) if rate != model_rate else samples
    limit = None if strength >= 1.0 else DEEPFILTER_FULL_DB * strength

    cleaned = enhance(network, df_state, torch.from_numpy(audio).unsqueeze(0), atten_lim_db=limit)
    cleaned = cleaned.squeeze(0).cpu().numpy()

    return librosa.resample(cleaned, orig_sr=model_rate, target_sr=rate) if rate != model_rate else cleaned


def _spectral(samples, rate, strength):
    import noisereduce

    return noisereduce.reduce_noise(y=samples, sr=rate, stationary=True, prop_decrease=strength)


def clean_audio(audio, method="deepfilter", strength=1.0, output=None):
    """
    Remove the background from `audio` and save the result.

    audio     path to the recording
    method    "deepfilter" (neural, best for most noise) or "spectral" (steady noise only)
    strength  0 to 1: how much of the noise to remove
    output    where to save; default output/cleaned/<name>_clean.wav

    Returns the path of the cleaned wav.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Use one of {METHODS}.")
    if not 0.0 < strength <= 1.0:
        raise ValueError("strength must be above 0 and at most 1.")

    audio = Path(audio)
    samples, rate = _read(audio)

    cleaned = (_deepfilter if method == "deepfilter" else _spectral)(samples, rate, strength)

    # Never let the cleaned audio clip
    peak = float(np.abs(cleaned).max()) if len(cleaned) else 0.0
    if peak > 0.99:
        cleaned = cleaned * (0.99 / peak)

    output = Path(output) if output else OUTPUT_DIR / f"{audio.stem}_clean.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), cleaned.astype(np.float32), rate)

    return output


def compare(before, after):
    """Estimated quality of both recordings (PERQ.py, SQUIM): {metric: (before, after)}."""
    from PERQ import METRICS, perceptual_quality

    return {metric: (perceptual_quality(before, metric), perceptual_quality(after, metric)) for metric in METRICS}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove background noise from a speech recording.")
    parser.add_argument("audio", help="the recording to clean")
    parser.add_argument("--method", choices=METHODS, default="deepfilter",
                        help="deepfilter (neural, default) or spectral (steady noise only)")
    parser.add_argument("--strength", type=float, default=1.0, help="0 to 1, how much noise to remove (default 1)")
    parser.add_argument("--output", default=None, help="where to save (default output/cleaned/<name>_clean.wav)")
    parser.add_argument("--compare", action="store_true", help="print the estimated quality before and after")
    args = parser.parse_args()

    path = clean_audio(args.audio, args.method, args.strength, args.output)
    print(f"Saved {path}")

    if args.compare:
        for metric, (old, new) in compare(args.audio, path).items():
            print(f"  {metric:7s} {old:6.2f} -> {new:6.2f}")

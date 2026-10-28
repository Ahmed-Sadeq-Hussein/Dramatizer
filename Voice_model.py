"""Voice_model.py | Speak a line in a cloned voice with CosyVoice 3.

The model is CosyVoice 3 with the Fun-CosyVoice3-0.5B-2512 weights. It is
loaded by model.py (which reads VOICE_MODEL_DIR and COSYVOICE_MODEL_DIR from
.env), so this file only decides what to say and where to save it:

    OUTPUT_DIR            where audio is written (optional, default: output)

Use it from Python:

    from Voice_model import generate_speech

    path = generate_speech(
        "D:/voice data/Voices/ahmed/reference.wav",
        "the words spoken in reference.wav",
        "I can't believe we actually pulled that off.",
    )
    # -> output/ahmed/ahmed_20261025_213512_481.wav

or from the command line:

    python Voice_model.py REFERENCE_AUDIO REFERENCE_TEXT TARGET_TEXT [--name NAME]
"""
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torchaudio

import model

# Where audio is written (settings come from .env, loaded by model.py)
OUTPUT_DIR = model.PROJECT_DIR / model.setting("OUTPUT_DIR", "output")

# CosyVoice 3 expects this system prompt in front of the reference text;
# CosyVoice 2 did not
COSYVOICE3_PROMPT = "You are a helpful assistant.<|endofprompt|>"

# Reference clips are usually called reference.wav, which says nothing about
# the voice, so the folder name is used instead ("refrence" is a common typo)
GENERIC_NAMES = ("reference", "refrence")


def generate_speech(reference_audio, reference_text, target_text, voice_name=None):
    """
    Speak target_text in the voice of reference_audio, using CosyVoice 3
    with the Fun-CosyVoice3-0.5B-2512 weights.

    reference_audio  a clip of the voice to clone (wav, a few seconds long)
    reference_text   the exact words spoken in reference_audio
    target_text      the words the cloned voice should say
    voice_name       the folder the audio is written to. Without it, the name
                     of the reference file is used, or its folder's name when
                     the file is just called "reference"

    Returns the path of the new wav file: OUTPUT_DIR/voice_name/voice_name_TIME.wav
    """
    reference_audio = Path(reference_audio)

    if not reference_audio.is_file():
        raise FileNotFoundError(f"Reference audio not found: {reference_audio}")

    if not reference_text.strip():
        raise ValueError("reference_text is empty: give the words spoken in the reference audio")

    if not target_text.strip():
        raise ValueError("target_text is empty: give the words to speak")

    if voice_name is None:
        voice_name = reference_audio.stem

        if voice_name.lower() in GENERIC_NAMES:
            voice_name = reference_audio.parent.name

    # Loaded by model.py on the first call (about a minute), then reused
    cosyvoice = model.voice_model()

    # Zero-shot cloning: the model hears the reference clip and its words,
    # then says the target text in that voice. Long text comes back in chunks.
    chunks = [
        output["tts_speech"]
        for output in cosyvoice.inference_zero_shot(
            target_text,
            COSYVOICE3_PROMPT + reference_text.strip(),
            str(reference_audio),
            stream=False,
        )
    ]

    audio = torch.cat(chunks, dim=1).cpu()

    folder = OUTPUT_DIR / voice_name
    folder.mkdir(parents=True, exist_ok=True)

    # The time, to the millisecond, keeps every render of a voice apart
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = folder / f"{voice_name}_{stamp}.wav"

    torchaudio.save(str(path), audio, cosyvoice.sample_rate)

    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Speak a line in a cloned voice with CosyVoice 3.")
    parser.add_argument("reference_audio", help="a clip of the voice to clone")
    parser.add_argument("reference_text", help="the words spoken in the clip")
    parser.add_argument("target_text", help="the words to speak")
    parser.add_argument("--name", default=None, help="output folder name (default: from the clip)")
    args = parser.parse_args()

    print(generate_speech(args.reference_audio, args.reference_text, args.target_text, args.name))

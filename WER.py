"""WER.py | How many words of the script does a recording get wrong?

    from WER import word_error_rate

    wer = word_error_rate("clone.wav", "I can't believe we actually pulled that off.")
    # 0.0 = every word was heard correctly

A speech recogniser (Whisper, from model.py, WER_MODEL in .env) writes down
what it hears. The word error rate is the number of words it got wrong
(replaced, missing or extra) divided by the number of words in the script:

    0.0     perfect
    0.10    the project's limit for an accepted clip (Explorer max_wer)
    0.5     half the words wrong
    above 1 is possible, when the recogniser adds many extra words

It needs the script because WER compares what was heard with what should have
been said.

    python WER.py clone.wav "the words that should be heard"
"""
import re
import sys

import jiwer

import model
from audio_io import load_mono

SAMPLE_RATE = 16000    # what Whisper expects


def normalise(text):
    """Lower case, no punctuation (apostrophes stay), single spaces: 'Can't!' -> "can't"."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return " ".join(text.split())


def transcribe(audio, language="en"):
    """What the recogniser hears in a recording: the text, as Whisper writes it."""
    recogniser = model.recogniser()

    result = recogniser.transcribe(
        load_mono(audio, SAMPLE_RATE),
        language=language,
        fp16=model.device() == "cuda",
    )

    return result["text"].strip()


def word_error_rate(audio, text, language="en"):
    """
    The word error rate of a recording against its script: 0.0 is perfect,
    higher is worse. `audio` is a path to a wav, flac or mp3 file, `text` the
    words that should be heard, `language` the language code Whisper listens for.
    """
    expected = normalise(text)

    if not expected:
        raise ValueError("The script has no words to compare with.")

    heard = normalise(transcribe(audio, language))

    # Nothing heard at all: every word is missing
    if not heard:
        return 1.0

    return float(jiwer.wer(expected, heard))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit('usage: python WER.py AUDIO "SCRIPT"')

    print(f"heard: {transcribe(sys.argv[1])}")
    print(f"WER:   {word_error_rate(sys.argv[1], sys.argv[2]):.3f}")

"""WER.py | How many words of the script does a recording get wrong?

    from backend.metrics.WER import word_error_rate

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

    python -m backend.metrics.WER clone.wav "the words that should be heard"
"""
import re
import sys

import jiwer

from backend import model
from backend.audio_io import load_mono

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
    # Checked before listening, which is the slow part
    if not normalise(text):
        raise ValueError("The script has no words to compare with.")

    return wer_of(text, transcribe(audio, language))


# Sounds the recogniser writes down as words, but that say nothing: a laugh, a hum, a hesitation.
# A genome's [laughter] or [mn] often comes out as one of these, and that is not a word gone wrong.
FILLERS = frozenset({
    "ha", "haha", "hahaha", "heh", "hehe", "hah", "ah", "aah", "ahh", "oh", "ooh",
    "uh", "uhh", "um", "umm", "hm", "hmm", "hmmm", "mm", "mmm", "mhm", "er", "erm", "eh",
    "huh", "phew", "ugh", "tsk", "shh", "psst",
})


def content_words(text):
    """The words of a text, normalised, without fillers: "Uh, I can't!" -> ["i", "can't"]."""
    return [word for word in normalise(text).split() if word not in FILLERS]


def word_differences(expected, heard):
    """
    How many words differ between two texts, fillers left out of both:
    {"errors": replaced + missing + extra words, "words": words in `expected`, "wer": errors / words}.
    With no words expected, every heard word is an error and wer is 1.0 if there are any.
    """
    expected, heard = content_words(expected), content_words(heard)

    if not expected:
        return {"errors": len(heard), "words": 0, "wer": 1.0 if heard else 0.0}
    if not heard:
        return {"errors": len(expected), "words": len(expected), "wer": 1.0}

    result = jiwer.process_words(" ".join(expected), " ".join(heard))
    errors = result.substitutions + result.deletions + result.insertions
    return {"errors": errors, "words": len(expected), "wer": errors / len(expected)}


def wer_of(text, heard):
    """The word error rate of what was heard against the script, without listening again."""
    expected = normalise(text)

    if not expected:
        raise ValueError("The script has no words to compare with.")

    heard = normalise(heard)

    # Nothing heard at all: every word is missing
    if not heard:
        return 1.0

    return float(jiwer.wer(expected, heard))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit('usage: python -m backend.metrics.WER AUDIO "SCRIPT"')

    print(f"heard: {transcribe(sys.argv[1])}")
    print(f"WER:   {word_error_rate(sys.argv[1], sys.argv[2]):.3f}")

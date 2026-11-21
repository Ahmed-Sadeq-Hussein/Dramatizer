"""lab.py | The Lab: measure recordings side by side, and against each other.

    from backend.metrics import lab

    result = lab.compare([
        {"label": "Original", "path": "reference.wav", "text": "the words it says"},
        {"label": "Render 1", "path": "render_1.wav", "text": "the line it should say"},
        {"label": "Render 2", "path": "render_2.wav", "text": "the line it should say"},
    ], reference=0)

Each recording gets:

    heard            what the speech recogniser (Whisper, WER.py) heard
    wer              word error rate against its text: 0 is perfect (None without a text)
    words            the word-by-word alignment of the text and what was heard:
                     [{"op": "equal"|"substitute"|"delete"|"insert", "expected": [...], "heard": [...]}]
    errors           replaced + missing + extra words, fillers ("uh", "haha") left out (WER.word_differences)
    similarity       speaker similarity to the reference recording (Similarity.py), when one is chosen
    quality          SQUIM's estimates: pesq (1-4.5), stoi (0-1), si_sdr (dB) (PERQ.py)
    seconds, speech_seconds, words_per_second

and the whole comparison gets `matrix`: the speaker similarity of every pair, so renders
can be compared with each other and not only with a reference.

Every measuring model loads on first use and stays loaded. A file's transcript and
quality are kept in memory by its checksum, so measuring it again costs nothing.
"""
import hashlib
import re
import time
from pathlib import Path

import numpy as np

MAX_RECORDINGS = 8
THRESHOLDS = {"wer": 0.10, "similarity": 0.75, "pesq": 2.5}     # the project's rules of thumb (Explorer, PERQ.py)

_kept = {}          # (sha256, what) -> a measurement, for this process


def _sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class Measurer:
    """The measuring models. Tests give a stand-in with the same methods."""

    def transcribe(self, path):
        from backend.metrics.WER import transcribe
        return transcribe(path)

    def embedding(self, path):
        from backend.metrics.Similarity import speaker_embedding
        return speaker_embedding(path)

    def qualities(self, path):
        from backend.metrics.PERQ import perceptual_qualities
        return perceptual_qualities(path)

    def seconds(self, path):
        from backend.audio_io import load_mono
        return float(len(load_mono(path, 16000))) / 16000

    def speech_seconds(self, path):
        from backend.metrics.reward import speech_seconds
        return speech_seconds(path)

    def models(self):
        from backend.metrics.reward import measuring_models
        return measuring_models()


def _once(path, what, work):
    key = (_sha(path), what)
    if key not in _kept:
        _kept[key] = work(path)
    return _kept[key]


def script_of(text):
    """The words a line should say: gene tags ([laughter], <emphasis>) are not words."""
    from backend.metrics.reward import TAGS
    return re.sub(r"\s+([.,!?;:])", lambda m: m.group(1), " ".join(TAGS.sub(" ", text or "").split()))


def alignment(expected, heard):
    """Word by word, how what was heard lines up with the text, fillers left out of both."""
    import jiwer

    from backend.metrics.WER import content_words

    want, got = content_words(expected), content_words(heard)
    if not want and not got:
        return []
    if not want:
        return [{"op": "insert", "expected": [], "heard": got}]
    if not got:
        return [{"op": "delete", "expected": want, "heard": []}]
    result = jiwer.process_words(" ".join(want), " ".join(got))
    out = []
    for chunk in result.alignments[0]:
        out.append({"op": chunk.type, "expected": want[chunk.ref_start_idx:chunk.ref_end_idx],
                    "heard": got[chunk.hyp_start_idx:chunk.hyp_end_idx]})
    return out


def cosine(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compare(recordings, reference=None, measure=("wer", "similarity", "quality"), measurer=None):
    """
    Measure recordings ([{"label", "path", "text"?}], 1 to MAX_RECORDINGS) and compare them.
    reference: the index of the recording the others should sound like (None: no reference;
    the similarity matrix still compares every pair). measure: which of wer, similarity and
    quality to run (the length and speaking rate are always measured).
    """
    from backend.metrics.WER import word_differences

    measurer = measurer or Measurer()
    if not 1 <= len(recordings) <= MAX_RECORDINGS:
        raise ValueError(f"Compare 1 to {MAX_RECORDINGS} recordings.")
    if reference is not None and not 0 <= reference < len(recordings):
        raise ValueError("The reference must be one of the recordings.")
    measure = set(measure)
    started = time.time()
    out, embeddings = [], []
    for index, rec in enumerate(recordings):
        path = Path(rec["path"])
        script = script_of(rec.get("text"))
        row = {"label": rec["label"], "text": script or None, "reference": index == reference,
               "seconds": round(measurer.seconds(path), 2)}
        speech = measurer.speech_seconds(path)
        row["speech_seconds"] = round(speech, 2)
        if "wer" in measure:
            heard = _once(path, "heard", measurer.transcribe)
            row["heard"] = heard
            if script:
                diff = word_differences(script, heard)
                row.update(wer=round(diff["wer"], 4), errors=diff["errors"], word_count=diff["words"],
                           words=alignment(script, heard))
            else:
                row.update(wer=None, errors=None, word_count=None, words=[])
        words = len(script.split()) if script else (len(row.get("heard", "").split()) or None)
        row["words_per_second"] = round(words / speech, 2) if words and speech > 0 else None
        if "quality" in measure:
            row["quality"] = {k: round(v, 3) for k, v in _once(path, "quality", measurer.qualities).items()}
        if "similarity" in measure:
            embeddings.append(measurer.embedding(path))
        out.append(row)

    matrix = None
    if "similarity" in measure:
        n = len(embeddings)
        matrix = [[round(cosine(embeddings[i], embeddings[j]), 4) for j in range(n)] for i in range(n)]
        for i, row in enumerate(out):
            row["similarity"] = matrix[i][reference] if reference is not None and i != reference else None

    return {"recordings": out, "matrix": matrix, "reference": reference, "measured": sorted(measure),
            "thresholds": THRESHOLDS, "models": measurer.models(), "seconds": round(time.time() - started, 1),
            "made": time.strftime("%Y-%m-%d %H:%M:%S")}

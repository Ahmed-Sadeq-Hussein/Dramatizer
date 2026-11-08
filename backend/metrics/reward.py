"""reward.py | Turn measurements into rewards, with every setting in one place.

Similarity.py, WER.py and PERQ.py MEASURE a recording. This file decides what
those numbers are WORTH to the search: a reward from 0.0 (unacceptable) to 1.0
(fully acceptable). All thresholds live in RewardConfig below; change them
there, nowhere else.

    from backend.metrics.reward import wer_reward, perq_reward, similarity_reward, similarity_matrix_reward

    wer_reward("clip.wav", "the line", speed=1.2, features=["cough"], neutral_audio="neutral.wav")
    perq_reward("clip.wav")
    similarity_reward("reference.wav", "clip.wav")
    similarity_matrix_reward({"a": "a.wav", "b": "b.wav", "c": "c.wav"})
    sanity_check("reference.wav", "baseline.wav", "test.wav")
    clone_metrics("reference.wav", "clone.wav", "the line", reference_text="the reference's words")

Every reward function also accepts details=True and then returns a dict with the
reward and every number behind it, for logs and for debugging a run.

Lessons from earlier work (see PLAN.md, section 7) built into the design:

- WER, quality and similarity are GUARDS: they give 1.0 for anything good
  enough and only fall when something is wrong. Being clearer, cleaner or
  more alike than "good enough" earns nothing, because maximising them
  pushes speech toward slow, flat, neutral delivery (Seed-TTS; arXiv 2509.18531).
- Rewarding low WER made other systems speak slowly. So wer_reward also
  checks the pace: speech much slower than the genome asked for loses reward.
- Some genes make the recogniser stumble for good reasons (a cough, laughing
  while speaking, a faster rate). Those get a small allowance on the WER
  limit instead of being punished for doing what they were asked.
"""
import math
import re
import time
from dataclasses import dataclass, field

import librosa
import numpy as np

from backend.audio_io import load_mono


@dataclass
class RewardConfig:
    """Every threshold and allowance of every reward. Tune values here."""

    # --- WER reward ---------------------------------------------------------
    # Full reward up to this word error rate (on top of the neutral render's own WER)
    wer_free: float = 0.10
    # The reward falls linearly to 0 over this much more WER
    wer_fade: float = 0.30

    # The allowance ("penalty refund") added to the WER limit per gene that
    # can make words harder to recognise. Genes not listed get no allowance.
    feature_allowance: dict = field(default_factory=lambda: {
        "cough": 0.05,
        "laughter": 0.04,
        "laughing_speech": 0.08,   # words spoken while laughing
        "sigh": 0.03,
        "breath": 0.02,
        "quick_breath": 0.02,
        "noise": 0.03,
        "vocalized_noise": 0.03,
        "hissing": 0.03,
        "clucking": 0.02,
        "lipsmack": 0.02,
        "mn": 0.03,                # hesitation hum, may be heard as a word
    })
    # Allowance for speaking faster or slower than normal: this much per 1.0 of
    # rate change, so speed 1.3 adds 0.3 * 0.15 = 0.045
    speed_allowance_per_unit: float = 0.15
    # The total allowance never exceeds this
    max_allowance: float = 0.15

    # Pace check: compare the clip's speaking time (pauses removed) with what
    # the genome's speed asks for. A clip this much slower than expected
    # starts to lose reward, and reaches 0 at pace_zero.
    pace_full: float = 0.85    # 15% slower than asked is still fine
    pace_zero: float = 0.60    # 40% slower than asked earns nothing
    # Silence threshold for finding speech, in dB below the loudest part
    speech_top_db: float = 35.0

    # --- PERQ reward (quality guard) ----------------------------------------
    # Estimated PESQ from SQUIM: full reward from perq_full up, 0 at perq_zero
    perq_metric: str = "pesq"
    perq_full: float = 3.0
    perq_zero: float = 2.0

    # --- Similarity reward (same-speaker guard) -----------------------------
    # Cosine similarity of speaker embeddings: full from sim_full, 0 at sim_zero.
    # Calibrated 28 Oct 2026 with calibrate_similarity.py (Resemblyzer, 40 speakers
    # x 6 clips per dataset, see output/calibration/similarity.json):
    #   different speakers: 95% below 0.70 in both VCTK and CREMA-D
    #   same speaker, VCTK full sentences: mean 0.91, 95% above 0.86
    #   same speaker, VCTK all clips:      mean 0.82, 95% above 0.73
    #   same actor, CREMA-D, another emotion: about 0.08 lower than same emotion
    sim_full: float = 0.80
    sim_zero: float = 0.70
    # With neutral_audio, the clip only has to stay this close to the neutral
    # render's own similarity: emotion lowers the score by itself (about 0.08),
    # and that is not the voice changing. Never below sim_zero.
    sim_emotion_margin: float = 0.08

    # --- Sanity check (a test clip against the baseline of the same line) ----
    # Most words that may be heard differently in the test clip than in the baseline:
    # sanity_wer_limit of the words, but never fewer than sanity_wer_words. On a short
    # line a percentage alone would allow no difference at all (one word of ten is
    # already 10 %). Fillers ("uh", "ha", "hmm") are left out first: a [laughter] or
    # [mn] gene is often written down as one, and that is not a word gone wrong.
    # Calibrated 6 Nov 2026 with calibrate_sanity.py: 10 validation voices (4 VCTK,
    # 3 CREMA-D, 3 LibriTTS-R) x 2 lines x 5 seeds, no genes, 400 pairs of renders:
    #   8-word line:  89 % heard identically, 99 % within 1 word (a flat 5 % passed 89 %)
    #   22-word line: 96 % heard identically, 100 % within 1 word
    sanity_wer_limit: float = 0.10
    sanity_wer_words: int = 1
    # Least pace of the test clip against the baseline, after allowing for the speed
    # the genome asked for. Pace is the baseline's speaking time (pauses removed)
    # divided by the test clip's, so 0.70 means 30 % slower than it should be. This
    # is what stops a clip from buying clear words by dragging them out.
    # Plain renders alone vary a lot (same calibration): lowest 0.63, 5 % below 0.75;
    # 0.70 lets 98 % of them through (96 % on the short line).
    sanity_pace_min: float = 0.70
    # Least speaker similarity between the test clip and the baseline. The speaker
    # encoder is unsteady on little speech, so a short baseline gets a lower limit.
    # Calibrated 6 Nov 2026 (calibrate_sanity.py, plain renders as above), by the
    # baseline's speech time:
    #   6 s or more:  98 % of same-voice pairs at least 0.90 (lowest 0.851)
    #   4.5 to 6 s:   79 % at least 0.90, 100 % at least 0.80
    #   under 4.5 s:  as low as 0.751; 0.80 lets 87 to 100 % through
    # Different voices on the same line (5 Nov 2026, 132 pairs, 3.8 to 7.7 s):
    #   highest 0.889; 6 % at least 0.80, under 1 % at least 0.88.
    # So below sanity_long_seconds the similarity check is weaker: use lines of
    # about 20 words or more for a search where it matters.
    sanity_similarity: float = 0.90
    sanity_similarity_short: float = 0.80
    sanity_long_seconds: float = 6.0


CONFIG = RewardConfig()


def ramp(value, full, zero):
    """1.0 at `full`, 0.0 at `zero`, a straight line between. Works in either direction."""
    if full == zero:
        return 1.0 if value == full else 0.0
    return float(min(1.0, max(0.0, (value - zero) / (full - zero))))


def _feature_names(features):
    """Gene names from a list of names or of genome genes ({"feature": ...})."""
    return [f["feature"] if isinstance(f, dict) else str(f) for f in (features or [])]


def speech_seconds(audio, config=CONFIG):
    """How long a recording actually speaks: its length with silences and pauses removed."""
    samples = load_mono(audio, 16000)
    intervals = librosa.effects.split(samples, top_db=config.speech_top_db)
    return float(sum(end - start for start, end in intervals)) / 16000


# --------------------------------------------------------------------------- #
# Every metric of one clone
# --------------------------------------------------------------------------- #

# Tags such as [laughter] or <strong>...</strong> are spoken as sounds or stress, not words
TAGS = re.compile(r"\[[^\]]*\]|<[^>]*>")


def _words(text):
    return len(TAGS.sub(" ", text or "").split())


def measuring_models():
    """Which models clone_metrics measures with (from .env)."""
    from backend import model

    return {
        "wer": model.setting("WER_MODEL", "whisper-base"),
        "similarity": model.setting("SIMILARITY_MODEL", "resemblyzer"),
        "quality": model.setting("PERQ_MODEL", "squim"),
    }


def clone_metrics(reference_audio, clone_audio, text, reference_text=None, config=CONFIG):
    """
    Every measurement of a cloned clip against the recording it was cloned from.

    reference_audio  the recording the model heard
    clone_audio      the cloned clip
    text             the words the clone should say (tags are left out of the WER)
    reference_text   the words of the reference, for its speaking rate (optional)

    Returns a dict:
      wer, heard               word error rate against `text`, and what Whisper heard
      similarity               speaker similarity of the clone to the reference
      quality, reference_quality   estimated pesq, stoi and si_sdr of each (PERQ.py)
      seconds, speech_seconds  length of the clone, and without its pauses
      words_per_second         speaking rate of the clone (and reference_words_per_second)
      rate_vs_reference        the clone's rate divided by the reference's: below 1 is slower
      rewards                  what wer_reward, similarity_reward and perq_reward give,
                               without a neutral render to compare with
      models, measured         which models measured it, and when

    To measure each clip only once and keep every result, use measurements.measure.
    """
    from backend.metrics.PERQ import perceptual_qualities
    from backend.metrics.Similarity import speaker_similarity
    from backend.metrics.WER import transcribe, wer_of

    script = TAGS.sub(" ", text)
    heard = transcribe(clone_audio)
    wer = wer_of(script, heard)
    similarity = speaker_similarity(clone_audio, reference_audio)
    quality = perceptual_qualities(clone_audio)

    seconds = float(len(load_mono(clone_audio, 16000))) / 16000
    speech = speech_seconds(clone_audio, config)
    rate = _words(text) / speech if speech > 0 else None

    reference_rate = None
    if reference_text:
        reference_speech = speech_seconds(reference_audio, config)
        reference_rate = _words(reference_text) / reference_speech if reference_speech > 0 else None

    return {
        "wer": wer,
        "heard": heard,
        "similarity": similarity,
        "quality": quality,
        "reference_quality": perceptual_qualities(reference_audio),
        "seconds": seconds,
        "speech_seconds": speech,
        "words_per_second": rate,
        "reference_words_per_second": reference_rate,
        "rate_vs_reference": rate / reference_rate if rate and reference_rate else None,
        "rewards": {
            "wer": ramp(wer, config.wer_free, config.wer_free + config.wer_fade),
            "similarity": ramp(similarity, config.sim_full, config.sim_zero),
            "quality": ramp(quality[config.perq_metric], config.perq_full, config.perq_zero),
        },
        "models": measuring_models(),
        "measured": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# --------------------------------------------------------------------------- #
# Sanity check
# --------------------------------------------------------------------------- #


def sanity_words_allowed(words, config=CONFIG):
    """How many words of a line of `words` words may be heard differently: 10 %, but at least 1."""
    return max(config.sanity_wer_words, math.floor(config.sanity_wer_limit * words))


def sanity_check(reference_audio, baseline_audio, test_audio, config=CONFIG, baseline_heard=None, speed=1.0):
    """
    Is a test clip acceptable next to the baseline of the same line in the same voice?

    reference_audio  the recording of the voice being cloned
    baseline_audio   the line made with default settings
    test_audio       the same line, to judge
    baseline_heard   what the recogniser hears in the baseline, if already known
                     (a search judges many clips against one baseline)
    speed            the speaking rate the test clip was asked for (a genome's speed gene),
                     so asking for slower speech is not counted against it

    Acceptable means all three of:
      - words: what is heard in the test clip differs from what is heard in the
        baseline by at most sanity_words_allowed words (fillers left out of both)
      - similarity: its speaker similarity to the baseline is at least sanity_similarity,
        or sanity_similarity_short when the baseline speaks for less than sanity_long_seconds
      - pace: it is not much slower than asked (pace / speed at least sanity_pace_min)

    The similarities to the reference are reported but do not decide the result.

    Returns a dict with "acceptable" (a bool) and every number behind it.
    """
    from backend.metrics.Similarity import speaker_similarity
    from backend.metrics.WER import transcribe, word_differences

    if baseline_heard is None:
        baseline_heard = transcribe(baseline_audio)
    test_heard = transcribe(test_audio)
    words = word_differences(baseline_heard, test_heard)
    allowed = sanity_words_allowed(words["words"], config)
    similarity = speaker_similarity(test_audio, baseline_audio)
    baseline_seconds = speech_seconds(baseline_audio, config)
    pace = baseline_seconds / speech_seconds(test_audio, config)
    pace_vs_asked = pace / speed
    short = baseline_seconds < config.sanity_long_seconds
    similarity_limit = config.sanity_similarity_short if short else config.sanity_similarity

    wer_ok = words["errors"] <= allowed
    similarity_ok = similarity >= similarity_limit
    pace_ok = pace_vs_asked >= config.sanity_pace_min

    return {
        "acceptable": wer_ok and similarity_ok and pace_ok,
        "wer_ok": wer_ok,
        "similarity_ok": similarity_ok,
        "pace_ok": pace_ok,
        "wer": words["wer"],
        "word_errors": words["errors"],
        "words": words["words"],
        "words_allowed": allowed,
        "wer_limit": config.sanity_wer_limit,
        "baseline_heard": baseline_heard,
        "test_heard": test_heard,
        "similarity": similarity,
        "similarity_limit": similarity_limit,
        "short_baseline": short,
        "baseline_speech_seconds": baseline_seconds,
        "pace": pace,
        "speed": speed,
        "pace_vs_asked": pace_vs_asked,
        "pace_limit": config.sanity_pace_min,
        "baseline_to_reference": speaker_similarity(baseline_audio, reference_audio),
        "test_to_reference": speaker_similarity(test_audio, reference_audio),
    }


# --------------------------------------------------------------------------- #
# WER reward
# --------------------------------------------------------------------------- #


def wer_reward(audio, text, speed=1.0, features=(), neutral_audio=None, neutral_wer=None,
               config=CONFIG, details=False):
    """
    Is the clip intelligible enough, without having been slowed down to get there?

    audio          the clip to judge
    text           the words it should say
    speed          the genome's speaking rate (1.0 = normal)
    features       the genome's genes (names, or the gene dicts themselves)
    neutral_audio  the same line in the same voice with no genes. Turns on the
                   pace check, and gives the WER the recogniser already makes
    neutral_wer    that WER, if already known (saves recognising the neutral again)

    Returns a reward from 0 to 1. The allowance for coughs, laughter, speed and
    the like raises the WER limit a little; it never adds reward on its own.
    """
    from backend.metrics.WER import word_error_rate

    names = _feature_names(features)
    wer = word_error_rate(audio, text)

    # Errors the recogniser already makes on the plain line are not the genome's fault
    if neutral_wer is None and neutral_audio is not None:
        neutral_wer = word_error_rate(neutral_audio, text)
    baseline = neutral_wer or 0.0

    allowance = sum(config.feature_allowance.get(name, 0.0) for name in names)
    allowance += config.speed_allowance_per_unit * abs(float(speed) - 1.0)
    allowance = min(allowance, config.max_allowance)

    limit = baseline + config.wer_free + allowance
    intelligibility = ramp(wer, limit, limit + config.wer_fade)

    # Pace check: speech much slower than the genome's speed asks for means the
    # clip bought its clarity by dragging, so the reward drops
    pace = None
    pace_score = 1.0
    if neutral_audio is not None:
        expected = speech_seconds(neutral_audio, config) / float(speed)
        actual = speech_seconds(audio, config)
        pace = expected / actual if actual > 0 else 0.0
        pace_score = 1.0 if pace >= config.pace_full else ramp(pace, config.pace_full, config.pace_zero)

    value = intelligibility * pace_score

    if not details:
        return value

    return {
        "reward": value,
        "wer": wer,
        "neutral_wer": neutral_wer,
        "allowance": allowance,
        "wer_limit": limit,
        "intelligibility": intelligibility,
        "pace": pace,            # 1.0 = the pace the genome asked for, below 1 = slower
        "pace_score": pace_score,
        "features": names,
        "speed": float(speed),
    }


# --------------------------------------------------------------------------- #
# PERQ reward
# --------------------------------------------------------------------------- #


def perq_reward(audio, config=CONFIG, details=False):
    """
    Is the clip clean enough? A guard on the estimated quality (PERQ.py):
    1.0 for anything at or above perq_full, falling to 0 at perq_zero.
    A cleaner clip earns no more, so the search is never pushed toward plain speech.
    """
    from backend.metrics.PERQ import perceptual_quality

    quality = perceptual_quality(audio, config.perq_metric)
    value = ramp(quality, config.perq_full, config.perq_zero)

    if not details:
        return value

    return {"reward": value, config.perq_metric: quality}


# --------------------------------------------------------------------------- #
# Similarity reward
# --------------------------------------------------------------------------- #


def similarity_reward(base_audio, target_audio, neutral_audio=None, config=CONFIG, details=False):
    """
    Does the clip still sound like the person? A guard on speaker similarity:
    1.0 at or above the full limit, falling to 0 at sim_zero.

    base_audio     the voice it should sound like (the reference recording)
    target_audio   the clip to judge
    neutral_audio  optional: the same line in this voice with no genes. Then the
                   full limit becomes "no more than sim_emotion_margin below the
                   neutral render", because expressive delivery lowers the score
                   by itself (see the calibration notes in RewardConfig)
    """
    from backend.metrics.Similarity import speaker_similarity

    similarity = speaker_similarity(target_audio, base_audio)

    full = config.sim_full
    neutral_similarity = None

    if neutral_audio is not None:
        neutral_similarity = speaker_similarity(neutral_audio, base_audio)
        full = min(full, neutral_similarity - config.sim_emotion_margin)

    # The full limit always stays above the point where different speakers start
    full = max(full, config.sim_zero + 0.02)
    value = ramp(similarity, full, config.sim_zero)

    if not details:
        return value

    return {
        "reward": value,
        "similarity": similarity,
        "neutral_similarity": neutral_similarity,
        "full_limit": full,
    }


# --------------------------------------------------------------------------- #
# Similarity matrix reward
# --------------------------------------------------------------------------- #


def similarity_matrix_reward(voices, details=False):
    """
    Rank recordings against each other by how distinct they sound.

    voices   {name: audio path}, or a list of paths (then the names are the paths)

    Every recording is compared with every other. Its reward is its "rift":
    1 - its similarity to the most similar OTHER recording. A recording far from
    all the others gets a high reward; two near-copies both get a low one.
    For later use: pushing genomes of one voice apart from each other.

    Returns {name: reward}, highest first. With details=True also returns the
    full similarity matrix, the order of the names, each one's nearest
    neighbour, and the ranking.
    """
    from backend.metrics.Similarity import speaker_similarity

    if not isinstance(voices, dict):
        voices = {str(path): path for path in voices}

    names = list(voices)

    if len(names) < 2:
        raise ValueError("Ranking needs at least two recordings.")

    n = len(names)
    matrix = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            matrix[i, j] = matrix[j, i] = speaker_similarity(voices[names[i]], voices[names[j]])

    others = matrix - np.eye(n) * 2   # hide each recording's similarity to itself
    nearest = others.argmax(axis=1)

    rewards = {names[i]: float(1.0 - matrix[i, nearest[i]]) for i in range(n)}
    ranking = sorted(rewards, key=rewards.get, reverse=True)
    rewards = {name: rewards[name] for name in ranking}

    if not details:
        return rewards

    return {
        "rewards": rewards,
        "ranking": ranking,
        "names": names,
        "matrix": matrix.round(4).tolist(),
        "nearest": {names[i]: names[nearest[i]] for i in range(n)},
        "mean_similarity": {
            names[i]: float((matrix[i].sum() - 1.0) / (n - 1)) for i in range(n)
        },
    }

"""reward.py | Turn measurements into rewards, with every setting in one place.

Similarity.py, WER.py and PERQ.py MEASURE a recording. This file decides what
those numbers are WORTH to the search: a reward from 0.0 (unacceptable) to 1.0
(fully acceptable). All thresholds live in RewardConfig below; change them
there, nowhere else.

    from reward import wer_reward, perq_reward, similarity_reward, similarity_matrix_reward

    wer_reward("clip.wav", "the line", speed=1.2, features=["cough"], neutral_audio="neutral.wav")
    perq_reward("clip.wav")
    similarity_reward("reference.wav", "clip.wav")
    similarity_matrix_reward({"a": "a.wav", "b": "b.wav", "c": "c.wav"})

Every function also accepts details=True and then returns a dict with the
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
from dataclasses import dataclass, field

import librosa
import numpy as np

from audio_io import load_mono


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
    from WER import word_error_rate

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
    from PERQ import perceptual_quality

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
    from Similarity import speaker_similarity

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
    from Similarity import speaker_similarity

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

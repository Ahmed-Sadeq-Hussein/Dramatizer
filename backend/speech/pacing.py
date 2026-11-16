"""pacing.py | Speak slower or faster without the "slowed-down recording" sound.

CosyVoice's own speed= argument stretches the finished mel spectrogram by linear interpolation
(cosyvoice/cli/model.py, token2wav). Every sound, every transition and every pause is stretched by
the same factor, so a slow line sounds like an edited recording. tts_formatter.synthesize now does
this instead (SPEED_METHOD=natural in .env, the default; SPEED_METHOD=stretch is the old way):

  1. The reference clip is time-scaled to the asked rate first (paced_prompt). CosyVoice 3 carries
     on in the pace of the voice it clones, so the model itself speaks the words slower (or faster):
     they are generated at that pace, not stretched afterwards. The reference is scaled with WSOLA
     (wsola), which moves whole short pieces of the waveform and keeps the pitch and the voice.
  2. The line is rendered at speed 1.0, with no spectrogram stretch.
  3. The pauses between words are scaled by the same factor (scale_pauses): a silence in the line
     grows by 1/rate when slower and shrinks when faster (never below MIN_PAUSE_MS). The words are
     not touched. Timed pauses (pause_ms) are scaled the same way.

How close the result comes to the asked rate depends on how closely the model follows the
reference's pace; the sanity check measures the pace of every render against the plain line
(pace_vs_asked), so a rate that is not reached shows in the numbers.
"""
import hashlib
from pathlib import Path

import numpy as np

FRAME_MS = 40              # WSOLA window
TOLERANCE_MS = 10          # how far WSOLA may shift a window to fit the waveform
PAUSE_FRAME_MS = 10        # frames for finding silences
SILENCE_DB = 35            # a frame this far below the loud frames (95th percentile) is silent
MIN_GAP_MS = 60            # a silence shorter than this is part of speech (stops, short closures)
MIN_PAUSE_MS = 50          # a pause is never shortened below this
FADE_MS = 5


def wsola(x, rate, sr):
    """x (1-D float) played `rate` times as fast (rate 0.85: 1/0.85 as long), pitch kept. Waveform similarity
    overlap-add: each output window is the input window near the expected place that best continues the output."""
    x = np.asarray(x, dtype=np.float64)
    if rate == 1.0 or len(x) == 0:
        return x.copy()
    n = int(sr * FRAME_MS / 1000) // 2 * 2
    hop_out = n // 2
    hop_in = hop_out * rate
    tol = int(sr * TOLERANCE_MS / 1000)
    window = np.hanning(n)
    out_len = int(len(x) / rate) + n
    out = np.zeros(out_len)
    norm = np.zeros(out_len)
    padded = np.concatenate([np.zeros(tol + n), x, np.zeros(tol + 2 * n)])
    offset = tol + n
    previous = 0
    k = 0
    while True:
        out_pos = k * hop_out
        expected = int(round(k * hop_in))
        if expected >= len(x) or out_pos + n > out_len:
            break
        if k == 0:
            start = expected
        else:
            target = padded[offset + previous + hop_out: offset + previous + hop_out + n]   # the natural continuation
            best, best_score = expected, -np.inf
            for shift in range(-tol, tol + 1, 2):
                s = expected + shift
                cand = padded[offset + s: offset + s + n]
                score = float(np.dot(cand, target))
                if score > best_score:
                    best, best_score = s, score
            start = best
        out[out_pos: out_pos + n] += padded[offset + start: offset + start + n] * window
        norm[out_pos: out_pos + n] += window
        previous = start
        k += 1
    out = out / np.maximum(norm, 1e-6)
    return out[:int(len(x) / rate)]


def silences(x, sr):
    """[(start, end)] sample ranges of the pauses inside the speech (not before the first sound or after the last)."""
    x = np.asarray(x, dtype=np.float64)
    hop = max(1, int(sr * PAUSE_FRAME_MS / 1000))
    frames = len(x) // hop
    if frames < 3:
        return []
    rms = np.sqrt(np.mean(x[:frames * hop].reshape(frames, hop) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    quiet = db < np.percentile(db, 95) - SILENCE_DB
    loud = np.flatnonzero(~quiet)
    if len(loud) == 0:
        return []
    first, last = loud[0], loud[-1]
    out, i = [], first
    min_frames = MIN_GAP_MS // PAUSE_FRAME_MS
    while i <= last:
        if quiet[i]:
            j = i
            while j <= last and quiet[j]:
                j += 1
            if j - i >= min_frames:
                out.append((i * hop, j * hop))
            i = j
        else:
            i += 1
    return out


def _fade(n):
    return np.linspace(0.0, 1.0, n, endpoint=False) if n > 0 else np.zeros(0)


def scale_pauses(x, sr, factor):
    """x with every pause inside the speech `factor` times as long (factor > 1 longer). The pause's own quiet
    sound is repeated (or cut) in its middle, with short cross-fades, so the room tone carries on."""
    x = np.asarray(x, dtype=np.float64)
    if factor == 1.0:
        return x.copy()
    fade = int(sr * FADE_MS / 1000)
    pieces, cursor = [], 0
    for start, end in silences(x, sr):
        length = end - start
        new = max(int(length * factor), min(length, int(sr * MIN_PAUSE_MS / 1000)))
        pieces.append(x[cursor:start])
        gap = x[start:end]
        if new > length:                         # repeat the middle of the pause until it is long enough
            middle = gap[length // 4: length - length // 4] if length >= 4 * fade else gap
            extra = new - length
            filler = np.resize(middle, extra + fade) if len(middle) else np.zeros(extra + fade)
            half = length // 2
            a, b = gap[:half], gap[half:]
            joined = _join(_join(a, filler, fade), b, fade)
            pieces.append(joined[:new] if len(joined) >= new else np.pad(joined, (0, new - len(joined))))
        else:                                    # cut from the middle of the pause
            keep_a = new // 2
            keep_b = new - keep_a
            pieces.append(_join(gap[:keep_a + fade], gap[length - keep_b:], fade)[:new])
        cursor = end
    pieces.append(x[cursor:])
    return np.concatenate(pieces) if pieces else x.copy()


def _join(a, b, fade):
    """a then b, cross-faded over `fade` samples (as many as both have)."""
    f = min(fade, len(a), len(b))
    if f == 0:
        return np.concatenate([a, b])
    ramp = _fade(f)
    return np.concatenate([a[:-f], a[-f:] * (1 - ramp) + b[:f] * ramp, b[f:]])


def paced_prompt(prompt_wav, rate, cache_dir):
    """The reference clip time-scaled to `rate` (WSOLA), as a wav in cache_dir (made once per clip and rate)."""
    import soundfile

    prompt_wav = Path(prompt_wav)
    digest = hashlib.sha256(prompt_wav.read_bytes()).hexdigest()[:16]
    out = Path(cache_dir) / f"{prompt_wav.stem}_{digest}_x{rate:.3f}.wav"
    if not out.is_file():
        audio, sr = soundfile.read(str(prompt_wav), dtype="float64", always_2d=True)
        mono = audio.mean(axis=1)
        paced = wsola(mono, rate, sr)
        peak = np.max(np.abs(paced)) if len(paced) else 0
        if peak > 0.999:
            paced = paced * (0.999 / peak)
        out.parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(str(out), paced.astype(np.float32), sr)
    return str(out)

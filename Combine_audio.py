"""Combine_audio.py | Combine two recordings into one.

One function, two ways to combine:

    combine_audio(first, second, mode="join", ...)

    mode="join"     one after the other: first, an optional pause, then second.
                    With crossfade, the end of the first fades into the start
                    of the second instead. Use it to build a longer reference
                    voice from two short clips of the same speaker (references
                    under 3 s clone less well, see Picky_voice.py).
    mode="overlay"  on top of each other: second plays over first, starting
                    `offset` seconds in. Use it to put a sound (laughter, a
                    cough, background) over speech, or two voices at once.

Options:

    gap          join: seconds of silence between the two (default 0)
    crossfade    join: seconds the two overlap while one fades into the other
    offset       overlay: when the second starts, in seconds from the start of the first
    second_db    volume change of the second, in dB (-6 = about half as loud)
    output       where to save; default output/combined/<first>+<second>.wav

The second recording is converted to the first one's sample rate; both become
mono. If the mix would clip, the whole result is turned down to fit.

When both recordings have a script next to them (a .txt with the same name, or
reference.txt in the same folder), "join" also writes the combined script next
to the result, so the pair can be used as a reference in Voice_model.py.

    from Combine_audio import combine_audio

    combine_audio("a.wav", "b.wav")                                  # a then b
    combine_audio("a.wav", "b.wav", gap=0.4)                         # with a pause
    combine_audio("speech.wav", "laugh.wav", mode="overlay", offset=1.2, second_db=-6)

    python Combine_audio.py a.wav b.wav --gap 0.4
    python Combine_audio.py speech.wav laugh.wav --mode overlay --offset 1.2 --second-db -6
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf

import model
from audio_io import load_mono

OUTPUT_DIR = model.PROJECT_DIR / "output" / "combined"
MODES = ("join", "overlay")


def _script(audio):
    """The script of a recording: <name>.txt beside it, or reference.txt in its folder."""
    for candidate in (audio.with_suffix(".txt"), audio.parent / "reference.txt"):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8").strip()
    return None


def combine_audio(first, second, mode="join", gap=0.0, crossfade=0.0, offset=0.0, second_db=0.0, output=None):
    """
    Combine two recordings and save the result (see the top of this file).
    Returns the path of the new wav.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}. Use one of {MODES}.")
    if min(gap, crossfade, offset) < 0:
        raise ValueError("gap, crossfade and offset cannot be negative.")
    if gap and crossfade:
        raise ValueError("Use either a gap or a crossfade, not both.")

    first, second = Path(first), Path(second)
    rate = sf.info(str(first)).samplerate

    a = load_mono(first, rate)
    b = load_mono(second, rate) * (10 ** (second_db / 20))

    if mode == "join":
        fade = min(int(crossfade * rate), len(a), len(b))

        if fade:
            # Equal-power fade: the loudness stays even through the overlap
            t = np.linspace(0, np.pi / 2, fade, dtype=np.float32)
            middle = a[-fade:] * np.cos(t) + b[:fade] * np.sin(t)
            combined = np.concatenate([a[:-fade], middle, b[fade:]])
        else:
            silence = np.zeros(int(gap * rate), dtype=np.float32)
            combined = np.concatenate([a, silence, b])

    else:
        start = int(offset * rate)
        combined = np.zeros(max(len(a), start + len(b)), dtype=np.float32)
        combined[:len(a)] += a
        combined[start:start + len(b)] += b

    # Turn the whole result down if it would clip
    peak = float(np.abs(combined).max()) if len(combined) else 0.0
    if peak > 0.99:
        combined *= 0.99 / peak

    output = Path(output) if output else OUTPUT_DIR / f"{first.stem}+{second.stem}.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), combined, rate)

    # The joined script, so the result can be a reference voice
    if mode == "join":
        scripts = [_script(first), _script(second)]
        if all(scripts):
            output.with_suffix(".txt").write_text(" ".join(scripts), encoding="utf-8")

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine two recordings into one.")
    parser.add_argument("first")
    parser.add_argument("second")
    parser.add_argument("--mode", choices=MODES, default="join", help="join (one after the other) or overlay (on top)")
    parser.add_argument("--gap", type=float, default=0.0, help="join: seconds of silence between them")
    parser.add_argument("--crossfade", type=float, default=0.0, help="join: seconds of fade from one into the other")
    parser.add_argument("--offset", type=float, default=0.0, help="overlay: when the second starts, in seconds")
    parser.add_argument("--second-db", type=float, default=0.0, help="volume change of the second, in dB")
    parser.add_argument("--output", default=None, help="where to save (default output/combined/<first>+<second>.wav)")
    args = parser.parse_args()

    path = combine_audio(args.first, args.second, args.mode, args.gap, args.crossfade,
                         args.offset, args.second_db, args.output)
    print(f"Saved {path} ({sf.info(str(path)).duration:.2f} s)")

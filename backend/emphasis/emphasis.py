"""emphasis.py | How much should each word be stressed? A small LLM decides, from the line and its context.

Every word gets a value from 0.0 to 1.0:

    0.0  unimportant (filler, words already said): spoken quickly
    0.5  normal
    1.0  the key word (contrast, correction, new information, negation,
         strong feeling): a pause before it, <strong> around it, spoken slowly

Why an LLM: which word to stress depends on meaning in context, not on word
class. "No, I took the BLUE car" answers "the red car?", but "No, I TOOK the
blue car" answers "did you sell it?". Older prominence predictors (BERT, about
83% on the Helsinki Prosody Corpus) read one sentence at a time; an LLM can
read the conversation before it.

    from backend.emphasis.emphasis import score_emphasis, speak_with_emphasis

    values = score_emphasis("No, I took the blue car.", context='A: "Did you take the red car?"')
    # [("No,", 0.6), ("I", 0.2), ("took", 0.3), ("the", 0.1), ("blue", 1.0), ("car.", 0.3)]

    path = speak_with_emphasis("reference.wav", "the reference script", "No, I took the blue car.", values)

The LLM is loaded by model.py (EMPHASIS_MODEL and LLM_LIBRARY_DIR in .env).
Speaking needs CosyVoice 3 and Whisper; scoring only needs the LLM, so the
two halves can run in different environments.

How the values become speech: CosyVoice speaks a whole line at one speed and
cannot change it within a line, so the line is rendered once (with <strong>
on the top words) and the speed and pauses are then applied to the audio,
word by word, using Whisper's word timings.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from backend import model

# --------------------------------------------------------------------------- #
# What a value means for the speech (change the feel of emphasis here)
# --------------------------------------------------------------------------- #

FASTEST = 1.25          # value 0.0: words play 25% faster
SLOWEST = 0.75          # value 1.0: words play at 75% speed (a third longer)
NEUTRAL = 0.5           # values near this are left as they are
DEAD_ZONE = 0.1         # |value - 0.5| below this: no speed change at all
PAUSE_FROM = 0.75       # values above this get a pause before the word...
MAX_PAUSE_MS = 350      # ...growing to this at 1.0
STRONG_FROM = 0.8       # values at or above this also get <strong> in the text

# --------------------------------------------------------------------------- #
# Asking the LLM
# --------------------------------------------------------------------------- #

INSTRUCTIONS = """You mark word emphasis for a speech synthesizer.
Give every numbered word of the LINE a value from 0.0 to 1.0:
  0.0-0.2  unimportant: function words, or words already said in the context (spoken quickly)
  0.3-0.6  normal content
  0.7-0.8  clearly stressed
  0.9-1.0  the key word: contrast, correction, new information, negation or strong feeling (spoken slowly, after a pause)
Use the CONTEXT to decide what is new, contrasted or corrected. Usually only one word is 0.9 or more, rarely more than two words are above 0.7.
Answer with a JSON list of numbers only, one per word, in order. Nothing else.

Example
CONTEXT:
A: "Did you paint the kitchen?"
LINE:
1 Actually,  2 we  3 painted  4 the  5 bathroom  6 green  7 instead.
Answer: [0.5, 0.2, 0.3, 0.1, 1.0, 0.3, 0.6]"""

# Generation length: about 4 tokens per number, plus a margin
TOKENS_PER_WORD = 6
# With thinking on, room for the reasoning before the answer
THINKING_TOKENS = 1536


def _prompt(line, context):
    parts = []
    if context:
        parts.append(f"CONTEXT:\n{context.strip()}")
    numbered = "  ".join(f"{i} {word}" for i, word in enumerate(line.split(), start=1))
    parts.append(f"LINE:\n{numbered}")
    return "\n".join(parts)


def _normal(word):
    return re.sub(r"[^\w']", "", word.lower())


def _numbers(answer):
    """The values in an answer, whatever shape the LLM chose: [0.6, ...], [["No", 0.6], ...] or [{"value": 0.6}, ...]."""
    match = re.search(r"\[.*\]", answer, re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            out = []
            for entry in data:
                if isinstance(entry, (int, float)):
                    out.append(float(entry))
                elif isinstance(entry, list) and entry and isinstance(entry[-1], (int, float)):
                    out.append(float(entry[-1]))
                elif isinstance(entry, dict):
                    out.append(float(next(v for v in entry.values() if isinstance(v, (int, float)))))
                else:
                    raise ValueError
            return out
        except (ValueError, TypeError, StopIteration):
            pass
    # Broken JSON: take the numbers between 0 and 1 in order
    return [float(x) for x in re.findall(r"(?<![\d.])(?:0?\.\d+|[01](?:\.0+)?)(?![\d.])", match.group(0) if match else answer)]


def parse_values(answer, line):
    """
    Read the LLM's answer into one value per word of the line.
    Returns (values, ok): ok is False when the answer did not have exactly one
    number per word; then missing words get 0.5 and extra numbers are dropped.
    """
    words = line.split()
    numbers = _numbers(answer)
    ok = len(numbers) == len(words)
    numbers = (numbers + [NEUTRAL] * len(words))[:len(words)]
    return [(word, min(max(value, 0.0), 1.0)) for word, value in zip(words, numbers)], ok


def ask(line, context=None, name=None, sample=False, think=False):
    """
    One LLM answer for one line: (answer text, generated tokens).

    think  let the model reason before answering (Qwen3's thinking mode; other
           models ignore it). Slower, often better at context. The reasoning
           is removed from the returned answer.
    """
    import torch

    tokenizer, llm = model.emphasis_model(name)
    messages = [{"role": "system", "content": INSTRUCTIONS},
                {"role": "user", "content": _prompt(line, context)}]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    # model.device(), not llm.device: with parts offloaded to the CPU, llm.device names those
    inputs = tokenizer(text, return_tensors="pt").to(model.device())

    with torch.inference_mode():
        output = llm.generate(
            **inputs,
            max_new_tokens=THINKING_TOKENS if think else TOKENS_PER_WORD * len(line.split()) + 40,
            do_sample=sample,
            temperature=0.7 if sample else None,
            top_p=0.95 if sample else None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    new = output[0, inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new, skip_special_tokens=True)
    # Only what comes after the reasoning is the answer
    return answer.split("</think>")[-1], len(new)


def score_emphasis(line, context=None, name=None, samples=1, think=False):
    """
    [(word, value 0-1)] for every word of the line (split on spaces).

    context  what came before: earlier lines, who is speaking, the scene
    name     which LLM in LLM_LIBRARY_DIR (default EMPHASIS_MODEL)
    samples  1: one deterministic answer. More: that many sampled answers,
             averaged, which smooths out the LLM's randomness
    think    let the model reason first (slower, often better with context)
    """
    if samples <= 1:
        values, _ = parse_values(ask(line, context, name, think=think)[0], line)
        return values

    runs = [parse_values(ask(line, context, name, sample=True, think=think)[0], line)[0] for _ in range(samples)]
    return [(word, round(sum(run[i][1] for run in runs) / samples, 3)) for i, word in enumerate(line.split())]


# --------------------------------------------------------------------------- #
# Turning values into speech
# --------------------------------------------------------------------------- #


def rate(value):
    """Playback rate for a word: FASTEST at 0, 1.0 near 0.5, SLOWEST at 1."""
    distance = value - NEUTRAL
    if abs(distance) < DEAD_ZONE:
        return 1.0
    if distance < 0:
        return 1.0 + (FASTEST - 1.0) * (-distance - DEAD_ZONE) / (NEUTRAL - DEAD_ZONE)
    return 1.0 - (1.0 - SLOWEST) * (distance - DEAD_ZONE) / (NEUTRAL - DEAD_ZONE)


def pause_ms(value):
    """Silence before a word: none up to PAUSE_FROM, then up to MAX_PAUSE_MS at 1.0."""
    return 0 if value <= PAUSE_FROM else round(MAX_PAUSE_MS * (value - PAUSE_FROM) / (1.0 - PAUSE_FROM))


def emphasis_operations(values):
    """tts_formatter operations for the text side: <strong> on every word at or above STRONG_FROM."""
    seen, operations = {}, []
    for word, value in values:
        seen[word] = seen.get(word, 0) + 1
        if value >= STRONG_FROM and _normal(word):
            operations.append({"feature": "emphasis", "kwargs": {"target": re.sub(r"[.,!?;:]+$", "", word),
                                                                 "occurrence": seen[word]}})
    return operations


def word_times(audio, text):
    """Whisper's [(start, end)] in seconds for each word of the rendered line, matched in order."""
    from backend.audio_io import load_mono

    samples = load_mono(audio, 16000)
    result = model.recogniser().transcribe(samples, language="en", word_timestamps=True,
                                           fp16=model.device() == "cuda")
    heard = [(w["word"], w["start"], w["end"]) for seg in result["segments"] for w in seg.get("words", [])]

    times, j = [], 0
    for word in text.split():
        k = next((k for k in range(j, min(j + 3, len(heard))) if _normal(heard[k][0]) == _normal(word)), None)
        if k is None:
            times.append(None)
        else:
            times.append((heard[k][1], heard[k][2]))
            j = k + 1
    return times


def apply_emphasis(audio, values, output):
    """
    Re-time a rendered line: every word is sped up or slowed down by its value,
    and the key words get a pause before them. Words Whisper could not find
    are left as they are. Returns the path of the new wav.
    """
    import librosa
    import numpy as np
    import soundfile as sf

    samples, sr = sf.read(str(audio), dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    times = word_times(audio, " ".join(word for word, _ in values))

    pieces, position = [], 0
    for (word, value), span in zip(values, times):
        if span is None:
            continue
        start, end = int(span[0] * sr), int(span[1] * sr)
        if start < position or end <= start:
            continue
        pieces.append(samples[position:start])                       # the gap before the word, unchanged
        if pause_ms(value):
            pieces.append(np.zeros(int(sr * pause_ms(value) / 1000), dtype=np.float32))
        segment = samples[start:end]
        r = rate(value)
        if r != 1.0 and len(segment) > 2048:
            segment = librosa.effects.time_stretch(segment, rate=r, n_fft=1024)
        pieces.append(segment)
        position = end
    pieces.append(samples[position:])

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), np.concatenate(pieces), sr)
    return output


def speak_with_emphasis(reference_audio, reference_text, line, values, voice_name="emphasis"):
    """
    Speak `line` in the reference voice with these emphasis values:
    <strong> on the top words (CosyVoice 3), then speed and pauses per word.
    Saves output/<voice_name>/<voice_name>_<time>.wav (and _plain.wav, the
    render before re-timing) and returns the emphasised path.
    """
    import torchaudio

    from backend.speech.tts_formatter import process_tts_string, synthesize

    plan = process_tts_string(line, emphasis_operations(values))
    cosyvoice = model.voice_model()
    audio = synthesize(cosyvoice, plan, str(reference_audio), reference_text)

    folder = model.PROJECT_DIR / model.setting("OUTPUT_DIR", "output") / voice_name
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plain = folder / f"{voice_name}_{stamp}_plain.wav"
    torchaudio.save(str(plain), audio, cosyvoice.sample_rate)

    return apply_emphasis(plain, values, folder / f"{voice_name}_{stamp}.wav")

"""tts_formatter.py | The paralinguistic features: functions that edit a line's text to add them.

Each feature is a function that rewrites the text (insert [laughter] after a
word, wrap a word in <strong>, add a pause...) or sets something on the plan
(speed, an instruction). process_tts_string applies a list of them, and
synthesize speaks the result with CosyVoice 3.

    from backend.speech.tts_formatter import process_tts_string, synthesize

    plan = process_tts_string("I can't believe we actually pulled that off.", [
        {"feature": "emphasis", "kwargs": {"target": "actually"}},
        {"feature": "laughter", "kwargs": {"target": "off."}},
        {"feature": "pause", "kwargs": {"target": "believe", "pause_ms": 300}},
    ])
    audio = synthesize(model.voice_model(), plan, "reference.wav", "the reference script")

This is the original Dramatizer's tts_formatter.py (written for CosyVoice 2),
ported to CosyVoice 3. Every feature of the original is kept. The comments
mark what is different:

    NEW       an addition to the original
    CHANGED   works differently than in the original, because CosyVoice 3 does

The additions, in short:
    - pronunciation   say a word from CMU phonemes, e.g. [R][EH1][D]: a new
                      CosyVoice 3 ability that can also move the stress
                      inside a word (finer than emphasis)
    - emotion         happy / sad / angry, one of CosyVoice 3's trained instructions
    - volume          loud / soft, also a trained instruction
    - dialect         one of the 17 Chinese dialects CosyVoice 3 was trained on
    - INSTRUCT_PRESETS  the fixed list of instructions CosyVoice 3 knows
    - phoneme tokens in raw input text are removed like other markup
The changes, in short:
    - every prompt carries "You are a helpful assistant.<|endofprompt|>"
      (CosyVoice 3 stops with an error without it)
    - speed by instruction only has "fastest" and "slowest" in CosyVoice 3
    - no event is marked verified yet: they were confirmed by listening on
      CosyVoice 2 only, and must be listened to again on CosyVoice 3

Only the text logic is imported at the top, so this file loads without torch.
"""
from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass, field
from typing import Union

logger = logging.getLogger(__name__)


# CosyVoice point-event tokens, keyed by feature name. Each key is also
# registered in the Option Table below. CosyVoice 3 has the same 12 as
# CosyVoice 2 (CosyVoice3Tokenizer in cosyvoice/tokenizer/tokenizer.py).
EVENT_TAGS = {
    "breath": "[breath]",
    "quick_breath": "[quick_breath]",
    "noise": "[noise]",
    "laughter": "[laughter]",
    "cough": "[cough]",
    "clucking": "[clucking]",  # tongue click
    "accent": "[accent]",
    "hissing": "[hissing]",
    "sigh": "[sigh]",
    "vocalized_noise": "[vocalized-noise]",
    "lipsmack": "[lipsmack]",
    "mn": "[mn]",  # hesitation hum
}

# Alternate spellings accepted by add_micro_expression
EVENT_ALIASES = {
    "laugh": "laughter",
    "vocalized-noise": "vocalized_noise",
}

# CHANGED: features confirmed by listening ON COSYVOICE 3. The original had
# {"laughter", "breath"}, confirmed on CosyVoice 2. CosyVoice notes that these
# tokens start out random, so being in the vocabulary proves nothing: add a
# feature here only after listening to it on CosyVoice 3.
VERIFIED_FEATURES: set[str] = set()

SPAN_TAGS = ("strong", "laughter")

# Soft pauses are punctuation only: a hint to the model, length not controllable
SOFT_PAUSES = {
    "dramatic": "...",
    "realization": " —",  # Em-dash for sudden stop
}

# Inclusive ranges for numeric kwargs; process_tts_string clamps to these
BOUNDS = {
    "rate": (0.5, 2.0),
    "pause_ms": (0, 5000),
}

END_OF_PROMPT = "<|endofprompt|>"

# NEW: CosyVoice 3 expects this system prompt before every reference script and
# instruction, ended by END_OF_PROMPT; without the token it stops with an error
SYSTEM_PROMPT = "You are a helpful assistant."

INSTRUCT_MAX_CHARS = 200

# NEW: the instructions CosyVoice 3 was trained on (cosyvoice/utils/common.py).
# Other text may be accepted but ignored, so features use these keys.
INSTRUCT_PRESETS = {
    # Chinese dialects
    "cantonese": "请用广东话表达。",
    "dongbei": "请用东北话表达。",
    "gansu": "请用甘肃话表达。",
    "guizhou": "请用贵州话表达。",
    "henan": "请用河南话表达。",
    "hubei": "请用湖北话表达。",
    "hunan": "请用湖南话表达。",
    "jiangxi": "请用江西话表达。",
    "minnan": "请用闽南话表达。",
    "ningxia": "请用宁夏话表达。",
    "shanxi": "请用山西话表达。",
    "shaanxi": "请用陕西话表达。",
    "shandong": "请用山东话表达。",
    "shanghai": "请用上海话表达。",
    "sichuan": "请用四川话表达。",
    "tianjin": "请用天津话表达。",
    "yunnan": "请用云南话表达。",
    # Volume
    "loud": "Please say a sentence as loudly as possible.",
    "soft": "Please say a sentence in a very soft voice.",
    # Speed
    "slowest": "请用尽可能慢地语速说一句话。",
    "fastest": "请用尽可能快地语速说一句话。",
    # Emotion
    "happy": "请非常开心地说一句话。",
    "sad": "请非常伤心地说一句话。",
    "angry": "请非常生气地说一句话。",
    # Character voices
    "peppa": "我想体验一下小猪佩奇风格，可以吗？",
    "robot": "你可以尝试用机器人的方式解答吗？",
}
EMOTIONS = ("happy", "sad", "angry")                                   # NEW
VOLUMES = ("loud", "soft")                                             # NEW
DIALECTS = tuple(INSTRUCT_PRESETS)[:17]                                # NEW

# CHANGED: the original had four English speed instructions ("Speaking fast"...).
# CosyVoice 3 was trained on two, both extremes, so they are preset keys now.
SPEED_INSTRUCTIONS = ("fastest", "slowest")

# NEW: the English CMU phonemes CosyVoice 3 can be given instead of a word's
# letters ("pronunciation inpainting"). Vowels carry a stress digit:
# 0 unstressed, 1 primary stress, 2 secondary stress.
_CMU_VOWELS = ("AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW")
_CMU_CONSONANTS = ("B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S",
                   "SH", "T", "TH", "V", "W", "Y", "Z", "ZH")
CMU_PHONES = {*_CMU_CONSONANTS, *_CMU_VOWELS, *(f"{v}{s}" for v in _CMU_VOWELS for s in "012")}

# With text_frontend=False the model gets each text segment unsplit; the
# frontend would have cut it into pieces of roughly this size
FRONTEND_OFF_MAX_WORDS = 80

_PUNCT = ".!?,"

_EVENT = "|".join(re.escape(tag) for tag in EVENT_TAGS.values())
_SPAN = "|".join(SPAN_TAGS)
_PAUSE = r"⟦pause:(\d+)⟧"
# NEW: one phoneme token, e.g. [EH1]. Longest first, so [NG] is not read as [N]
_PHONE = "|".join(re.escape(f"[{p}]") for p in sorted(CMU_PHONES, key=len, reverse=True))
# Any other SSML/HTML-style tag: opening, closing or self-closing. A bare "<"
# or ">" (as in "3 < 5") is not followed by a tag name and does not match.
_OTHER_TAG = r"</?[A-Za-z][^<>]*>"
# Han and kana are written without spaces, so they take no word boundary
_CJK = r"぀-ヿ㐀-䶿一-鿿豈-﫿"
_SPACED_WORD_CHAR = rf"[^\W{_CJK}]"

PAUSE_MARKER_RE = re.compile(_PAUSE)
_EVENT_RE = re.compile(_EVENT)
_PHONE_RE = re.compile(_PHONE)                                                    # NEW
_CJK_RE = re.compile(rf"[{_CJK}]")
# Same test as contains_chinese in the CosyVoice frontend, which picks its Chinese path
_CHINESE_RE = re.compile(r"[一-鿿]")
# CHANGED: phoneme tokens are reserved too
_RESERVED_RE = re.compile(rf" ?(?:{_EVENT})|</?(?:{_SPAN})>|{_PAUSE}|(?:{_PHONE})+|<\|[^|]*\|>|<\||\|>|{_OTHER_TAG}")
_SPAN_TAG_RE = re.compile(rf"<(/?)({_SPAN})>")
# CHANGED: a word written as phonemes counts as markup when searching for words
_MARKUP_RE = re.compile(rf" ?(?:{_EVENT})|</?(?:{_SPAN})>|{_PAUSE}|(?:{_PHONE})+")
_OPENERS_BEFORE_RE = re.compile(rf"(?:<(?:{_SPAN})>)*$")
_CLOSERS_RE = re.compile(rf"(?:</(?:{_SPAN})>)*")
_CLOSERS_AND_EVENTS_RE = re.compile(rf"(?: ?(?:{_EVENT})|</(?:{_SPAN})>)*")


class TargetNotFoundError(ValueError):
    """Raised when an operation's target is not present in the text."""


@dataclass
class TTSPlan:
    text: str                 # markup text, may contain pause markers
    speed: float = 1.0        # passed to inference as speed=
    instruct: str | None = None   # optional instruction: a key of INSTRUCT_PRESETS, or free text
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TextSegment:
    text: str


@dataclass(frozen=True)
class SilenceSegment:
    ms: int


Segment = Union[TextSegment, SilenceSegment]


def _word_pattern(target_word: str) -> str:
    """Builds a whole-word pattern for the target.

    A plain \\b fails when the target starts or ends with punctuation
    (e.g. "off."), so each edge only gets a boundary check if it is a word
    character. CJK characters are exempt on both sides of the check: they are
    written without spaces, so a target may sit anywhere in a CJK sentence.
    """
    prefix = rf'(?<!{_SPACED_WORD_CHAR})' if re.match(_SPACED_WORD_CHAR, target_word) else ''
    suffix = rf'(?!{_SPACED_WORD_CHAR})' if re.search(rf'{_SPACED_WORD_CHAR}$', target_word) else ''
    return rf'{prefix}({re.escape(target_word)}){suffix}'


def _strip_markup(text: str) -> tuple[str, list[int]]:
    """Returns the text without tags, pause markers or phonemes, and each kept character's original index."""
    index = []
    pos = 0
    for m in _MARKUP_RE.finditer(text):
        index.extend(range(pos, m.start()))
        pos = m.end()
    index.extend(range(pos, len(text)))
    return ''.join(text[i] for i in index), index


def _locate(text: str, target: str, trailing_punct: bool = False, occurrence: int = 1) -> tuple[int, int, int | None]:
    """Finds the occurrence-th match (1-based) of target in the tag-stripped text.

    Returns (start, end, punct) as offsets into the original text, where punct
    is the offset of the match's trailing punctuation, or None if it has none.
    With trailing_punct, punctuation following the target is part of the match.
    """
    if not target:
        raise ValueError("target must not be empty")
    if isinstance(occurrence, bool) or not isinstance(occurrence, int) or occurrence < 1:
        raise ValueError(f"occurrence must be a positive int, got {occurrence!r}")
    plain, index = _strip_markup(text)
    pattern = _word_pattern(target) + (rf'[{_PUNCT}]*' if trailing_punct else '')
    matches = list(re.finditer(pattern, plain))
    if not matches:
        raise TargetNotFoundError(f"target {target!r} not found")
    if len(matches) < occurrence:
        raise TargetNotFoundError(f"only {len(matches)} occurrences of {target!r}, asked for occurrence {occurrence}")
    m = matches[occurrence - 1]
    punct = re.search(rf'[{_PUNCT}]+$', m.group(0))
    punct_pos = index[m.start() + punct.start()] if punct else None
    return index[m.start()], index[m.end() - 1] + 1, punct_pos


def _insert_tag(text: str, target_word: str, tag: str, occurrence: int = 1) -> str:
    """Inserts a point-event tag after a match of target_word, before any trailing punctuation."""
    _, end, punct = _locate(text, target_word, occurrence=occurrence)
    pos = punct if punct is not None else _CLOSERS_AND_EVENTS_RE.match(text, end).end()
    return f'{text[:pos]} {tag}{text[pos:]}'


def _balanced(fragment: str) -> bool:
    """True if every span tag in the fragment is opened and closed inside it."""
    stack = []
    for closing, name in _SPAN_TAG_RE.findall(fragment):
        if not closing:
            stack.append(name)
        elif not stack or stack.pop() != name:
            return False
    return not stack


def _wrap(text: str, target: str, name: str, occurrence: int = 1) -> str:
    """Wraps a match of target in a span tag, keeping existing spans properly nested."""
    start, end, _ = _locate(text, target, occurrence=occurrence)
    left = _OPENERS_BEFORE_RE.search(text, 0, start).start()
    right = _CLOSERS_RE.match(text, end).end()
    for s, e in ((start, end), (left, end), (start, right), (left, right)):
        if _balanced(text[s:e]):
            return f'{text[:s]}<{name}>{text[s:e]}</{name}>{text[e:]}'
    raise ValueError(f"target {target!r} partially overlaps an existing span")


# --------------------------------------------------------------------------- #
# Text features: they rewrite the line
# --------------------------------------------------------------------------- #


def add_micro_expression(text: str, target_word: str, expr_type: str = "laughter", occurrence: int = 1) -> str:
    """Injects a CosyVoice event tag (e.g. [laughter], [breath]) after a target word."""
    key = expr_type.lower()
    key = EVENT_ALIASES.get(key, key)
    if key not in EVENT_TAGS:
        valid = ", ".join(sorted([*EVENT_TAGS, *EVENT_ALIASES]))
        raise ValueError(f"unknown expr_type {expr_type!r}; valid keys: {valid}")

    return _insert_tag(text, target_word, EVENT_TAGS[key], occurrence)


def _event_feature(tag: str):
    """Builds the Option Table function for a single event tag."""
    def add_event(text: str, target_word: str, occurrence: int = 1) -> str:
        return _insert_tag(text, target_word, tag, occurrence)
    add_event.__doc__ = f"Injects {tag} after a target word."
    return add_event


def add_laughing_speech(text: str, target_phrase: str, occurrence: int = 1) -> str:
    """Wraps a phrase in <laughter> tags so it is spoken while laughing."""
    return _wrap(text, target_phrase, "laughter", occurrence)


def add_emphasis(text: str, target_word: str, occurrence: int = 1) -> str:
    """Wraps a specific word or phrase in <strong> tags."""
    return _wrap(text, target_word, "strong", occurrence)


def add_pause(text: str, target_word: str, pause_type: str = "dramatic", pause_ms: int | None = None,
              occurrence: int = 1) -> str:
    """Injects a pause after a target word.

    pause_ms gives a timed pause: a private ⟦pause:N⟧ marker that
    split_on_pauses turns into silence. Otherwise pause_type picks a soft
    pause: "dramatic" or "realization" punctuation, or an audible "breath".
    """
    if pause_ms is not None:
        if isinstance(pause_ms, bool) or not isinstance(pause_ms, int) or pause_ms < 0:
            raise ValueError(f"pause_ms must be a non-negative int, got {pause_ms!r}")
        _, end, _ = _locate(text, target_word, trailing_punct=True, occurrence=occurrence)
        pos = _CLOSERS_RE.match(text, end).end()
        # CosyVoice has no timed-pause token (neither 2 nor 3): the line is split here and silence added
        return f'{text[:pos]}⟦pause:{pause_ms}⟧{text[pos:]}'

    if pause_type == "breath":
        return _insert_tag(text, target_word, EVENT_TAGS["breath"], occurrence)
    if pause_type not in SOFT_PAUSES:
        valid = ", ".join(sorted([*SOFT_PAUSES, "breath"]))
        raise ValueError(f"unknown pause_type {pause_type!r}; valid keys: {valid} (or pass pause_ms)")
    # The frontend's Chinese path rewrites " - " to a comma and "——" to a space
    if pause_type == "realization" and _CHINESE_RE.search(text):
        raise ValueError("a realization pause is not supported in text containing Chinese; "
                         "use pause_type 'breath' or pause_ms")

    marker = SOFT_PAUSES[pause_type]
    _, end, punct = _locate(text, target_word, trailing_punct=True, occurrence=occurrence)
    if punct is None:
        pos = _CLOSERS_RE.match(text, end).end()
        return f'{text[:pos]}{marker}{text[pos:]}'
    # ? and ! carry the intonation cue, so they are kept rather than replaced
    if re.search(r'[?!]', text[punct:end]):
        if pause_type == "realization":
            raise ValueError(f"a realization pause cannot follow {text[punct:end]!r}; use dramatic or breath")
        return f'{text[:punct]}{marker}{text[punct:]}'
    return f'{text[:punct]}{marker}{text[end:]}'


def add_pronunciation(text: str, target_word: str, phonemes: str, occurrence: int = 1) -> str:
    """NEW: Replaces an English word with CMU phonemes, e.g. "R EH1 D", to fix its pronunciation or move its stress.

    Phonemes are given with spaces ("R EH1 D") or as tokens ("[R][EH1][D]").
    Vowels take a stress digit: 1 primary, 2 secondary, 0 none, so
    "P R AA1 JH EH2 K T" and "P R AH0 JH EH1 K T" say "project" two ways.
    The word is replaced, so other features cannot target it afterwards;
    process_tts_string therefore applies pronunciation after everything else.
    """
    if " " in target_word.strip() or _CJK_RE.search(target_word):
        raise ValueError("pronunciation takes one English word at a time")
    phones = [p.upper() for p in re.findall(r"[A-Za-z]+[012]?", phonemes)]
    if not phones:
        raise ValueError(f"no phonemes in {phonemes!r}; write them like 'R EH1 D'")
    unknown = [p for p in phones if p not in CMU_PHONES]
    if unknown:
        raise ValueError(f"not CMU phonemes: {', '.join(unknown)}")

    start, end, punct = _locate(text, target_word, occurrence=occurrence)
    end = punct if punct is not None else end
    if _MARKUP_RE.search(text, start, end):
        raise ValueError(f"target {target_word!r} already contains markup")
    return f'{text[:start]}{"".join(f"[{p}]" for p in phones)}{text[end:]}'


# --------------------------------------------------------------------------- #
# Plan features: they set something for the whole line
# --------------------------------------------------------------------------- #


def speed_instruction(rate: float) -> str | None:
    """CHANGED: Maps a rate to a CosyVoice 3 speed instruction: only the extremes exist."""
    if rate >= 1.5:
        return "fastest"
    if rate <= 0.7:
        return "slowest"
    return None


def adjust_speed(plan: TTSPlan, rate: float = 1.0, mode: str = "param") -> None:
    """Sets the speaking rate; 1.0 is normal, 2.0 is twice as fast.

    "param" stores the rate for the inference call's speed argument;
    "instruct" stores a speed instruction instead and resets the numeric
    speed. A speed instruction and a numeric speed are never both applied; an
    instruction set by another feature is left alone.
    CHANGED: CosyVoice 3 only has "fastest" and "slowest", so in "instruct"
    mode a rate between 0.7 and 1.5 falls back to the numeric speed.
    """
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0:
        raise ValueError(f"rate must be a positive number, got {rate!r}")
    if mode == "param":
        plan.speed = float(rate)
        if plan.instruct in SPEED_INSTRUCTIONS:
            plan.instruct = None
    elif mode == "instruct":
        instruction = speed_instruction(rate)
        plan.speed = 1.0 if instruction else float(rate)
        if instruction or plan.instruct in SPEED_INSTRUCTIONS:
            plan.instruct = instruction
    else:
        raise ValueError(f"unknown speed mode {mode!r}; valid keys: instruct, param")


def set_instruct(plan: TTSPlan, text: str) -> None:
    """Sets an instruction for the whole utterance: a key of INSTRUCT_PRESETS (e.g. "sad"), or free text.

    CHANGED: CosyVoice 3 only follows the instructions it was trained on
    (INSTRUCT_PRESETS). Free text is still accepted, with a warning in
    process_tts_string, because the model may ignore it.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"instruct text must be a non-empty string, got {text!r}")
    if "<|" in text or "|>" in text:
        raise ValueError("instruct text must not contain <| or |>")
    if len(text) > INSTRUCT_MAX_CHARS:
        raise ValueError(f"instruct text is {len(text)} characters; the limit is {INSTRUCT_MAX_CHARS}")
    plan.instruct = text.strip()


def _preset_feature(choices: tuple[str, ...], name: str, what: str):
    """NEW: Builds a plan feature that picks one preset instruction from `choices`."""
    def set_preset(plan: TTSPlan, *args, **kwargs) -> None:
        # Called as set_preset(plan, "sad") or set_preset(plan, emotion="sad")
        if len(args) + len(kwargs) != 1 or (kwargs and name not in kwargs):
            raise TypeError(f"{name} takes one argument: {name}")
        value = args[0] if args else kwargs[name]
        if value not in choices:
            raise ValueError(f"unknown {name} {value!r}; valid keys: {', '.join(choices)}")
        plan.instruct = value
    set_preset.__doc__ = f"NEW: Speaks the whole line {what} (a CosyVoice 3 instruction)."
    set_preset.__signature__ = inspect.Signature([
        inspect.Parameter("plan", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation="str"),
    ])
    return set_preset


set_emotion = _preset_feature(EMOTIONS, "emotion", "happy, sad or angry")      # NEW
set_volume = _preset_feature(VOLUMES, "volume", "loudly or softly")            # NEW
set_dialect = _preset_feature(DIALECTS, "dialect", "in a Chinese dialect")     # NEW


def _instruct_text(instruct: str) -> str:
    """NEW: The full CosyVoice 3 instruction: system prompt, the instruction, <|endofprompt|>."""
    body = INSTRUCT_PRESETS.get(instruct, instruct)
    return f"{SYSTEM_PROMPT} {body}{END_OF_PROMPT}"


# --------------------------------------------------------------------------- #
# The Option Table
# --------------------------------------------------------------------------- #

FORMATTING_INDEX = {
    "micro_expression": add_micro_expression,
    "emphasis": add_emphasis,
    "pause": add_pause,
    "laughing_speech": add_laughing_speech,
    "pronunciation": add_pronunciation,                                        # NEW
    **{feature: _event_feature(tag) for feature, tag in EVENT_TAGS.items()},
}

# Features that set data on the plan instead of rewriting the text
PLAN_INDEX = {
    "speed": adjust_speed,
    "instruct": set_instruct,
    "emotion": set_emotion,                                                    # NEW
    "volume": set_volume,                                                      # NEW
    "dialect": set_dialect,                                                    # NEW
}

# NEW: run after every other feature, because they replace the word they target
_APPLIED_LAST = {"pronunciation"}

# Kept working for older callers, but left out of describe_options
_HIDDEN_FEATURES = {"micro_expression"}

_TARGET_PARAMS = ("target_word", "target_phrase")
_KWARG_ENUMS = {
    "pause_type": sorted([*SOFT_PAUSES, "breath"]),
    "mode": ["param", "instruct"],
    "emotion": list(EMOTIONS),            # NEW
    "volume": list(VOLUMES),              # NEW
    "dialect": list(DIALECTS),            # NEW
}
_JSON_TYPES = {"str": "string", "int": "integer", "float": "number"}


def _target_param(func) -> str | None:
    """Returns the name of the function's target parameter, if it has one."""
    return next((name for name in inspect.signature(func).parameters if name in _TARGET_PARAMS), None)


def describe_options() -> list[dict]:
    """Describes the Option Table for an LLM or UI, built from the functions themselves.

    Every target parameter is advertised as "target". CHANGED: every text
    feature not yet verified by listening on CosyVoice 3 is "experimental"
    (the original only marked unverified event tokens).
    """
    options = []
    for feature, func in {**FORMATTING_INDEX, **PLAN_INDEX}.items():
        if feature in _HIDDEN_FEATURES:
            continue
        kwargs = {}
        # The first parameter is the text or the plan, supplied by process_tts_string
        for param in list(inspect.signature(func).parameters.values())[1:]:
            spec = {"type": _JSON_TYPES[str(param.annotation).split(" | ")[0]]}
            if param.default is inspect.Parameter.empty:
                spec["required"] = True
            elif param.default is not None:
                spec["default"] = param.default
            if param.name in _KWARG_ENUMS:
                spec["enum"] = _KWARG_ENUMS[param.name]
            if param.name in BOUNDS:
                spec["minimum"], spec["maximum"] = BOUNDS[param.name]
            kwargs["target" if param.name in _TARGET_PARAMS else param.name] = spec
        options.append({
            "feature": feature,
            "description": inspect.getdoc(func).split("\n\n")[0].replace("\n", " "),
            "kwargs": kwargs,
            "experimental": feature in FORMATTING_INDEX and feature not in VERIFIED_FEATURES,
        })
    return options


# --------------------------------------------------------------------------- #
# Applying the operations
# --------------------------------------------------------------------------- #


def sanitize_text(text: str) -> tuple[str, list[str]]:
    """Removes model tokens, pause markers, phonemes, <|…|> and any angle-bracket tag from raw input text.

    Returns the cleaned text and the pieces that were removed. The text between
    tags is kept, and a double space left behind by a removal is collapsed.
    Other square-bracketed text and a plain "<" or ">" are left alone.
    """
    clean, removed, pos = '', [], 0
    for m in _RESERVED_RE.finditer(text):
        clean += text[pos:m.start()]
        removed.append(m.group(0).strip())
        pos = m.end()
        if (not clean or clean.endswith(' ')) and text.startswith(' ', pos):
            pos += 1
    return clean + text[pos:], removed


def _clamp_kwargs(kwargs: dict, strict: bool) -> list[str]:
    """Clamps numeric kwargs to BOUNDS in place and returns a note per change; raises in strict mode."""
    notes = []
    for key, (low, high) in BOUNDS.items():
        value = kwargs.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        clamped = min(max(value, low), high)
        if clamped != value:
            if strict:
                raise ValueError(f"{key}={value!r} is outside {low} to {high}")
            kwargs[key] = clamped
            notes.append(f"{key}={value!r} is outside {low} to {high}; clamped to {clamped!r}")
    return notes


def _advisories(plan: TTSPlan) -> list[str]:
    """Notes on the finished plan that never fail an operation: unverified features and risky timed pauses."""
    text = plan.text
    notes = []
    verified = {EVENT_TAGS[key] for key in VERIFIED_FEATURES if key in EVENT_TAGS}
    for tag in dict.fromkeys(_EVENT_RE.findall(text)):
        if tag not in verified:
            notes.append(f"{tag} is experimental: not yet verified by listening on CosyVoice 3")
    # NEW: the new features get the same note
    if _PHONE_RE.search(text) and "pronunciation" not in VERIFIED_FEATURES:
        notes.append("pronunciation is experimental: not yet verified by listening on CosyVoice 3")
    if plan.instruct and plan.instruct not in INSTRUCT_PRESETS:
        notes.append(f"instruction {plan.instruct!r} is not one CosyVoice 3 was trained on and may be ignored; "
                     f"use a preset ({', '.join(k for k in INSTRUCT_PRESETS if k not in DIALECTS)}, or a dialect)")

    pieces = PAUSE_MARKER_RE.split(text)
    if len(pieces) == 1:
        return notes
    for before in pieces[:-1:2]:
        plain = _strip_markup(before)[0].rstrip()
        if plain and plain[-1] not in ".?!":
            notes.append(f"timed pause after {plain.split()[-1]!r} is not at a sentence end; "
                         "intonation may reset there, consider a soft pause")
    for segment in split_on_pauses(text):
        if isinstance(segment, TextSegment) and len(_strip_markup(segment.text)[0].split()) < 3:
            notes.append(f"text segment {segment.text!r} next to a timed pause has fewer than three words; "
                         "short segments may sound poor, consider a soft pause")
    return notes


def _warn(plan: TTSPlan, message: str) -> None:
    logger.warning(message)
    plan.warnings.append(message)


def process_tts_string(base_text: str, operations: list[dict], strict: bool = False) -> TTSPlan:
    """
    Applies a list of formatting operations to the base text.

    Expected operation format:
    {"feature": "feature_name", "kwargs": {"param1": "value1"}}

    Every targeted feature also accepts "target" in place of its own target
    kwarg, and "occurrence" (1-based) to pick a repeated word.

    Markup already present in base_text, unknown features, failed operations
    and out-of-range numbers are fixed or skipped and recorded in
    plan.warnings, or raised in strict mode. Warnings about an operation
    start with "op N (feature)", counting from 1.
    CHANGED: pronunciation operations run after all others (they replace words).
    """
    plan = TTSPlan(text=base_text)

    clean, removed = sanitize_text(base_text)
    if removed:
        message = f"base_text: removed markup {', '.join(removed)}"
        if strict:
            raise ValueError(message)
        plan.text = clean
        _warn(plan, message)

    numbered = list(enumerate(operations, start=1))
    # Stable: the order among the others, and among the pronunciations, is kept
    numbered.sort(key=lambda item: isinstance(item[1], dict) and item[1].get("feature") in _APPLIED_LAST)

    for number, op in numbered:
        feature = None
        try:
            feature = op.get("feature")
            kwargs = dict(op.get("kwargs") or {})

            # Lookup the function in the index
            func = PLAN_INDEX.get(feature) or FORMATTING_INDEX.get(feature)
            if func is None:
                raise ValueError("feature not found in Option Table")

            if "target" in kwargs:
                name = _target_param(func)
                if name is None or name in kwargs:
                    raise TypeError("'target' is not accepted here" if name is None
                                    else f"pass either 'target' or {name!r}, not both")
                kwargs[name] = kwargs.pop("target")
            for note in _clamp_kwargs(kwargs, strict):
                _warn(plan, f"op {number} ({feature}): {note}")

            if feature in PLAN_INDEX:
                previous = plan.instruct
                func(plan, **kwargs)
                if previous is not None and plan.instruct != previous:
                    _warn(plan, f"op {number} ({feature}): replaced earlier instruction {previous!r}")
            else:
                plan.text = func(plan.text, **kwargs)
        except Exception as e:
            if strict:
                raise
            _warn(plan, f"op {number} ({feature}): {e}")

    for note in _advisories(plan):
        _warn(plan, note)

    return plan


def split_on_pauses(text: str) -> list[Segment]:
    """Splits processed text on timed-pause markers into text and silence segments.

    A marker inside an open <strong> or <laughter> span closes the span before
    the marker and reopens it after. Empty text segments are dropped.
    """
    segments: list[Segment] = []
    open_spans: list[str] = []

    # re.split with one capture group alternates text, ms, text, ms, ...
    for i, piece in enumerate(PAUSE_MARKER_RE.split(text)):
        if i % 2:
            ms = int(piece)
            if segments and isinstance(segments[-1], SilenceSegment):
                segments[-1] = SilenceSegment(segments[-1].ms + ms)
            elif ms:
                segments.append(SilenceSegment(ms))
            continue

        reopened = ''.join(f'<{name}>' for name in open_spans)
        for closing, name in _SPAN_TAG_RE.findall(piece):
            if not closing:
                open_spans.append(name)
            elif open_spans and open_spans[-1] == name:
                open_spans.pop()
        closed = ''.join(f'</{name}>' for name in reversed(open_spans))

        if _SPAN_TAG_RE.sub('', piece).strip():
            segments.append(TextSegment(f'{reopened}{piece.strip()}{closed}'))

    return segments


# --------------------------------------------------------------------------- #
# Speaking the plan
# --------------------------------------------------------------------------- #


def synthesize(cosyvoice, plan: TTSPlan, prompt_wav: str, prompt_text: str | None = None, **kwargs):
    """Synthesizes a plan with a loaded CosyVoice 3 model, stitching audio and silence.

    Uses inference_instruct2 when the plan has an instruction, inference_zero_shot
    when prompt_text is given, and inference_cross_lingual otherwise. Extra
    kwargs are passed to the inference call. prompt_wav is a path to the
    reference audio file.

    CHANGED: CosyVoice 3 needs "You are a helpful assistant.<|endofprompt|>"
    in every call. It is added here: before the reference script (zero-shot),
    around the instruction (instruct), or before the text (cross-lingual).
    Pass prompt_text without it; a prompt_text that already has it is kept.

    text_frontend defaults to False: the frontend's sentence splitter collapses
    "..." into a single ".", which cancels the dramatic pause. With it off,
    numbers are not spelled out and long text is not split, so the plan gets
    an advisory for either. Pass text_frontend=True to keep the frontend.
    """
    if kwargs.pop("stream", False):
        if plan.speed != 1.0:
            raise ValueError(f"speed={plan.speed} requires stream=False; CosyVoice cannot change speed while streaming")
        raise ValueError("synthesize returns one stitched tensor and cannot stream; call it with stream=False")

    # Imported here so the text logic stays importable without torch
    import torch

    instruct = _instruct_text(plan.instruct) if plan.instruct else None              # CHANGED
    if instruct and prompt_text:
        _warn(plan, "synthesize: prompt_text ignored because the plan has an instruction")

    # CHANGED: the reference script gets CosyVoice 3's system prompt in front
    if prompt_text and END_OF_PROMPT not in prompt_text:
        prompt_text = f"{SYSTEM_PROMPT}{END_OF_PROMPT}{prompt_text.strip()}"

    text_frontend = kwargs.setdefault("text_frontend", False)
    if text_frontend and SOFT_PAUSES["dramatic"] in plan.text:
        _warn(plan, "synthesize: text_frontend=True collapses '...' into '.', so the dramatic pause is lost; "
                    "leave text_frontend off, or use a breath or timed pause")

    kwargs.update(stream=False, speed=plan.speed)
    chunks = []
    for segment in split_on_pauses(plan.text):
        if isinstance(segment, TextSegment) and not text_frontend:
            plain = _strip_markup(segment.text)[0]
            if re.search(r'\d', plain):
                _warn(plan, f"synthesize: text segment {segment.text!r} contains digits, which are not "
                            "spelled out with text_frontend=False; write numbers as words")
            # CJK text has no spaces, so each character counts as a word
            words = len(_CJK_RE.sub(' ', plain).split()) + len(_CJK_RE.findall(plain))
            if words > FRONTEND_OFF_MAX_WORDS:
                _warn(plan, f"synthesize: a text segment has {words} words, which is not split into sentences "
                            f"with text_frontend=False; keep segments under about {FRONTEND_OFF_MAX_WORDS} words")
        if isinstance(segment, SilenceSegment):
            chunks.append(torch.zeros(1, int(cosyvoice.sample_rate * segment.ms / 1000)))
        elif instruct:
            outputs = cosyvoice.inference_instruct2(segment.text, instruct, prompt_wav, **kwargs)
            chunks.extend(out["tts_speech"] for out in outputs)
        elif prompt_text:
            outputs = cosyvoice.inference_zero_shot(segment.text, prompt_text, prompt_wav, **kwargs)
            chunks.extend(out["tts_speech"] for out in outputs)
        else:
            # CHANGED: with no reference script, the system prompt goes in front of the text itself
            outputs = cosyvoice.inference_cross_lingual(f"{SYSTEM_PROMPT}{END_OF_PROMPT}{segment.text}",
                                                        prompt_wav, **kwargs)
            chunks.extend(out["tts_speech"] for out in outputs)

    return torch.cat(chunks, dim=1) if chunks else torch.zeros(1, 0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 1. The raw input string
    raw_text = "I can't believe we actually pulled that off. It was incredible."

    # 2. The requested formatting operations (e.g., generated by a UI or LLM)
    requested_operations = [
        {"feature": "emphasis", "kwargs": {"target_word": "actually"}},
        {"feature": "pause", "kwargs": {"target_word": "off.", "pause_type": "dramatic"}},
        {"feature": "micro_expression", "kwargs": {"target_word": "incredible.", "expr_type": "laughter"}},
        {"feature": "speed", "kwargs": {"rate": 1.1}},
        # NEW features
        {"feature": "pronunciation", "kwargs": {"target": "incredible", "phonemes": "IH2 N K R EH1 D AH0 B AH0 L"}},
        {"feature": "emotion", "kwargs": {"emotion": "happy"}},
    ]

    # 3. Process the string
    print(process_tts_string(raw_text, requested_operations))

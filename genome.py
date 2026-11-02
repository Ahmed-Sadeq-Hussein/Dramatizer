"""Transferable delivery edits, independent of the voice model and search.

Positions run from 0 (first word) to 1 (last word). A genome contains no
voice, sentence, score or audio: the same edits can be tried on another line.
Editing returns new objects, so a future search cannot change its parents.
"""
import math
import re
from dataclasses import asdict, dataclass


MAX_GENES = 6
MIN_RATE, MAX_RATE = 0.5, 2.0
EVENT_TOKENS = {
    "breath": "[breath]", "quick_breath": "[quick_breath]",
    "laughter": "[laughter]", "cough": "[cough]", "sigh": "[sigh]",
    "clucking": "[clucking]", "hissing": "[hissing]",
    "vocalized_noise": "[vocalized-noise]", "lipsmack": "[lipsmack]",
    "mn": "[mn]", "noise": "[noise]",
}
FEATURES = tuple(EVENT_TOKENS) + ("emphasis", "laughing_speech", "pause", "speed")
PAUSES = {"short": ",", "long": "..."}
# One definition of a word for all genome positioning. Apostrophes and
# hyphens within words stay attached; punctuation and spacing stay intact.
WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)


def _number(value, name, low, high):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not low <= value <= high):
        raise ValueError(f"{name} must be a finite number between {low} and {high}")


@dataclass(frozen=True)
class Gene:
    """One edit. Speed is global (position=None); all other edits need a position.

    Gene('emphasis', position=0.5)
    Gene('pause', position=0.75, kind='long')
    Gene('speed', rate=1.2)
    """

    feature: str
    position: float | None = None
    rate: float | None = None
    kind: str | None = None

    def __post_init__(self):
        if self.feature not in FEATURES:
            raise ValueError(f"Unknown feature {self.feature!r}; choose from {FEATURES}")
        if self.feature == "speed":
            _number(self.rate, "rate", MIN_RATE, MAX_RATE)
            if self.position is not None:
                raise ValueError("speed is global: omit position")
        else:
            _number(self.position, "position", 0.0, 1.0)
            if self.rate is not None:
                raise ValueError("Only speed accepts rate")
        if self.feature == "pause":
            if not isinstance(self.kind, str) or self.kind not in PAUSES:
                raise ValueError("pause needs kind='short' or kind='long'")
        elif self.kind is not None:
            raise ValueError("Only pause accepts kind")

    def to_dict(self):
        return {key: value for key, value in asdict(self).items() if value is not None}

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("A gene must be a dictionary")
        if set(data) - {"feature", "position", "rate", "kind"}:
            raise ValueError("Unknown gene fields")
        try:
            return cls(**data)
        except TypeError as error:
            raise ValueError("A gene needs a feature and its parameters") from error


@dataclass(frozen=True)
class Genome:
    """Up to six genes; an empty genome is neutral delivery.

    Order is retained for simultaneous vocal events. Duplicate edits and
    multiple global speed genes are rejected rather than silently resolved.
    """

    genes: tuple[Gene, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "genes", tuple(self.genes))
        if len(self.genes) > MAX_GENES:
            raise ValueError(f"A genome may contain at most {MAX_GENES} genes")
        if any(not isinstance(gene, Gene) for gene in self.genes):
            raise ValueError("genes must contain Gene objects")
        if len(set(self.genes)) != len(self.genes):
            raise ValueError("Duplicate genes are not allowed")
        if sum(g.feature == "speed" for g in self.genes) > 1:
            raise ValueError("A genome may contain only one speed gene")

    def add(self, gene):
        return Genome(self.genes + (gene,))

    def remove(self, index):
        self._check_index(index)
        return Genome(self.genes[:index] + self.genes[index + 1:])

    def replace(self, index, gene):
        self._check_index(index)
        return Genome(self.genes[:index] + (gene,) + self.genes[index + 1:])

    def modify(self, index, **changes):
        """Change parameters of an existing feature; use replace to change feature."""
        self._check_index(index)
        if set(changes) - {"position", "rate", "kind"}:
            raise ValueError("modify accepts position, rate and kind only")
        data = self.genes[index].to_dict()
        data.update(changes)
        return self.replace(index, Gene.from_dict(data))

    def _check_index(self, index):
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self.genes):
            raise IndexError("Gene index is out of range")

    @property
    def speed(self):
        return next((g.rate for g in self.genes if g.feature == "speed"), 1.0)

    @property
    def features(self):
        """Names for reward functions, e.g. wer_reward(features=genome.features)."""
        return [g.feature for g in self.genes]

    def to_dict(self):
        return {"schema_version": 1, "genes": [g.to_dict() for g in self.genes]}

    @classmethod
    def from_dict(cls, data):
        if (not isinstance(data, dict) or set(data) != {"schema_version", "genes"}
                or type(data["schema_version"]) is not int or data["schema_version"] != 1
                or not isinstance(data["genes"], list)):
            raise ValueError("Expected schema_version=1 and a genes list")
        return cls(tuple(Gene.from_dict(g) for g in data["genes"]))


@dataclass(frozen=True)
class Delivery:
    """Backend inputs, keeping spoken words separate from markup and speed."""

    text: str
    markup: str
    speed: float


def to_cosyvoice(text, genome=None):
    """Render a genome once, against word indexes in the ORIGINAL plain text.

    Events precede a word; emphasis/laughing wrap one word; pauses follow it.
    Pause punctuation is a hint, never an exact silence duration. Existing
    punctuation after a word takes precedence over a requested pause.
    Tokens are from CosyVoice3Tokenizer; audibility still needs listening tests.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if genome is None:
        genome = Genome()
    if not isinstance(genome, Genome):
        raise TypeError("genome must be a Genome")
    if not genome.genes:
        return Delivery(text, text, 1.0)
    if any(char in text for char in "<>[]"):
        raise ValueError("With a genome, supply plain text without markup or brackets")
    words = list(WORD.finditer(text))
    if not words:
        raise ValueError("A genome needs text containing at least one word")
    edits = {}
    for gene in genome.genes:
        if gene.feature != "speed":
            # Nearest word, with .5 rounded upward; endpoints always map exactly.
            index = int(gene.position * (len(words) - 1) + 0.5)
            edits.setdefault(index, []).append(gene)

    parts, cursor = [], 0
    for index, word in enumerate(words):
        parts.append(text[cursor:word.start()])
        genes = edits.get(index, [])
        events = [EVENT_TOKENS[g.feature] for g in genes if g.feature in EVENT_TOKENS]
        if events:
            parts.append(" ".join(events) + " ")
        value = word.group()
        # Fixed nesting prevents crossing tags when several edits land together.
        if any(g.feature == "laughing_speech" for g in genes):
            value = f"<laughter>{value}</laughter>"
        if any(g.feature == "emphasis" for g in genes):
            value = f"<strong>{value}</strong>"
        parts.append(value)
        pauses = [g.kind for g in genes if g.feature == "pause"]
        following = text[word.end():word.end() + 1]
        if pauses and (not following or following.isspace()):
            parts.append(PAUSES["long" if "long" in pauses else "short"])
        cursor = word.end()
    parts.append(text[cursor:])
    return Delivery(text, "".join(parts), genome.speed)

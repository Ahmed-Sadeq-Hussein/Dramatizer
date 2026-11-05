"""Random-parent exploration with WER/speaker gates and a diverse shortlist.

The callback renders a genome and returns raw measurements in Evaluation.
Accepted genomes become parents. Exact repeats are skipped before evaluation.
"""
import math
import logging
import random
from dataclasses import asdict, dataclass
from numbers import Real

from genome import FEATURES, MAX_GENES, MIN_RATE, MAX_RATE, Gene, Genome
from genome_distance import genome_vector

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchConfig:
    steps: int = 100
    seed: int | None = None
    features: tuple[str, ...] = FEATURES
    position_sigma: float = 0.1
    rate_sigma: float = 0.1
    max_wer: float = 0.10
    min_similarity: float = 0.80
    shortlist_size: int = 10
    vector_bins: int = 5
    speed_weight: float = 0.25

    def __post_init__(self):
        object.__setattr__(self, "features", tuple(self.features))
        if type(self.steps) is not int or self.steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError("seed must be an integer or None")
        if (not self.features or any(f not in FEATURES for f in self.features)
                or len(set(self.features)) != len(self.features)):
            raise ValueError("features must be a nonempty, unique subset of FEATURES")
        for name in ("position_sigma", "rate_sigma"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, low, high in (("max_wer", 0, math.inf), ("min_similarity", -1, 1)):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or not low <= value <= high):
                raise ValueError(f"{name} must be finite and between {low} and {high}")
        if type(self.shortlist_size) is not int or self.shortlist_size < 1:
            raise ValueError("shortlist_size must be a positive integer")
        genome_vector(Genome(), bins=self.vector_bins, speed_weight=self.speed_weight)


@dataclass(frozen=True)
class Mutation:
    genome: Genome
    operator: str


def _random_gene(feature, rng):
    if feature == "speed":
        return Gene(feature, rate=rng.uniform(MIN_RATE, MAX_RATE))
    return Gene(feature, position=rng.random(),
                kind=rng.choice(("short", "long")) if feature == "pause" else None)


def mutate(genome, *, rng=None, config=None):
    """Make one add/remove/alter/replace edit, without changing the parent.

    Choose uniformly among feasible operators, then among relevant genes.
    Alter makes a Gaussian position/rate step (clamped to legal bounds), or
    flips pause kind. Replace changes feature, retaining a local position.
    features limits new features; existing seed genes can still be altered
    or removed. Restrict it to tokens you have checked by listening.
    """
    if not isinstance(genome, Genome):
        raise TypeError("genome must be a Genome")
    config = config if config is not None else SearchConfig()
    rng = rng if rng is not None else random.Random(config.seed)
    has_speed = any(g.feature == "speed" for g in genome.genes)
    additions = [f for f in config.features if f != "speed" or not has_speed]
    replacements = {
        i: [f for f in config.features if f != gene.feature
            and (f != "speed" or not has_speed)]
        for i, gene in enumerate(genome.genes)
    }
    replaceable = [i for i, options in replacements.items() if options]
    operators = []
    if len(genome.genes) < MAX_GENES and additions:
        operators.append("add")
    if genome.genes:
        operators.extend(("remove", "alter"))
    if replaceable:
        operators.append("replace")

    # Retry no-ops from clamping/duplicate genes. A removal is always a valid
    # fallback for a nonempty parent; an empty parent can always add a gene.
    for _ in range(100):
        operator = rng.choice(operators)
        try:
            if operator == "add":
                child = genome.add(_random_gene(rng.choice(additions), rng))
            elif operator == "remove":
                child = genome.remove(rng.randrange(len(genome.genes)))
            elif operator == "replace":
                index = rng.choice(replaceable)
                old = genome.genes[index]
                gene = _random_gene(rng.choice(replacements[index]), rng)
                if gene.feature != "speed" and old.position is not None:
                    gene = Gene(gene.feature, position=old.position, kind=gene.kind)
                child = genome.replace(index, gene)
            else:
                index = rng.randrange(len(genome.genes))
                gene = genome.genes[index]
                if gene.feature == "speed":
                    rate = min(MAX_RATE, max(MIN_RATE, gene.rate + rng.gauss(0, config.rate_sigma)))
                    child = genome.modify(index, rate=rate)
                elif gene.feature == "pause" and rng.random() < 0.5:
                    child = genome.modify(index, kind="long" if gene.kind == "short" else "short")
                else:
                    position = min(1.0, max(0.0, gene.position + rng.gauss(0, config.position_sigma)))
                    child = genome.modify(index, position=position)
        except ValueError:
            continue  # duplicate gene: try another edit
        if child != genome:
            return Mutation(child, operator)
    if genome.genes:
        return Mutation(genome.remove(rng.randrange(len(genome.genes))), "remove")
    return Mutation(genome.add(_random_gene(rng.choice(additions), rng)), "add")


@dataclass(frozen=True)
class Evaluation:
    """Raw WER and cosine speaker similarity, plus an optional rendered WAV path."""

    wer: float
    speaker_similarity: float
    audio_path: str | None = None


def _finite_metric(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        return None
    return float(value)


@dataclass(frozen=True)
class Candidate:
    genome: Genome
    evaluation: Evaluation
    step: int

    def to_dict(self):
        return {"genome": self.genome.to_dict(), "evaluation": asdict(self.evaluation), "step": self.step}


@dataclass(frozen=True)
class SearchStep:
    step: int
    parent_step: int
    operator: str
    status: str
    wer: float | None = None
    speaker_similarity: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class SearchResult:
    initial: Genome
    candidates: tuple[Candidate, ...]
    shortlist: tuple[Candidate, ...]
    evaluations: int
    history: tuple[SearchStep, ...]
    config: SearchConfig

    def to_dict(self):
        # Store each accepted genome once. Shortlist/history refer to step IDs.
        return {
            "schema_version": 2, "config": asdict(self.config),
            "initial": self.initial.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "shortlist_steps": [candidate.step for candidate in self.shortlist],
            "evaluations": self.evaluations,
            "history": [asdict(step) for step in self.history],
        }


def select_diverse(candidates, n=10, *, bins=5, speed_weight=0.25):
    """Greedy farthest-point sampling of accepted genomes, using their vectors.

    Start with the first accepted candidate. Repeatedly choose the candidate
    whose nearest selected neighbour is furthest away. Ties follow discovery
    order. Return all candidates if fewer than n exist, including zero.
    """
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    candidates = list(candidates)
    if len(candidates) <= n:
        return tuple(candidates)
    vectors = [genome_vector(c.genome, bins=bins, speed_weight=speed_weight) for c in candidates]
    selected = [0]
    remaining = list(range(1, len(candidates)))
    nearest = {index: math.dist(vectors[index], vectors[0]) for index in remaining}
    while len(selected) < n:
        chosen = max(remaining, key=lambda index: nearest[index])
        selected.append(chosen)
        remaining.remove(chosen)
        for index in remaining:
            nearest[index] = min(nearest[index], math.dist(vectors[index], vectors[chosen]))
    return tuple(candidates[index] for index in selected)


def explore(evaluate, initial=None, *, config=None):
    """Try config.steps mutations, each from a random member of the parent pool.

    The initial genome is an unevaluated starting parent (step 0), as in the
    previous Dramatizer. Only accepted children join the pool and shortlist.
    Both gates are inclusive: WER <= max_wer and similarity >= min_similarity.
    Evaluation errors and invalid/nonfinite measurements are recorded and
    skipped. Duplicate attempts consume a step but never invoke evaluate.

    Exact identity includes gene order, because simultaneous vocal events
    render in that order. Vector distance is never used for deduplication.
    The random seed controls proposals, while the callback controls TTS seeds.
    Rejected/error audio files written by the callback remain its responsibility.
    """
    config = config if config is not None else SearchConfig()
    initial = initial if initial is not None else Genome()
    if not isinstance(initial, Genome):
        raise TypeError("initial must be a Genome")
    rng = random.Random(config.seed)
    seen = {initial}
    parents = [(initial, 0)]
    accepted = []
    history = []
    evaluations = 0
    for step in range(1, config.steps + 1):
        parent, parent_step = rng.choice(parents)
        mutation = mutate(parent, rng=rng, config=config)
        child = mutation.genome
        if child in seen:
            history.append(SearchStep(step, parent_step, mutation.operator, "duplicate"))
            continue
        seen.add(child)
        evaluations += 1
        try:
            result = evaluate(child)
            if not isinstance(result, Evaluation):
                raise TypeError("evaluate(genome) must return Evaluation(wer, speaker_similarity, audio_path)")
            wer = _finite_metric(result.wer)
            similarity = _finite_metric(result.speaker_similarity)
            valid = (wer is not None and similarity is not None
                     and 0 <= wer <= config.max_wer and similarity >= config.min_similarity)
            audio_path = str(result.audio_path) if result.audio_path is not None else None
        except Exception as error:
            history.append(SearchStep(step, parent_step, mutation.operator, "error",
                                      error=f"{type(error).__name__}: {error}"))
            logger.warning("Step %s evaluation failed: %s: %s", step, type(error).__name__, error)
            continue
        history.append(SearchStep(step, parent_step, mutation.operator,
                                  "accepted" if valid else "rejected", wer, similarity))
        if valid:
            accepted.append(Candidate(child, Evaluation(wer, similarity, audio_path), step))
            parents.append((child, step))
    shortlist = select_diverse(accepted, config.shortlist_size,
                               bins=config.vector_bins, speed_weight=config.speed_weight)
    if not shortlist:
        errors = sum(step.status == "error" for step in history)
        rejected = sum(step.status == "rejected" for step in history)
        duplicates = sum(step.status == "duplicate" for step in history)
        logger.warning("Empty shortlist: %s evaluation errors, %s rejected, %s duplicates. "
                       "See result.history for errors and WER/similarity measurements.",
                       errors, rejected, duplicates)
    return SearchResult(initial, tuple(accepted), shortlist, evaluations, tuple(history), config)


def local_search(evaluate, initial=None, *, config=None):
    """Existing entry point for explore; callback now returns raw Evaluation."""
    return explore(evaluate, initial, config=config)


def explore_voice(reference_audio, reference_text, text, initial=None, *, config=None):
    """Run exploration with the project's generator, Whisper and speaker encoder.

    WER uses the original plain-text script. Similarity compares each output
    to the reference audio. The returned candidates include generated paths.
    """
    from Voice_model import generate_speech
    from WER import word_error_rate
    from Similarity import speaker_similarity

    # Fail before generating a batch if the requested measurement models
    # cannot load. Per-candidate audio/measurement errors are still recorded.
    import model
    model.speaker_encoder()
    model.recogniser()

    def evaluate(genome):
        audio = generate_speech(reference_audio, reference_text, text, genome=genome)
        return Evaluation(
            wer=word_error_rate(audio, text),
            speaker_similarity=speaker_similarity(audio, reference_audio),
            audio_path=str(audio),
        )

    return explore(evaluate, initial, config=config)

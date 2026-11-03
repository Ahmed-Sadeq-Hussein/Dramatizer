"""Seeded local mutation search. The caller supplies a higher-is-better score.

No rendering occurs implicitly: score(genome) may generate audio, apply the
existing rewards, store audio/measurements, or use a cheap development score.
"""
import math
import random
from dataclasses import asdict, dataclass
from numbers import Real

from genome import FEATURES, MAX_GENES, MIN_RATE, MAX_RATE, Gene, Genome


@dataclass(frozen=True)
class SearchConfig:
    steps: int = 100
    seed: int | None = None
    features: tuple[str, ...] = FEATURES
    position_sigma: float = 0.1
    rate_sigma: float = 0.1
    accept_equal: bool = True

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
        if type(self.accept_equal) is not bool:
            raise ValueError("accept_equal must be a bool")


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
class SearchStep:
    step: int
    parent: Genome
    genome: Genome
    operator: str
    score: float
    accepted: bool
    cached: bool

    def to_dict(self):
        return {**asdict(self), "parent": self.parent.to_dict(), "genome": self.genome.to_dict()}


@dataclass(frozen=True)
class SearchResult:
    initial: Genome
    initial_score: float
    best: Genome
    best_score: float
    current: Genome
    current_score: float
    evaluations: int
    history: tuple[SearchStep, ...]
    config: SearchConfig

    def to_dict(self):
        return {
            "schema_version": 1, "config": asdict(self.config),
            "initial": self.initial.to_dict(), "initial_score": self.initial_score,
            "best": self.best.to_dict(), "best_score": self.best_score,
            "current": self.current.to_dict(), "current_score": self.current_score,
            "evaluations": self.evaluations,
            "history": [step.to_dict() for step in self.history],
        }


def local_search(score, initial=None, *, config=None):
    """Evaluate the seed, then attempt config.steps single-edit mutations.

    Accept higher scores and, by default, ties. Reject worse scores and keep
    searching from the last accepted genome. Return all attempts and the best.
    At most steps+1 calls to score; exact repeat genomes reuse their score.
    The callback must return a finite real number; exceptions stop the run.
    There is no novelty objective or implicit reward from vector distance.

    Seed controls mutations only. For stochastic TTS, fix rendering seeds in
    the callback, or define a repeated-render aggregate: caching assumes a
    stable, context-independent score for each exact genome during this run.
    """
    config = config if config is not None else SearchConfig()
    initial = initial if initial is not None else Genome()
    if not isinstance(initial, Genome):
        raise TypeError("initial must be a Genome")
    rng = random.Random(config.seed)
    cache = {}

    def evaluate(genome):
        cached = genome in cache
        if not cached:
            value = score(genome)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ValueError("score(genome) must return a finite real number")
            cache[genome] = float(value)
        return cache[genome], cached

    current = best = initial
    current_score, _ = evaluate(initial)
    initial_score = best_score = current_score
    history = []
    for step in range(1, config.steps + 1):
        mutation = mutate(current, rng=rng, config=config)
        value, cached = evaluate(mutation.genome)
        accepted = value > current_score or (config.accept_equal and value == current_score)
        history.append(SearchStep(step, current, mutation.genome, mutation.operator, value, accepted, cached))
        if accepted:
            current, current_score = mutation.genome, value
        if value > best_score:
            best, best_score = mutation.genome, value
    return SearchResult(initial, initial_score, best, best_score, current, current_score,
                        len(cache), tuple(history), config)

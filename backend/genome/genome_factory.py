"""genome_factory.py | Random genomes and their mutations, ported from the first Dramatizer.

    from backend.genome.genome_factory import GenomeFactory

    factory = GenomeFactory(min_operations=3, max_operations=3, seed=7)
    genome = factory.random_genome("I can't believe we actually pulled that off.")
    child, mutation = factory.mutate(genome, text)

A random genome has min_operations to max_operations genes. Each gene's feature
is drawn with equal odds from FEATURES: 14 that sit on a word (emphasis,
laughing while speaking and the 12 sound tokens), a pause, and the speed. Only
one speed gene has an effect, so a genome never gets a second one.

The same as the first Dramatizer's genome_factory.py, with one change: it draws
from its own random generator, so a seed gives the same genomes every time.
"""
import random

from backend.genome.candidate import WORDS, Candidate

# Features that sit on a word, by position in the sentence
TARGETED_FEATURES = [
    "emphasis",
    "breath",
    "quick_breath",
    "noise",
    "laughter",
    "cough",
    "clucking",
    "accent",
    "hissing",
    "sigh",
    "vocalized_noise",
    "lipsmack",
    "mn",
    "laughing_speech",
]
FEATURES = TARGETED_FEATURES + ["pause", "speed"]
PAUSE_KINDS = ["dramatic", "realization"]

MIN_OPERATIONS = 1
MAX_OPERATIONS = 5
SPEED_RANGE = (0.7, 1.3)     # where a new speed gene starts
SPEED_BOUNDS = (0.5, 2.0)    # how far a mutation may take it
POSITION_STEP = 0.15         # how far "modify" moves a gene through the sentence
SPEED_STEP = 0.1             # how far "modify" changes the rate


class GenomeFactory:
    def __init__(self, min_operations=MIN_OPERATIONS, max_operations=MAX_OPERATIONS, seed=None):
        if not 1 <= min_operations <= max_operations:
            raise ValueError(f"Need 1 <= min_operations <= max_operations, not {min_operations} and {max_operations}.")
        self.min_operations = min_operations
        self.max_operations = max_operations
        self.random = random.Random(seed)

        self.targeted_features = list(TARGETED_FEATURES)
        self.features = list(FEATURES)

    def random_genome(self, text):
        """Create a completely random candidate."""
        self._words(text)
        n_operations = self.random.randint(self.min_operations, self.max_operations)
        candidate = Candidate([])

        for _ in range(n_operations):
            candidate.operations.append(self._new_operation(candidate, text))

        return candidate

    def mutate(self, candidate, text):
        """A copy with one random change (add, remove, modify or replace a gene), and the change's name."""
        child = candidate.copy()
        possible_mutations = ["add"]

        if child.operations:
            possible_mutations += ["remove", "modify", "replace"]

        mutation = self.random.choice(possible_mutations)

        if mutation == "add":
            self._add_operation(child, text)
        elif mutation == "remove":
            self._remove_operation(child)
        elif mutation == "modify":
            self._modify_operation(child, text)
        elif mutation == "replace":
            self._replace_operation(child, text)

        return child, mutation

    def _add_operation(self, candidate, text):
        if len(candidate.operations) >= self.max_operations:
            self._modify_operation(candidate, text)
            return
        candidate.operations.append(self._new_operation(candidate, text))

    def _remove_operation(self, candidate):
        if len(candidate.operations) <= self.min_operations:
            return
        candidate.operations.pop(self.random.randrange(len(candidate.operations)))

    def _modify_operation(self, candidate, text):
        if not candidate.operations:
            self._add_operation(candidate, text)
            return

        gene = candidate.operations[self.random.randrange(len(candidate.operations))]

        # Move a targeted operation slightly through the sentence
        if "position" in gene:
            shift = self.random.uniform(-POSITION_STEP, POSITION_STEP)
            gene["position"] = max(0.0, min(1.0, gene["position"] + shift))

        # Change speaking speed slightly
        elif gene["feature"] == "speed":
            shift = self.random.uniform(-SPEED_STEP, SPEED_STEP)
            gene["rate"] = round(max(SPEED_BOUNDS[0], min(SPEED_BOUNDS[1], gene["rate"] + shift)), 2)

    def _replace_operation(self, candidate, text):
        if not candidate.operations:
            self._add_operation(candidate, text)
            return
        index = self.random.randrange(len(candidate.operations))
        candidate.operations.pop(index)
        candidate.operations.insert(index, self._new_operation(candidate, text))

    def _new_operation(self, candidate, text):
        """A random operation that fits the genome: only one speed gene has an effect, so a second one is never added."""
        has_speed = any(gene["feature"] == "speed" for gene in candidate.operations)

        while True:
            operation = self._random_operation(text)
            if not (has_speed and operation["feature"] == "speed"):
                return operation

    def _random_operation(self, text):
        return self._random_operation_for_feature(self.random.choice(self.features), text)

    def _random_operation_for_feature(self, feature, text):
        # All features that target a position in the sentence
        if feature in self.targeted_features:
            return {"feature": feature, "position": round(self.random.random(), 3)}

        if feature == "pause":
            return {"feature": "pause", "position": round(self.random.random(), 3),
                    "kind": self.random.choice(PAUSE_KINDS)}

        if feature == "speed":
            return {"feature": "speed", "rate": round(self.random.uniform(*SPEED_RANGE), 2)}

        raise ValueError(f"Unsupported feature: {feature}")

    def _words(self, text):
        words = WORDS.findall(text)
        if not words:
            raise ValueError("Cannot generate targeted operations: text contains no words.")
        return words

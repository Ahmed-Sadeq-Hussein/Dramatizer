"""A fixed, interpretable genome space. No audio, model weights or training.

Each local feature has position bins. Linear interpolation avoids abrupt
jumps at bin boundaries. Counts are divided by MAX_GENES; speed is log2(rate).
Euclidean distance here measures these descriptors, not perceived emotion.
Different genomes can share a vector (it is not an identity/deduplication key).
"""
import math

from genome import FEATURES, MAX_GENES, Genome


CHANNELS = tuple(f for f in FEATURES if f not in ("pause", "speed")) + (
    "pause_short", "pause_long",
)


def _validate(bins, speed_weight):
    if type(bins) is not int or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    if (isinstance(speed_weight, bool) or not isinstance(speed_weight, (int, float))
            or not math.isfinite(speed_weight) or speed_weight <= 0):
        raise ValueError("speed_weight must be finite and positive")


def vector_labels(bins=5):
    """Coordinate names, in the same order as genome_vector."""
    _validate(bins, 1.0)
    return tuple(f"{feature}@{i / (bins - 1):.3f}"
                 for feature in CHANNELS for i in range(bins)) + ("log2_speed",)


def genome_vector(genome, *, bins=5, speed_weight=0.25):
    """Return a tuple of len(CHANNELS)*bins+1 coordinates (76 by default).

    A local gene contributes 1/MAX_GENES, split between adjacent bins.
    Thus repeated features retain their count as well as their placement.
    Neutral speed=1 maps to zero; 0.5 and 2 map to -/+ speed_weight.
    The default speed weight is a starting design choice, not learned data.
    Always use the same bins and weight when comparing or clustering vectors.
    """
    if not isinstance(genome, Genome):
        raise TypeError("genome must be a Genome")
    _validate(bins, speed_weight)
    vector = [0.0] * (len(CHANNELS) * bins + 1)
    for gene in genome.genes:
        if gene.feature == "speed":
            continue
        channel = f"pause_{gene.kind}" if gene.feature == "pause" else gene.feature
        offset = CHANNELS.index(channel) * bins
        location = gene.position * (bins - 1)
        left = int(location)
        fraction = location - left
        vector[offset + left] += (1 - fraction) / MAX_GENES
        if left + 1 < bins:
            vector[offset + left + 1] += fraction / MAX_GENES
    vector[-1] = speed_weight * math.log2(genome.speed)
    return tuple(vector)


def genome_distance(first, second, *, bins=5, speed_weight=0.25):
    """Euclidean distance between two genome descriptor vectors.

    Symmetric and nonnegative, but zero does not guarantee identical genomes:
    bins lose within-bin detail and ignore simultaneous event ordering.
    """
    return math.dist(genome_vector(first, bins=bins, speed_weight=speed_weight),
                     genome_vector(second, bins=bins, speed_weight=speed_weight))

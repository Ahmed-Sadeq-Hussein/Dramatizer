"""Feature descriptors for the current Candidate.operations schema.

genome_distance and the shortlist in search.py come from Baesar's genome branch (genome_distance.py,
genome_search.select_diverse, 2-5 Nov), moved onto this schema on 14 Nov. The distance measures these
descriptors, not perceived delivery, and 0 does not mean two genomes are the same.
"""
import math

from backend.genome.genome_factory import MAX_OPERATIONS, PAUSE_KINDS, TARGETED_FEATURES

BINS = 5
CHANNELS = (*TARGETED_FEATURES, *(f"pause_{kind}" for kind in PAUSE_KINDS), "pause_breath")
LABELS = tuple(f"{feature}@{i / (BINS - 1):g}" for feature in CHANNELS for i in range(BINS)) + ("log2_speed",)
SIZE = len(LABELS)


def genome_vector(genes):
    """Soft position bins, separate pause kinds, and signed log-speed."""
    if not isinstance(genes, list) or len(genes) > MAX_OPERATIONS:
        raise ValueError(f"A genome must be a list of at most {MAX_OPERATIONS} genes.")
    result = [0.0] * SIZE
    has_speed = False
    for gene in genes:
        try:
            feature = gene["feature"]
            if feature == "speed":
                rate = float(gene["rate"])
                if has_speed or not math.isfinite(rate) or not 0.5 <= rate <= 2:
                    raise ValueError("Expected at most one speed gene with rate 0.5 to 2.")
                result[-1] = 0.25 * math.log2(rate)
                has_speed = True
                continue
            channel = f"pause_{gene.get('kind', 'dramatic')}" if feature == "pause" else feature
            offset = CHANNELS.index(channel) * BINS
            position = float(gene["position"])
            if not math.isfinite(position) or not 0 <= position <= 1:
                raise ValueError("Gene position must be between 0 and 1.")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid preference descriptor gene {gene!r}: {error}") from error
        location = position * (BINS - 1)
        left = int(location)
        fraction = location - left
        result[offset + left] += (1 - fraction) / MAX_OPERATIONS
        if left < BINS - 1:
            result[offset + left + 1] += fraction / MAX_OPERATIONS
    return tuple(result)


def genome_distance(first, second):
    """Euclidean distance between two genomes' descriptor vectors (symmetric, never negative)."""
    return math.dist(genome_vector(first), genome_vector(second))

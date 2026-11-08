"""candidate.py | One genome: a short list of delivery edits, ported from the first Dramatizer.

A genome is a list of genes. A gene says what to do and where, as a position
in the sentence from 0 (the first word) to 1 (the last), not as a word, so the
same genome can be put on any line and any voice:

    {"feature": "emphasis", "position": 0.42}                      stress a word
    {"feature": "laughter", "position": 1.0}                       a sound after a word
    {"feature": "pause", "position": 0.18, "kind": "dramatic"}     a pause after a word
    {"feature": "speed", "rate": 1.12}                             the speaking rate (one at most)

resolve(text) turns the positions into words, giving the operations that
tts_formatter.process_tts_string understands:

    from backend.genome.candidate import Candidate

    genome = Candidate([{"feature": "emphasis", "position": 0.5}])
    genome.resolve("I can't believe we actually pulled that off.")
    # [{"feature": "emphasis", "kwargs": {"target": "we"}}]

The same as the first Dramatizer's candidate.py, plus describe() for people.
"""
import re
from copy import deepcopy

WORDS = re.compile(r"[A-Za-z']+")


class Candidate:
    def __init__(self, operations=None):
        self.operations = operations or []

    def copy(self):
        return Candidate(deepcopy(self.operations))

    def resolve(self, text):
        """
        Convert the abstract genome into operations understood
        by tts_formatter.py.
        """
        words = WORDS.findall(text)

        if not words:
            raise ValueError("Text contains no words.")

        resolved = []

        for gene in self.operations:
            feature = gene["feature"]

            if "position" in gene:
                position = max(0.0, min(1.0, gene["position"]))
                index = round(position * (len(words) - 1))
                kwargs = {"target": words[index]}

                # A repeated word: say which one, or the formatter
                # would always take the first
                occurrence = words[:index + 1].count(words[index])

                if occurrence > 1:
                    kwargs["occurrence"] = occurrence

                if "kind" in gene:
                    kwargs["pause_type"] = gene["kind"]

                resolved.append({"feature": feature, "kwargs": kwargs})

            elif feature == "speed":
                resolved.append({"feature": "speed", "kwargs": {"rate": gene["rate"]}})

            else:
                raise ValueError(f"Cannot resolve gene: {gene}")

        return resolved

    def describe(self, text):
        """Each gene in words, on this line: ["emphasis on 'actually'", "speed 1.12", ...]."""
        described = []
        for gene, operation in zip(self.operations, self.resolve(text)):
            kwargs = operation["kwargs"]
            if gene["feature"] == "speed":
                described.append(f"speed {gene['rate']:g}")
                continue
            word = f"'{kwargs['target']}'" + (f" (the {_ordinal(kwargs['occurrence'])})" if "occurrence" in kwargs else "")
            if gene["feature"] == "emphasis":
                described.append(f"emphasis on {word}")
            elif gene["feature"] == "laughing_speech":
                described.append(f"laughing while saying {word}")
            elif gene["feature"] == "pause":
                described.append(f"{gene.get('kind', 'dramatic')} pause after {word}")
            else:
                described.append(f"{gene['feature'].replace('_', ' ')} after {word}")
        return described

    def __repr__(self):
        return f"Candidate(operations={self.operations})"


def _ordinal(n):
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"

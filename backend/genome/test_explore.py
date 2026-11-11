"""Tests of explore.py's choices (no rendering, no models).

    python -m unittest backend.genome.test_explore
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))    # run as a file by main_backend.py test

from backend.genome import explore as ex  # noqa: E402


def pool():
    feats = ["emphasis", "laughter", "breath", "sigh", "cough", "pause"]
    out = {}
    for n in ex.GENE_CYCLE:
        genomes = []
        for i in range(40):
            genes = []
            for j in range(n):
                f = feats[(i + j * 7) % len(feats)]
                gene = {"feature": f, "position": ((i * 13 + j * 5) % 10) / 9}
                if f == "pause":
                    gene["kind"] = "dramatic" if i % 2 else "realization"
                genes.append(gene)
            genomes.append(genes)
        out[n] = genomes
    return out


class Explore(unittest.TestCase):
    def test_vector_of_plain_is_zero_and_speed_counts(self):
        self.assertFalse(ex.genome_vector([]).any())
        self.assertNotEqual(ex.genome_vector([{"feature": "speed", "rate": 1.2}])[-1], 0)

    def test_spread_never_repeats_a_genome_and_cycles_counts(self):
        out = ex.spread_candidates(pool(), ["a", "b", "c"], 10)
        every = [str(g) for gs in out.values() for g in gs]
        self.assertEqual(len(every), len(set(every)))
        for genomes in out.values():
            self.assertEqual(len(genomes), 10)
            self.assertEqual(sorted({len(g) for g in genomes}), list(ex.GENE_CYCLE))

    def test_spread_beats_random_on_distance(self):
        p = pool()
        out = ex.spread_candidates(p, ["a"], 10)["a"]
        X = np.array([ex.genome_vector(g) for g in out])
        rng = np.random.default_rng(0)
        flat = [g for gs in p.values() for g in gs]
        rand = [np.array([ex.genome_vector(flat[i]) for i in rng.choice(len(flat), 10, replace=False)]) for _ in range(50)]
        nearest = lambda Y: np.mean(np.sort(np.linalg.norm(Y[:, None] - Y[None], axis=2), axis=1)[:, 1])
        self.assertGreater(nearest(X), np.mean([nearest(r) for r in rand]))

    def test_farthest_points_starts_far_and_never_repeats(self):
        pts = np.array([[0.1, 0], [5, 0], [0, 4], [5, 0.1], [2, 2]])
        picks = ex.farthest_points(pts, 3, np.zeros(2))
        self.assertEqual(picks[0], 3)
        self.assertEqual(len(set(picks)), 3)
        self.assertEqual(ex.farthest_points(pts, 9, np.zeros(2)).__len__(), 5)
        self.assertEqual(ex.farthest_points([], 3, np.zeros(2)), [])

    def test_piles_are_even_and_mixed(self):
        voices = [{"voice": f"{d}/{i}", "dataset": d, "pitch_hz": 100 + 10 * i, "loner": i == 0}
                  for d, n in (("A", 12), ("B", 10), ("C", 2)) for i in range(n)]
        piles = ex.deal_piles(voices)
        sizes = [list(piles.values()).count(p) for p in (1, 2, 3)]
        self.assertEqual(sizes, [8, 8, 8])
        for d in "AB":
            per = [sum(1 for v in voices if v["dataset"] == d and piles[v["voice"]] == p) for p in (1, 2, 3)]
            self.assertLessEqual(max(per) - min(per), 1)

    def test_heard_change_of_the_plain_render_is_zero(self):
        a = {"seconds": 7.0, "pitch_st": 5.0, "pitch_spread_st": 2.0, "loudness_spread_db": 6.0, "longest_pause": 0.3}
        change = ex.heard_change(a, a, {"word_errors": 0, "similarity": 1.0})
        self.assertEqual(set(change), set(ex.HEARD))
        self.assertTrue(all(abs(x) < 1e-12 for x in change.values()))


if __name__ == "__main__":
    unittest.main()

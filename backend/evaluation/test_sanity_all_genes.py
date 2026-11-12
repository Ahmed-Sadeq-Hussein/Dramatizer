"""Tests of the every-gene-type sanity run: which genomes it draws. No GPU needed.

    python -m unittest backend.evaluation.test_sanity_all_genes
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))    # run as a file by main_backend.py test
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sanity_all_genes as sag  # noqa: E402

LINE = "Please close the door on your way out."


class Genomes(unittest.TestCase):
    def test_one_genome_of_one_gene_per_gene_type(self):
        genomes = sag.genomes_for(LINE, 0)
        names = [g["gene_type"] for g in genomes]
        self.assertEqual(len(names), 17)
        self.assertEqual(len(set(names)), 17)
        self.assertIn("hissing", names)
        self.assertIn("pause (dramatic)", names)
        self.assertIn("pause (realization)", names)
        self.assertTrue(all(len(g["genes"]) == 1 for g in genomes))
        self.assertEqual([g["number"] for g in genomes], list(range(1, 18)))

    def test_the_same_job_draws_the_same_genomes_and_another_job_others(self):
        a, b, c = sag.genomes_for(LINE, 3), sag.genomes_for(LINE, 3), sag.genomes_for(LINE, 4)
        self.assertEqual([g["genes"] for g in a], [g["genes"] for g in b])
        self.assertNotEqual([g["genes"] for g in a], [g["genes"] for g in c])

    def test_speed_stays_in_range(self):
        for job in range(20):
            speed = next(g for g in sag.genomes_for(LINE, job) if g["gene_type"] == "speed")
            self.assertTrue(0.7 <= speed["speed"] <= 1.3)


if __name__ == "__main__":
    unittest.main()

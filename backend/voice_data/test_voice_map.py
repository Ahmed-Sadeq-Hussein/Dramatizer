"""Tests of the voice map's maths on made-up embeddings (no audio, no models).

    python -m unittest backend.voice_data.test_voice_map
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))    # run as a file by main_backend.py test

from backend.voice_data import voice_map as vm  # noqa: E402


def made_up(seed=0, per=10):
    """Two 'datasets' that differ by a large shared offset (the recording), each with two voice groups."""
    rng = np.random.default_rng(seed)
    dim = 16
    offset = {"A": np.eye(dim)[0] * 3, "B": np.eye(dim)[1] * 3}
    group = {0: np.eye(dim)[2], 1: np.eye(dim)[3]}
    emb, datasets, groups = [], [], []
    for d in ("A", "B"):
        for g in (0, 1):
            for _ in range(per):
                emb.append(offset[d] + group[g] + rng.normal(0, 0.15, dim))
                datasets.append(d)
                groups.append(g)
    return vm.unit(np.array(emb)), np.array(datasets), np.array(groups)


def agreement(labels, truth):
    return max(np.mean(labels == truth), np.mean(labels != truth))


class VoiceMap(unittest.TestCase):
    def test_similarity_is_cosine(self):
        emb, _, _ = made_up()
        sim = vm.similarity_matrix(emb)
        self.assertTrue(np.allclose(np.diag(sim), 1.0))
        self.assertTrue(np.allclose(sim, sim.T))

    def test_plain_similarity_groups_by_dataset(self):
        emb, datasets, _ = made_up()
        share, chance = vm.same_dataset_share(vm.similarity_matrix(emb), datasets)
        self.assertEqual(share, 1.0)
        self.assertLess(chance, 0.5)

    def test_voice_similarity_groups_by_voice(self):
        emb, datasets, groups = made_up()
        sim = vm.voice_similarity(emb, datasets)
        family = vm.nested_clusters(sim, levels=1)[0]
        self.assertEqual(agreement(family, groups == 0), 1.0)
        self.assertEqual(agreement(vm.communities(sim) == vm.communities(sim)[0], groups == groups[0]), 1.0)

    def test_snorm_keeps_the_order_within_a_dataset(self):
        emb, datasets, _ = made_up()
        sim = vm.similarity_matrix(vm.remove_recording(emb, datasets))
        normed = vm.snorm(sim, datasets)
        a = np.flatnonzero(datasets == "A")
        for i in a[:5]:
            others = [j for j in a if j != i]
            self.assertEqual(list(np.argsort(sim[i, others])), list(np.argsort(normed[i, others])))

    def test_nested_clusters_split_in_two_each_level(self):
        emb, datasets, _ = made_up(per=12)
        levels = vm.nested_clusters(vm.voice_similarity(emb, datasets))
        counts = [len(set(lv)) for lv in levels]
        self.assertEqual(counts[0], 2)
        for level, count in enumerate(counts, 1):       # at most double each level (groups under 8 stay whole)
            self.assertTrue(counts[max(level - 2, 0)] <= count <= 2 ** level)
        for coarse, fine in zip(levels, levels[1:]):      # every fine cluster sits inside one coarse cluster
            for c in set(fine):
                self.assertEqual(len(set(coarse[fine == c])), 1)

    def test_quotas_are_even_and_never_exceed_a_dataset(self):
        self.assertEqual(vm.quotas(["A"] * 10 + ["B"] * 10 + ["C"] * 2, 9), {"A": 4, "B": 3, "C": 2})
        self.assertEqual(sum(vm.quotas(["A"] * 3, 10).values()), 3)

    def test_spread_pick_keeps_shares_and_beats_random(self):
        emb, datasets, _ = made_up()
        sim = vm.voice_similarity(emb, datasets)
        picks = vm.spread_pick(sim, datasets, 4)
        self.assertEqual(len(set(picks)), 4)
        self.assertEqual(sorted(datasets[picks]), ["A", "A", "B", "B"])
        self.assertGreaterEqual(vm.coverage(sim, picks)["mean"], vm.random_coverage(sim, datasets, 4, draws=100))

    def test_dedupe_keeps_one_per_tight_cluster_and_every_loner(self):
        rng = np.random.default_rng(1)
        centres = np.eye(8)[:3] * 4
        points = [c + rng.normal(0, 0.1, 8) for c in centres for _ in range(5)] + [np.eye(8)[5] * 4]   # 3 tight + 1 loner
        sim = -np.linalg.norm(np.array(points)[:, None] - np.array(points)[None], axis=2)
        picks, loners, stands_for = vm.dedupe_pick(sim, close_share=0.2, loner_share=1 / 16)
        self.assertEqual(loners, [15])
        self.assertEqual(len(picks), 4)
        self.assertEqual(sorted(p // 5 for p in picks if p != 15), [0, 1, 2])
        self.assertEqual(sorted(i for m in stands_for.values() for i in m), list(range(16)))
        for p, members in stands_for.items():
            self.assertEqual(members[0], p)
            self.assertEqual({i // 5 for i in members}, {p // 5})

    def test_medoid_is_a_member(self):
        emb, _, groups = made_up()
        sim = vm.similarity_matrix(emb)
        for cluster, index in vm.medoids(sim, groups).items():
            self.assertEqual(groups[index], cluster)


if __name__ == "__main__":
    unittest.main()

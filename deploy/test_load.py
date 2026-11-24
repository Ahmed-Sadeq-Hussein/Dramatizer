"""test_load.py | The load test's arithmetic: percentiles, the run summary and the capacity by Little's law.

    python deploy/test_load.py
"""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "deploy"))

import load  # noqa: E402


class Numbers(unittest.TestCase):
    def test_percentiles(self):
        self.assertEqual(load.percentile([1, 2, 3, 4, 5], 50), 3)
        self.assertEqual(load.percentile([10, 20], 95), 19.5)
        self.assertIsNone(load.percentile([None], 50))

    def test_capacity_by_littles_law(self):
        # 7.5 s of GPU per render: 0.133 renders/s; each person asks 1/45 a second -> 6 people, /1.3 -> 4
        served, queue = load.capacity(7.5, 20)
        self.assertEqual(served, 4)
        # those 4.6 people ask 0.1026 renders/s, each 20 s in the system -> 2.05 in it, x1.3 -> 3
        self.assertEqual(queue, 3)

    def test_a_faster_gpu_serves_more_people(self):
        self.assertGreater(load.capacity(3, 10)[0], load.capacity(9, 10)[0])


class Summary(unittest.TestCase):
    def test_a_run_is_summed_up(self):
        events = [
            {"person": 1, "kind": "render", "at": 0, "status": "done", "total": 10, "work": 8, "wait": 2, "place": 0},
            {"person": 2, "kind": "render", "at": 1, "status": "done", "total": 20, "work": 8, "wait": 12, "place": 2},
            {"person": 3, "kind": "render", "at": 2, "status": "http 503", "total": 0.1},
            {"person": 3, "kind": "render", "at": 3, "status": "timeout", "total": 300, "place": 5},
            {"person": 1, "kind": "question", "at": 4, "status": "done", "total": 6, "source": "model"},
            {"person": 2, "kind": "question", "at": 5, "status": "http 429", "total": 0.1},
        ]
        gpu = [{"at": 0, "memory_mb": 7000, "busy_percent": 90}, {"at": 5, "memory_mb": 8000, "busy_percent": 50}]
        s = load.summarize(events, gpu, people=3, minutes=2)
        self.assertEqual(s["renders"]["done"], 2)
        self.assertEqual(s["renders"]["per_minute"], 1)
        self.assertEqual(s["renders"]["refused_503"], 1)
        self.assertEqual(s["renders"]["timeouts"], 1)
        self.assertEqual(s["renders"]["deepest_queue"], 5)
        self.assertEqual(s["renders"]["work_p50"], 8)
        self.assertEqual(s["questions"]["sources"], {"model": 1})
        self.assertEqual(s["questions"]["refused_429"], 1)
        self.assertEqual(s["gpu"], {"memory_mb_max": 8000, "busy_percent_mean": 70.0})
        self.assertIn("| 3 | 1.0 |", load.table([s]))

    def test_the_users_file_is_read_without_comments(self):
        path = Path(tempfile.mkdtemp()) / "users.txt"
        path.write_text("# who\nload-01 secret1\nload-02 secret2\n\n", encoding="utf-8")
        self.assertEqual([u for u, _ in load.users(path)], ["load-01", "load-02"])


if __name__ == "__main__":
    unittest.main()

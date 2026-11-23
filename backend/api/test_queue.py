"""test_queue.py | The limits on the GPU job queue (Milestone 4, step 6): MAX_QUEUE for everyone together, MAX_WAIT,
dropping a render whose page was closed, and cancelling your own job.

    python backend/api/test_queue.py
"""

import queue
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.api import server  # noqa: E402
from backend.evidence_views import Refused  # noqa: E402


def nothing():
    return {}


class Queue(unittest.TestCase):
    def setUp(self):
        for p in (mock.patch.object(server, "jobs", {}),
                  mock.patch.object(server, "waiting", queue.Queue()),
                  mock.patch.object(server, "took", server.defaultdict(lambda: server.deque(maxlen=20))),
                  mock.patch.object(server, "MAX_QUEUE", 12),
                  mock.patch.object(server, "MAX_WAIT", 600)):
            p.start()
            self.addCleanup(p.stop)

    def fill(self, n, kind="speak"):
        return [server.queue_job(kind, f"person {i}", nothing, {}) for i in range(n)]

    def test_a_flood_is_refused_at_once_with_503_and_retry_after(self):
        self.fill(12)
        started = time.perf_counter()
        for i in range(200):
            with self.assertRaises(Refused) as caught:
                server.queue_job("speak", f"someone else {i}", nothing, {})
        self.assertLess(time.perf_counter() - started, 1.0)             # fast refusals, not waits
        self.assertEqual(caught.exception.status, 503)
        self.assertGreaterEqual(caught.exception.retry_after, 5)
        self.assertIn("busy", str(caught.exception))
        self.assertEqual(len(server.jobs), 12)                         # nothing refused was kept
        self.assertEqual(server.waiting.qsize(), 12)

    def test_the_503_goes_out_with_a_retry_after_header(self):
        full = Refused("The host is busy.", 503)
        full.retry_after = 42
        with server.app.test_request_context():
            reply, status, headers = server.refused(full)
        self.assertEqual((status, headers["Retry-After"]), (503, "42"))

    def test_one_person_still_gets_429_first(self):
        for _ in range(server.MAX_WAITING):
            server.queue_job("speak", "ana", nothing, {})
        with self.assertRaises(Refused) as caught:
            server.queue_job("speak", "ana", nothing, {})
        self.assertEqual(caught.exception.status, 429)

    def test_a_job_not_started_in_time_is_dropped_and_frees_its_slot(self):
        first, *_ = self.fill(12)
        first["_queued"] -= 601
        job = server.queue_job("speak", "late comer", nothing, {})       # the old job's slot is free again
        self.assertEqual(first["status"], "expired")
        self.assertIn("10 minutes", first["error"])
        self.assertEqual(job["status"], "queued")

    def test_a_render_nobody_waits_for_is_dropped_but_a_search_is_not(self):
        render = server.queue_job("speak", "ana", nothing, {})
        search = server.queue_job("genome", "ana", nothing, {})
        for job in (render, search):
            job["_asked"] -= server.ABANDONED + 1
        server.expire()
        self.assertEqual(render["status"], "cancelled")
        self.assertEqual(search["status"], "queued")                   # searches are started and left to run

    def test_asking_about_a_job_keeps_it(self):
        render = server.queue_job("speak", "ana", nothing, {})
        render["_asked"] -= server.ABANDONED - 5
        with server.app.test_request_context():
            server.g.user = "ana"
            server.job_state.__wrapped__(render["id"])
        render["_queued"] -= 1
        server.expire()
        self.assertEqual(render["status"], "queued")

    def test_the_worker_skips_dropped_jobs(self):
        ran = []
        dropped = server.queue_job("speak", "ana", lambda: ran.append("dropped") or {}, {})
        kept = server.queue_job("speak", "bo", lambda: ran.append("kept") or {}, {})
        server.drop(dropped, "cancelled", "Cancelled by you.")
        threading.Thread(target=server.worker, daemon=True).start()
        for _ in range(200):
            if kept["status"] == "done":
                break
            time.sleep(0.01)
        self.assertEqual((ran, dropped["status"], kept["status"]), (["kept"], "cancelled", "done"))
        self.assertEqual(len(server.took["speak"]), 1)                 # its time now feeds the wait estimate

    def test_cancel_only_your_own_queued_job(self):
        mine = server.queue_job("speak", "ana", nothing, {})
        cancel = server.cancel_job.__wrapped__
        with server.app.test_request_context():
            server.g.user = "bo"
            with self.assertRaises(Refused) as caught:
                cancel(mine["id"])
            self.assertEqual(caught.exception.status, 404)             # someone else's job: as if it did not exist
            server.g.user = "ana"
            self.assertEqual(cancel(mine["id"]).get_json()["job"]["status"], "cancelled")
            with self.assertRaises(Refused) as caught:
                cancel(mine["id"])
            self.assertEqual(caught.exception.status, 409)
        mine2 = server.queue_job("speak", "ana", nothing, {})
        mine2["status"] = "running"
        with server.app.test_request_context():
            server.g.user = "ana"
            with self.assertRaises(Refused) as caught:
                cancel(mine2["id"])
        self.assertEqual(caught.exception.status, 409)                 # the GPU finishes what it started

    def test_the_wait_estimate(self):
        with mock.patch.dict(server.TYPICAL_SECONDS, {"speak": 30, "genome": 900}):
            first, second = self.fill(2)
            third = server.queue_job("genome", "carl", nothing, {})
            self.assertEqual([server.wait_seconds(j["id"]) for j in (first, second, third)], [0, 30, 60])
            first.update(status="running", _started=time.time() - 10)
            self.assertEqual(server.wait_seconds(second["id"]), 20)
            server.took["speak"].extend([40, 50, 60])                  # measured times replace the first guess
            self.assertEqual(server.seconds_per("speak"), 50)
            shown = server.job_json(third)
            self.assertEqual(shown["expected_start_s"], 40 + 50)
            self.assertFalse([k for k in shown if k.startswith("_")])  # internal times are not sent


if __name__ == "__main__":
    unittest.main()

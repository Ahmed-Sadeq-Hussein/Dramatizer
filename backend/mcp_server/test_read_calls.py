"""Tests for read_calls.py with a stand-in for the API: no server, no model, no mcp package.

    python backend/mcp_server/test_read_calls.py
"""
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import read_calls as rc  # noqa: E402
from read_calls import CallError, Session  # noqa: E402


class FakeApi:
    """Stands in for dramatizer_api.Api. Records each GET and answers from a table.
    A refusal is a SystemExit, which is what dramatizer_api raises when the API says no."""

    def __init__(self, answers=None, refuse=()):
        self.asked = []
        self.answers = answers or {}
        self.refuse = refuse

    def log_in(self, user=None, key=None):
        self.logged_in = (user, key)

    def json(self, method, path, body=None):
        self.asked.append((method, path))
        if any(path.startswith(prefix) for prefix in self.refuse):
            raise SystemExit(f"{method} {path}: 403 Your account may not do this. Ask the host.")
        return self.answers.get(path.split("?")[0], {"ok": True})


def session(**kwargs):
    return Session(api=FakeApi(**kwargs), who="listener_b")


class Calls(unittest.TestCase):
    def test_filters_become_the_query_and_empty_ones_are_left_out(self):
        s = session()
        rc.query_scores(s, listener="listener_a", group_by="gene_type", sort="mean")
        method, path = s.api.asked[-1]
        self.assertEqual(method, "GET")
        self.assertTrue(path.startswith("/evidence/scores?"))
        self.assertIn("listener=listener_a", path)
        rc.query_scores(s, listener=None, voice="")
        self.assertEqual(s.api.asked[-1][1], "/evidence/scores")

    def test_research_questions_are_1_to_6_and_bad_ones_never_reach_the_api(self):
        s = session()
        rc.evidence_question(s, 5)
        self.assertEqual(s.api.asked[-1][1], "/evidence/questions/5")
        for bad in (0, 7, "5", True):
            with self.subTest(bad=bad), self.assertRaises(CallError):
                rc.evidence_question(s, bad)
        self.assertEqual(len(s.api.asked), 1)

    def test_search_size_is_capped_at_20(self):
        s = session()
        rc.search_evidence(s, "sigh", limit=50)
        self.assertIn("k=20", s.api.asked[-1][1])

    def test_an_empty_question_is_refused_before_any_call(self):
        s = session()
        with self.assertRaises(CallError):
            rc.answer_evidence(s, "   ")
        self.assertEqual(s.api.asked, [])

    def test_a_run_id_with_slashes_is_refused(self):
        s = session()
        with self.assertRaises(CallError):
            rc.search_details(s, "../secret")
        self.assertEqual(s.api.asked, [])

    def test_renders_mine_is_sent_only_when_asked(self):
        s = session()
        rc.renders(s, mine=True)
        self.assertIn("mine=1", s.api.asked[-1][1])
        rc.renders(s)
        self.assertNotIn("mine", s.api.asked[-1][1])


class Refusals(unittest.TestCase):
    def test_a_refusal_from_the_api_comes_back_with_its_reason(self):
        s = session(refuse=("/evidence/questions/",))
        with self.assertRaisesRegex(CallError, "may not do this"):
            rc.evidence_question(s, 1)

    def test_results_can_be_narrowed_to_one_name(self):
        answers = {"/results": {"baselines": [{"name": "v4"}, {"name": "v5"}],
                                "milestone2": [{"name": "random-v2"}]}}
        got = rc.results(session(answers=answers), name="v4")
        self.assertEqual([b["name"] for b in got["baselines"]], ["v4"])
        self.assertEqual(got["milestone2"], [])


class LoggingIn(unittest.TestCase):
    def test_an_account_needs_its_password_in_the_environment(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DRAMATIZER_PASSWORD", None)
            with self.assertRaises(CallError):
                rc.log_in(FakeApi(), user="listener_b")

    def test_an_api_key_gives_a_session_and_the_key_is_never_shown(self):
        api = FakeApi(answers={"/me": {"name": "listener_c", "account": {"permissions": ["rate", "search"]}}})
        s = rc.log_in(api, key="drm_secret")
        self.assertEqual(api.logged_in, (None, "drm_secret"))
        self.assertEqual(s.who, "listener_c")
        self.assertIn("search", s.permissions)
        self.assertNotIn("drm_secret", repr(s))


if __name__ == "__main__":
    unittest.main()
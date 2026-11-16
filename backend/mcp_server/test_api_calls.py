"""test_api_calls.py | Keep the planned MCP and login calls (api_calls.py) in step with the API.

The calls are stubs for now. These tests check the plan, not the work: every route named exists in
backend/api/server.py with the same method, every permission is one accounts.py knows, every call that
needs a permission is one the API guards with it, and every stub says it is not implemented yet.
No server, no model, and the mcp package is not needed.

    python backend/mcp_server/test_api_calls.py
"""
import inspect
import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import api_calls  # noqa: E402
from backend.api import accounts  # noqa: E402

SERVER = (HERE.parent / "api" / "server.py").read_text(encoding="utf-8")


def server_routes():
    """{(METHOD, route): permission or None} from server.py's decorators."""
    found = {}
    for m in re.finditer(r'@app\.(get|post)\(f"\{API\}([^"]+)"\)\n(?:@needs_key\n)?(?:@needs\("(\w+)"\)\n)?', SERVER):
        found[(m.group(1).upper(), m.group(2))] = m.group(3)
    return found


class ThePlan(unittest.TestCase):
    def test_every_route_exists_with_its_method(self):
        routes = server_routes()
        wanted = [(t.method, t.route) for t in api_calls.MCP_TOOLS.values()] + list(api_calls.LOGIN_ROUTES.values())
        missing = [w for w in wanted if w not in routes]
        self.assertEqual(missing, [])

    def test_permissions_match_what_the_api_guards(self):
        routes = server_routes()
        for name, tool in api_calls.MCP_TOOLS.items():
            with self.subTest(name):
                self.assertEqual(tool.permission, routes[(tool.method, tool.route)])
                self.assertTrue(tool.permission is None or tool.permission in accounts.PERMISSIONS)

    def test_every_tool_takes_the_session_first_and_has_a_docstring(self):
        for name, tool in api_calls.MCP_TOOLS.items():
            with self.subTest(name):
                self.assertEqual(next(iter(inspect.signature(tool.function).parameters)), "session")
                self.assertTrue(inspect.getdoc(tool.function))

    def test_nothing_is_implemented_yet(self):
        conn = api_calls.Connection("https://127.0.0.1:5005", "0" * 64)
        session = api_calls.Session(conn, token="drs_test")
        for name, tool in api_calls.MCP_TOOLS.items():
            with self.subTest(name):
                params = list(inspect.signature(tool.function).parameters.values())[1:]
                extra = [None for p in params if p.kind is p.POSITIONAL_OR_KEYWORD]
                with self.assertRaises(NotImplementedError):
                    tool.function(session, *extra)

    def test_the_token_is_not_shown(self):
        session = api_calls.Session(api_calls.Connection("u", "f"), token="drs_secret")
        self.assertNotIn("drs_secret", repr(session))


if __name__ == "__main__":
    unittest.main()

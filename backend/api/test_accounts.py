"""Tests of the login accounts (accounts.py) and the API routes that use them. No models, no audio.

    python -m unittest backend.api.test_accounts
"""
import json
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))    # run as a file by main_backend.py test

from backend.api import accounts  # noqa: E402

FAST = 1000     # hashing rounds in tests; the real number is accounts.ROUNDS


class Temporary(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.dir = Path(folder.name)
        patches = [mock.patch.object(accounts, "ACCOUNTS_PATH", self.dir / "accounts.json"),
                   mock.patch.object(accounts, "SESSIONS_PATH", self.dir / "sessions.json"),
                   mock.patch.object(accounts, "ROUNDS", FAST)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)


class Passwords(Temporary):
    def test_stored_only_as_a_salted_hash(self):
        account, password = accounts.new_account("julius", "Julius")
        text = (self.dir / "accounts.json").read_text(encoding="utf-8")
        self.assertNotIn(password, text)
        self.assertTrue(account.password.startswith("pbkdf2_sha256$"))
        _, other = accounts.new_account("rama", "Rama", password=password)
        stored = json.loads(text)["accounts"]["julius"]["password"]
        self.assertNotEqual(stored, accounts.get("rama").password)      # same password, different salt

    def test_check_password(self):
        stored = accounts.hash_password("correct horse battery")
        self.assertTrue(accounts.check_password("correct horse battery", stored))
        self.assertFalse(accounts.check_password("correct horse batterY", stored))
        self.assertFalse(accounts.check_password("anything", "not a hash"))

    def test_random_passwords_are_long_and_mixed(self):
        for _ in range(20):
            word = accounts.new_password()
            self.assertEqual(len(word), 16)
            self.assertIsNone(accounts.password_problem(word))

    def test_weak_passwords_refused(self):
        with self.assertRaises(ValueError):
            accounts.new_account("julius", "Julius", password="short")
        with self.assertRaises(ValueError):
            accounts.new_account("julius", "Julius", password="aaaaaaaaaaaaaaaa")


class Logging_in(Temporary):
    def test_right_and_wrong_password(self):
        _, password = accounts.new_account("julius", "Julius")
        self.assertEqual(accounts.authenticate("Julius ", password).username, "julius")
        with self.assertRaises(accounts.LoginRefused):
            accounts.authenticate("julius", password + "x")

    def test_unknown_user_gets_the_same_answer(self):
        accounts.new_account("julius", "Julius")
        messages = []
        for name in ("nobody", "julius"):
            try:
                accounts.authenticate(name, "wrong password!!")
            except accounts.LoginRefused as error:
                messages.append(str(error))
        self.assertEqual(messages[0], messages[1])

    def test_locks_after_failures_then_unlocks(self):
        _, password = accounts.new_account("julius", "Julius")
        now = accounts._now()
        for _ in range(accounts.MAX_FAILURES):
            with self.assertRaises(accounts.LoginRefused):
                accounts.authenticate("julius", "wrong password!!", now=now)
        with self.assertRaisesRegex(accounts.LoginRefused, "locked"):
            accounts.authenticate("julius", password, now=now)
        later = now + timedelta(minutes=accounts.LOCK_MINUTES + 1)
        self.assertEqual(accounts.authenticate("julius", password, now=later).username, "julius")

    def test_disabled_account_cannot_log_in(self):
        _, password = accounts.new_account("julius", "Julius")
        accounts.set_disabled("julius", True)
        with self.assertRaisesRegex(accounts.LoginRefused, "switched off"):
            accounts.authenticate("julius", password)

    def test_sessions(self):
        account, _ = accounts.new_account("julius", "Julius")
        token = accounts.open_session(account)
        self.assertNotIn(token, (self.dir / "sessions.json").read_text(encoding="utf-8"))
        self.assertEqual(accounts.session_account(token).username, "julius")
        later = accounts._now() + timedelta(hours=accounts.SESSION_HOURS + 1)
        self.assertIsNone(accounts.session_account(token, now=later))
        accounts.close_session(token)
        self.assertIsNone(accounts.session_account(token))

    def test_new_password_ends_sessions(self):
        account, _ = accounts.new_account("julius", "Julius")
        token = accounts.open_session(account)
        new = accounts.set_password("julius")
        self.assertIsNone(accounts.session_account(token))
        self.assertEqual(accounts.authenticate("julius", new).username, "julius")

    def test_roles_decide_permissions(self):
        listener, _ = accounts.new_account("julius", "Julius")
        self.assertTrue(listener.can("rate"))
        self.assertTrue(listener.can("speak"))          # a normal account makes audio in the studio
        self.assertFalse(listener.can("search"))
        self.assertFalse(listener.can("name"))
        self.assertNotIn("password", listener.public())
        with self.assertRaises(ValueError):
            accounts.new_account("rama", "Rama", role="king")

    def test_display_names_must_differ(self):
        accounts.new_account("julius", "Julius")
        with self.assertRaises(ValueError):
            accounts.new_account("julius2", "julius")


def fake_run(folder, run_id, pile, genomes=3):
    folder.mkdir(parents=True)
    run = {"id": run_id, "voice": f"VCTK/{run_id}", "text": "A line.", "created": "2026-11-10 10:00:00",
           "status": "finished", "batch": "test-batch", "pile": pile, "genes_per_genome": 2, "seed": 0,
           "baseline": {"audio": "baseline.wav"},
           "genomes": [{"number": n, "audio": f"genome_{n:02d}.wav", "described": [f"gene {n}"], "markup": f"m{n}",
                        "speed": 1.0, "verdict": "fine", "genes": [], "sanity": {}} for n in range(1, genomes + 1)]}
    (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
    return run


class Api(Temporary):
    """The routes, through Flask's test client."""

    def setUp(self):
        super().setUp()
        from backend.api import server
        from backend.genome import ratings, search

        self.server = server
        runs = self.dir / "runs"
        for p in (mock.patch.object(search, "RUNS_DIR", runs),
                  mock.patch.object(ratings, "LOG_PATH", self.dir / "mos.jsonl"),
                  mock.patch.object(server.time, "sleep", lambda s: None),
                  mock.patch.object(server, "API_DIR", self.dir),
                  mock.patch.object(server, "ACCESS_LOG", self.dir / "access.log"),
                  mock.patch.object(server, "AUDIT_LOG", self.dir / "audit.jsonl"),
                  mock.patch.object(server, "_failed_logins", [])):
            p.start()
            self.addCleanup(p.stop)
        self.ratings = ratings
        fake_run(runs / "r1", "r1", 1)
        fake_run(runs / "r2", "r2", 1)
        fake_run(runs / "r3", "r3", 2)
        _, self.password = accounts.new_account("julius", "Julius")
        self.client = server.app.test_client()

    def login(self, password=None):
        reply = self.client.post("/api/v1/login", json={"username": "julius", "password": password or self.password})
        return reply

    def test_login_and_me(self):
        self.assertEqual(self.login("wrong password!!").status_code, 401)
        reply = self.login()
        self.assertEqual(reply.status_code, 200)
        token = reply.get_json()["token"]
        me = self.client.get("/api/v1/me", headers={"Authorization": f"Bearer {token}"}).get_json()
        self.assertEqual(me["account"]["display_name"], "Julius")
        self.assertNotIn("password", json.dumps(me))
        self.client.post("/api/v1/logout", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(self.client.get("/api/v1/me", headers={"Authorization": f"Bearer {token}"}).status_code, 401)

    # --- the website served by the API itself (phones, through a tunnel) ---------------------------
    PAGE = {"base_url": "https://localhost", "headers": {"Origin": "https://localhost", "Sec-Fetch-Site": "same-origin"}}

    def page_login(self):
        return self.client.post("/api/v1/login", json={"username": "julius", "password": self.password}, **self.PAGE)

    def test_the_page_is_served(self):
        reply = self.client.get("/", base_url="https://localhost")
        self.assertEqual(reply.status_code, 200)
        self.assertIn(b"app.js", reply.data)
        self.assertIn("default-src 'self'", reply.headers["Content-Security-Policy"])
        self.assertEqual(self.client.get("/static/app.js", base_url="https://localhost").status_code, 200)
        self.assertEqual(self.client.get("/static/../server.py", base_url="https://localhost").status_code, 404)
        self.assertEqual(self.client.get("/api/v1/health").headers["Content-Security-Policy"],
                         "default-src 'none'; frame-ancestors 'none'")

    def test_page_login_keeps_the_session_in_a_cookie_scripts_cannot_read(self):
        reply = self.page_login()
        self.assertEqual(reply.status_code, 200)
        self.assertNotIn("token", reply.get_json())                   # the page's scripts never see it
        cookie = reply.headers["Set-Cookie"]
        for part in ("HttpOnly", "Secure", "SameSite=Strict", "dramatizer_session="):
            self.assertIn(part, cookie)
        self.assertEqual(self.client.get("/connection", base_url="https://localhost").get_json()["name"], "Julius")
        self.assertEqual(self.client.get("/api/v1/me", base_url="https://localhost").get_json()["account"]["username"],
                         "julius")
        self.client.post("/api/v1/logout", json={}, **self.PAGE)
        self.assertEqual(self.client.get("/api/v1/me", base_url="https://localhost").status_code, 401)
        self.assertIsNone(self.client.get("/connection", base_url="https://localhost").get_json()["name"])

    def test_scripts_still_get_a_token(self):
        self.assertIn("token", self.login().get_json())              # no Origin: a script or the client app

    def test_another_website_cannot_use_the_cookie(self):
        self.page_login()
        scores = {"scores": {"Julius": {"genome_01": 4}}}
        evil = {"base_url": "https://localhost", "headers": {"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"}}
        self.assertEqual(self.client.post("/api/v1/genome/searches/r1/ratings", json=scores, **evil).status_code, 403)
        no_origin = {"base_url": "https://localhost"}
        self.assertEqual(self.client.post("/api/v1/genome/searches/r1/ratings", json=scores, **no_origin).status_code, 403)
        self.assertEqual(self.client.post("/api/v1/genome/searches/r1/ratings", json=scores, **self.PAGE).status_code, 200)

    # --- renders: everyone's spoken and rendered lines ------------------------------------------
    def test_renders_list_audio_and_only_mine(self):
        from types import SimpleNamespace
        folder = self.dir / "renders"
        with mock.patch.object(self.server, "GENOME_RENDERS", folder):
            clip = SimpleNamespace(dataset="VCTK", speaker="p250", clip_id="p250_001")
            spoken = self.dir / "elsewhere" / "x.wav"
            spoken.parent.mkdir()
            spoken.write_bytes(b"RIFF....WAVE")
            kept = self.server.keep_render(spoken, clip, "Hello there.", "Julius", 4.2, seed=0)
            self.assertEqual(kept.parent, folder)                       # a /speak line is moved in with a note
            self.assertFalse(spoken.exists())
            (folder / "VCTK_p250_genome_1.wav").write_bytes(b"RIFF....WAVE")
            (folder / "VCTK_p250_genome_1.json").write_text(json.dumps({
                "voice": "VCTK/p250", "text": "Well then.", "genome": "Sigh", "genes": [{"feature": "sigh"}],
                "markup": "Well [sigh] then.", "by": "Ash", "made": "2030-01-01 10:00:00"}), encoding="utf-8")
            auth = {"Authorization": "Bearer " + self.login().get_json()["token"]}
            every = self.client.get("/api/v1/renders", headers=auth).get_json()
            self.assertEqual(every["total"], 2)
            self.assertEqual(every["renders"][0]["genome"], "Sigh")       # newest first
            mine = self.client.get("/api/v1/renders?mine=1", headers=auth).get_json()["renders"]
            self.assertEqual([r["by"] for r in mine], ["Julius"])
            found = self.client.get("/api/v1/renders?q=sigh", headers=auth).get_json()["renders"]
            self.assertEqual([r["genome"] for r in found], ["Sigh"])
            audio = self.client.get(f"/api/v1/renders/{every['renders'][1]['id']}/audio", headers=auth)
            self.assertEqual(audio.status_code, 200)
            self.assertTrue(audio.data.startswith(b"RIFF"))
            audio.close()                                               # lets Windows delete the folder
            self.assertEqual(self.client.get("/api/v1/renders/..%2Fsecret/audio", headers=auth).status_code, 404)
            self.assertEqual(self.client.get("/api/v1/renders").status_code, 401)

    def test_rate_only_your_own_renders_kept_apart(self):
        folder = self.dir / "renders2"
        folder.mkdir()
        for stem, by in (("VCTK_p250_genome_a", "Julius"), ("VCTK_p250_genome_b", "Ash")):
            (folder / f"{stem}.wav").write_bytes(b"RIFF....WAVE")
            (folder / f"{stem}.json").write_text(json.dumps({"voice": "VCTK/p250", "text": "Hi.", "by": by,
                                                            "made": "2030-01-01 10:00:00"}), encoding="utf-8")
        scores = self.dir / "render_ratings.json"
        with mock.patch.object(self.server, "GENOME_RENDERS", folder), mock.patch.object(self.server, "RENDER_RATINGS", scores):
            auth = {"Authorization": "Bearer " + self.login().get_json()["token"]}
            rate = lambda stem, score: self.client.post(f"/api/v1/renders/{stem}/rating", headers=auth, json={"score": score})
            self.assertEqual(rate("VCTK_p250_genome_a", 4).status_code, 200)
            self.assertEqual(rate("VCTK_p250_genome_b", 4).status_code, 403)     # someone else's
            self.assertEqual(rate("VCTK_p250_genome_a", 9).status_code, 400)
            self.assertEqual(json.loads(scores.read_text(encoding="utf-8")), {"VCTK_p250_genome_a": {"Julius": 4}})
            listed = {r["id"]: r for r in self.client.get("/api/v1/renders", headers=auth).get_json()["renders"]}
            self.assertEqual((listed["VCTK_p250_genome_a"]["my_score"], listed["VCTK_p250_genome_a"]["mine"]), (4, True))
            self.assertFalse(listed["VCTK_p250_genome_b"]["mine"])
            self.assertEqual(rate("VCTK_p250_genome_a", None).status_code, 200)  # cleared

    def test_private_genomes_are_seen_by_their_maker_only(self):
        from types import SimpleNamespace
        kept = self.dir / "private_genomes.json"
        clip = SimpleNamespace(dataset="VCTK", speaker="p250", clip_id="p250_001")
        with mock.patch.object(self.server, "PRIVATE_GENOMES", kept):
            self.server.keep_private("Julius", "My sigh", [{"feature": "sigh", "position": 0.5}], "Hi.", clip)
            self.server.keep_private("Julius", "My sigh", [{"feature": "laugh", "position": 0.2}], "Hi.", clip)  # replaced
            self.server.keep_private("Rama", "Rama's", [{"feature": "sigh", "position": 0.5}], "Hi.", clip)
            auth = {"Authorization": "Bearer " + self.login().get_json()["token"]}
            genomes = self.client.get("/api/v1/genome/library", headers=auth).get_json()["genomes"]
            mine = [g for g in genomes if g["source"] == "private"]
            self.assertEqual([(g["name"], g["genes"][0]["feature"], g["group"]) for g in mine], [("My sigh", "laugh", "My genomes")])
            self.assertEqual(genomes[0]["source"], "private")                    # yours first
            others = [g for g in genomes if g["source"] != "private"]
            self.assertEqual([g["group"] for g in others], sorted([g["group"] for g in others],
                                                                    key=lambda x: (x == "Made by hand", str(x))))

    def test_listener_rates_only_as_themself_and_cannot_search(self):
        auth = {"Authorization": "Bearer " + self.login().get_json()["token"]}
        other = self.client.post("/api/v1/genome/searches/r1/ratings", headers=auth,
                                 json={"scores": {"Rama": {"genome_01": 4}}})
        self.assertEqual(other.status_code, 403)
        own = self.client.post("/api/v1/genome/searches/r1/ratings", headers=auth,
                               json={"scores": {"Julius": {"baseline": 2.5, "genome_01": 4, "genome_02": 1.5}}})
        self.assertEqual(own.status_code, 200, own.get_json())
        search = self.client.post("/api/v1/genome/searches", headers=auth, json={})
        self.assertEqual(search.status_code, 403)

    def test_listening_pile_ranking(self):
        auth = {"Authorization": "Bearer " + self.login().get_json()["token"]}
        self.client.post("/api/v1/genome/searches/r1/ratings", headers=auth,
                         json={"scores": {"Julius": {"baseline": 2.5, "genome_01": 4, "genome_03": 1.5}}})
        piles = self.client.get("/api/v1/genome/listening", headers=auth).get_json()["piles"]
        self.assertEqual(piles, {"test-batch": [1, 2]})
        data = self.client.get("/api/v1/genome/listening?batch=test-batch&pile=1", headers=auth).get_json()
        self.assertEqual([r["id"] for r in data["runs"]], ["r1", "r2"])
        self.assertEqual([r["mine"] for r in data["runs"]], [3, 0])
        self.assertEqual(len(data["clips"]), 8)
        first = next(c for c in data["clips"] if c["run"] == "r1" and c["genome"] == 1)
        self.assertEqual((first["mine"], first["mine_above_baseline"]), (4.0, 1.5))
        self.assertEqual(self.client.get("/api/v1/genome/listening?batch=test-batch&pile=9", headers=auth).status_code, 404)

    def test_assigned_listener_sees_and_rates_only_their_pile(self):
        accounts.assign("julius", "test-batch", 2)
        auth = {"Authorization": "Bearer " + self.login().get_json()["token"]}
        data = self.client.get("/api/v1/genome/listening", headers=auth).get_json()
        self.assertEqual((data["piles"], data["assigned"]), ({"test-batch": [2]}, ["test-batch|2"]))
        self.assertEqual(self.client.get("/api/v1/genome/listening?batch=test-batch&pile=1", headers=auth).status_code, 404)
        other = self.client.post("/api/v1/genome/searches/r1/ratings", headers=auth,
                                 json={"scores": {"Julius": {"genome_01": 4}}})
        self.assertEqual(other.status_code, 404)      # another pile is not even shown to exist
        own = self.client.post("/api/v1/genome/searches/r3/ratings", headers=auth,
                               json={"scores": {"Julius": {"genome_01": 4}}})
        self.assertEqual(own.status_code, 200)

    def test_scores_are_kept_with_who_and_when(self):
        auth = {"Authorization": "Bearer " + self.login().get_json()["token"]}
        url = "/api/v1/genome/searches/r1/ratings"
        reply = self.client.post(url, headers=auth, json={"scores": {"Julius": {"baseline": 2, "genome_01": 4}}}).get_json()
        self.assertEqual(reply["stored"]["Julius"], {"baseline": 2.0, "genome_01": 4.0})     # read back from disk
        self.assertTrue(reply["rated_on"]["Julius"]["genome_01"])
        self.client.post(url, headers=auth, json={"scores": {"Julius": {"genome_01": 4}}})   # the same again
        self.client.post(url, headers=auth, json={"scores": {"Julius": {"genome_01": 3.5}}})  # changed

        given = json.loads((self.dir / "runs" / "r1" / "ratings.json").read_text(encoding="utf-8"))["given"]["Julius"]
        self.assertEqual((given["genome_01"]["score"], given["genome_01"]["account"], given["genome_01"]["changes"]),
                         (3.5, "julius", 1))
        log = [json.loads(line) for line in (self.dir / "mos.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(log), 3)                                   # the repeat added no line
        self.assertEqual((log[-1]["previous"], log[-1]["score"], log[-1]["account"], log[-1]["pile"]), (4.0, 3.5, "julius", 1))

        import csv
        with open(self.dir / "who_rated_what.csv", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([(r["rater"], r["account"], r["clip"], r["score"]) for r in rows],
                         [("Julius", "julius", "baseline", "2.0"), ("Julius", "julius", "genome_01", "3.5")])
        self.assertEqual(rows[1]["above_plain_line"], "1.5")
        self.assertTrue(rows[1]["rated_on"].startswith("20"))
        who = self.ratings.who_summary(self.dir / "runs")
        self.assertEqual(who["Julius"]["piles"], {"test-batch pile 1": [2, 8]})

        page = self.client.get("/api/v1/genome/listening?batch=test-batch&pile=1", headers=auth).get_json()
        clip = next(c for c in page["clips"] if c["run"] == "r1" and c["genome"] == 1)
        self.assertEqual(clip["mine_on"], given["genome_01"]["time"])

    def test_audit_log_and_no_secrets_in_it(self):
        self.login("wrong password!!")
        auth = {"Authorization": "Bearer " + self.login().get_json()["token"]}
        self.client.post("/api/v1/genome/searches/r1/ratings", headers=auth, json={"scores": {"Julius": {"genome_01": 4}}})
        self.client.post("/api/v1/logout", headers=auth)
        text = (self.dir / "audit.jsonl").read_text(encoding="utf-8")
        events = [json.loads(line)["event"] for line in text.splitlines()]
        self.assertEqual(events, ["login refused", "login", "rated", "logout"])
        self.assertNotIn(self.password, text)
        self.assertNotIn("wrong password", text)
        self.assertNotIn(auth["Authorization"][7:], text)

    def test_many_failed_logins_hold_logins_back(self):
        for i in range(31):
            self.client.post("/api/v1/login", json={"username": f"guess{i}", "password": "wrong password!!"})
        self.assertEqual(self.login().status_code, 429)
        self.server._failed_logins.clear()
        self.assertEqual(self.login().status_code, 200)

    def test_strict_headers(self):
        reply = self.client.get("/api/v1/health")
        for header in ("X-Frame-Options", "Content-Security-Policy", "Referrer-Policy", "X-Content-Type-Options"):
            self.assertIn(header, reply.headers)

    def test_admin_sees_every_pile(self):
        accounts.set_role("julius", "admin")
        accounts.assign("julius", "test-batch", 2)
        auth = {"Authorization": "Bearer " + self.login().get_json()["token"]}
        self.assertEqual(self.client.get("/api/v1/genome/listening", headers=auth).get_json()["piles"], {"test-batch": [1, 2]})

    def test_listener_sees_only_their_pile_and_rates_blind(self):
        accounts.assign("julius", "test-batch", 1)
        self.ratings.save_scores(self.dir / "runs" / "r1", {"Ash": {"genome_01": 5, "baseline": 1}})   # someone else
        auth = {"Authorization": "Bearer " + self.login().get_json()["token"]}
        get = lambda path: self.client.get("/api/v1" + path, headers=auth)
        self.assertEqual([s["id"] for s in get("/genome/searches").get_json()["searches"]], ["r1", "r2"])
        self.assertEqual(get("/genome/searches/r3").status_code, 404)                 # pile 2: not theirs
        self.assertEqual(get("/genome/searches/r1").get_json()["ratings"]["scores"], {})   # Ash's scores hidden
        page = get("/genome/listening?batch=test-batch&pile=1").get_json()
        self.assertFalse(page["sees_others"])
        self.assertNotIn("Ash", json.dumps(page))
        self.assertTrue(all(c["mos"] is None for c in page["clips"]))
        reply = self.client.post("/api/v1/genome/searches/r1/ratings", headers=auth,
                                 json={"scores": {"Julius": {"genome_01": 2}}}).get_json()
        self.assertEqual(reply["raters"], ["Julius"])
        self.assertNotIn("Ash", json.dumps(reply))

    def test_artifact_answer_is_kept_with_the_score(self):
        auth = {"Authorization": "Bearer " + self.login().get_json()["token"]}
        url = "/api/v1/genome/searches/r1/ratings"
        reply = self.client.post(url, headers=auth, json={"scores": {"Julius": {"baseline": 3, "genome_01": 2}},
                                                          "artifacts": {"Julius": {"baseline": False, "genome_01": True}}})
        self.assertEqual(reply.status_code, 200, reply.get_json())
        self.assertEqual(reply.get_json()["stored_artifacts"], {"Julius": {"baseline": False, "genome_01": True}})

        # Changed on its own, for a clip already scored: kept, and logged with the answer before
        reply = self.client.post(url, headers=auth, json={"artifacts": {"Julius": {"genome_01": False}}})
        self.assertEqual(reply.status_code, 200, reply.get_json())
        self.assertEqual(reply.get_json()["stored"]["Julius"]["genome_01"], 2.0)
        log = [json.loads(line) for line in (self.dir / "mos.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual((log[-1]["clip"], log[-1]["score"], log[-1]["artifact"], log[-1]["previous_artifact"]),
                         ("genome_01", 2.0, False, True))
        given = json.loads((self.dir / "runs" / "r1" / "ratings.json").read_text(encoding="utf-8"))["given"]["Julius"]
        self.assertEqual((given["genome_01"]["artifact"], given["genome_01"]["changes"]), (False, 1))

        # Not for a clip with no score yet, and only true or false
        self.assertEqual(self.client.post(url, headers=auth, json={"artifacts": {"Julius": {"genome_02": True}}}).status_code, 400)
        self.assertEqual(self.client.post(url, headers=auth, json={"artifacts": {"Julius": {"genome_01": "maybe"}}}).status_code, 400)
        self.assertEqual(self.client.post(url, headers=auth, json={"artifacts": {"Rama": {"genome_01": True}}}).status_code, 403)

        self.client.post(url, headers=auth, json={"scores": {"Julius": {"genome_03": 4}}, "artifacts": {"Julius": {"genome_03": True}}})
        import csv
        with open(self.dir / "who_rated_what.csv", encoding="utf-8-sig") as f:
            rows = {r["clip"]: r["artifact"] for r in csv.DictReader(f)}
        self.assertEqual(rows, {"baseline": "no", "genome_01": "no", "genome_03": "yes"})
        page = self.client.get("/api/v1/genome/listening?batch=test-batch&pile=1", headers=auth).get_json()
        mine = {c["clip"]: c["mine_artifact"] for c in page["clips"] if c["run"] == "r1"}
        self.assertEqual(mine, {"baseline": False, "genome_01": False, "genome_02": None, "genome_03": True})

    def test_old_scores_have_no_artifact_answer_and_others_stay_hidden(self):
        accounts.assign("julius", "test-batch", 1)
        self.ratings.save_scores(self.dir / "runs" / "r1", {"Ash": {"genome_01": 5}}, artifacts={"Ash": {"genome_01": True}})
        self.ratings.save_scores(self.dir / "runs" / "r1", {"Julius": {"genome_02": 3}})        # before the box existed
        rating = self.ratings.load(self.dir / "runs" / "r1")
        self.assertIsNone(self.ratings.artifact_of(rating, "Julius", "genome_02"))
        self.assertEqual(self.ratings.mos(rating, "genome_01")["artifact_heard"], 1)
        auth = {"Authorization": "Bearer " + self.login().get_json()["token"]}
        shown = self.client.get("/api/v1/genome/searches/r1", headers=auth).get_json()["ratings"]
        self.assertEqual(shown["artifacts"], {"Julius": {}})                   # Ash's answer is not shown


if __name__ == "__main__":
    unittest.main()

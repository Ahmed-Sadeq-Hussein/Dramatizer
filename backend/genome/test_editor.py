"""Tests of the genome studio: checking genes, previews, saving, the library, and its API routes. No models, no GPU.

    python -m unittest backend.genome.test_editor
"""
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))    # run as a file by main_backend.py test

from backend.api import accounts  # noqa: E402
from backend.api.test_accounts import Temporary, fake_run  # noqa: E402
from backend.genome import editor, ratings, search  # noqa: E402

LINE = "Well, you really did it this time, and I honestly didn't think you had it in you."


class Genes(unittest.TestCase):
    def test_good_genes_are_kept_and_rounded(self):
        genes = editor.clean_genes([{"feature": "laughter", "position": 0.12345, "extra": "dropped"},
                                    {"feature": "pause", "position": 1, "kind": "realization"},
                                    {"feature": "speed", "rate": "1.234"}])
        self.assertEqual(genes, [{"feature": "laughter", "position": 0.123},
                                 {"feature": "pause", "position": 1.0, "kind": "realization"},
                                 {"feature": "speed", "rate": 1.23}])

    def test_a_pause_without_a_kind_is_dramatic(self):
        self.assertEqual(editor.clean_genes([{"feature": "pause", "position": 0.5}])[0]["kind"], "dramatic")

    def test_bad_genes_are_refused(self):
        for bad in ([], "laughter", [{"feature": "shout", "position": 0.5}], [{"feature": "sigh", "position": 1.5}],
                    [{"feature": "sigh", "position": "x"}], [{"feature": "sigh", "position": float("nan")}],
                    [{"feature": "sigh", "position": True}], [{"feature": "speed", "rate": 3}],
                    [{"feature": "pause", "position": 0.5, "kind": "long"}],
                    [{"feature": "speed", "rate": 1.1}, {"feature": "speed", "rate": 0.9}],
                    [{"feature": "sigh", "position": 0.5}] * (editor.MAX_GENES + 1), [None]):
            with self.assertRaises(ValueError, msg=bad):
                editor.clean_genes(bad)

    def test_no_genes_only_for_a_render(self):
        self.assertEqual(editor.clean_genes([], allow_empty=True), [])


class Preview(unittest.TestCase):
    def test_genes_on_a_line(self):
        shown = editor.preview(LINE, [{"feature": "emphasis", "position": 0.0}, {"feature": "speed", "rate": 1.2}])
        self.assertEqual(shown["described"], ["emphasis on 'Well'", "speed 1.2"])
        self.assertEqual(shown["speed"], 1.2)
        self.assertIn("Well", shown["markup"])
        self.assertNotEqual(shown["markup"], LINE)

    def test_no_genes_is_the_plain_line(self):
        self.assertEqual(editor.preview(LINE, [])["markup"], LINE)

    def test_a_line_without_words_is_refused(self):
        with self.assertRaises(ValueError):
            editor.preview("...", [])


class Library(Temporary):
    def setUp(self):
        super().setUp()
        for p in (mock.patch.object(ratings, "NAMED_PATH", self.dir / "named.json"),
                  mock.patch.object(ratings, "LOG_PATH", self.dir / "mos.jsonl")):
            p.start()
            self.addCleanup(p.stop)
        self.runs = self.dir / "runs"
        fake_run(self.runs / "r1", "r1", 1)

    def test_save_list_and_remove(self):
        genes = [{"feature": "sigh", "position": 0.2}]
        entry = editor.save("Sad start", genes, named_by="Julius", text=LINE, based_on="r1 genome 2")
        self.assertTrue(entry["edited"])
        self.assertEqual((entry["run"], entry["markup"] is not None), (None, True))
        with self.assertRaises(ValueError):
            editor.save("sad START", genes)                         # names are unique, ignoring case

        items = editor.library(search.list_runs(self.runs))
        self.assertEqual(items[0]["key"], "named:Sad start")
        self.assertEqual(items[0]["source"], "edited")
        self.assertEqual([i["key"] for i in items[1:]], ["search:r1/1", "search:r1/2", "search:r1/3"])

        editor.remove("sad start")
        self.assertEqual(ratings.load_named()["genomes"], [])
        with self.assertRaises(ValueError):
            editor.remove("Sad start")

    def test_a_name_from_a_search_is_not_taken_by_hand_and_not_deleted_here(self):
        ratings.pick(self.runs / "r1", 1, "Proud")
        with self.assertRaises(ValueError):
            editor.save("proud", [{"feature": "sigh", "position": 0.2}])
        with self.assertRaises(ValueError):
            editor.remove("Proud")

    def test_mos_only_when_asked(self):
        ratings.save_scores(self.runs / "r1", {"Ash": {"genome_01": 4}})
        items = {i["key"]: i for i in editor.library(search.list_runs(self.runs))}
        self.assertEqual(items["search:r1/1"]["mos"], 4.0)
        blind = {i["key"]: i for i in editor.library(search.list_runs(self.runs), with_mos=False)}
        self.assertIsNone(blind["search:r1/1"]["mos"])


class Routes(Library):
    def setUp(self):
        super().setUp()
        from backend.api import server

        self.server = server
        for p in (mock.patch.object(search, "RUNS_DIR", self.runs),
                  mock.patch.object(server.time, "sleep", lambda s: None),
                  mock.patch.object(server, "API_DIR", self.dir),
                  mock.patch.object(server, "ACCESS_LOG", self.dir / "access.log"),
                  mock.patch.object(server, "AUDIT_LOG", self.dir / "audit.jsonl"),
                  mock.patch.object(server, "_failed_logins", [])):
            p.start()
            self.addCleanup(p.stop)
        self.client = server.app.test_client()

    def auth(self, role):
        _, password = accounts.new_account(role, role.title(), role=role)
        token = self.client.post("/api/v1/login", json={"username": role, "password": password}).get_json()["token"]
        return {"Authorization": f"Bearer {token}"}

    def test_researcher_previews_saves_and_lists(self):
        auth = self.auth("researcher")
        genes = [{"feature": "laughter", "position": 0.5}]
        reply = self.client.post("/api/v1/genome/preview", headers=auth, json={"text": LINE, "genes": genes})
        self.assertEqual(reply.status_code, 200, reply.get_json())
        self.assertEqual(len(reply.get_json()["preview"]["described"]), 1)
        bad = self.client.post("/api/v1/genome/preview", headers=auth, json={"text": LINE, "genes": [{"feature": "shout"}]})
        self.assertEqual(bad.status_code, 400)

        saved = self.client.post("/api/v1/genome/edited", headers=auth, json={"name": "Giggle", "genes": genes, "text": LINE})
        self.assertEqual(saved.status_code, 200, saved.get_json())
        self.assertEqual(saved.get_json()["named"]["named_by"], "Researcher")
        library = self.client.get("/api/v1/genome/library", headers=auth).get_json()
        self.assertEqual(library["genomes"][0]["name"], "Giggle")
        self.assertIn("speed", library["options"]["features"])
        self.assertIn("genome saved", (self.dir / "audit.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(self.client.post("/api/v1/genome/edited/remove", headers=auth, json={"name": "Giggle"}).status_code, 200)

    def test_render_queues_a_job_with_the_genome(self):
        auth = self.auth("researcher")
        clip = mock.Mock(dataset="VCTK", clip_id="p250_001", speaker="p250", text="Hello.", details={})
        queued = []
        with mock.patch.object(self.server, "clip_of", lambda d, c: clip), \
             mock.patch.object(self.server, "queue_job", lambda kind, owner, work, about: queued.append(about) or
                               {"id": "0" * 32, "kind": kind, "owner": owner, "status": "queued", "about": about,
                                "created": "", "result": None}):
            reply = self.client.post("/api/v1/genome/render", headers=auth,
                                     json={"dataset": "VCTK", "clip_id": "p250_001", "text": LINE, "seed": 7,
                                           "genes": [{"feature": "sigh", "position": 0}]})
            self.assertEqual(reply.status_code, 202, reply.get_json())
            self.assertEqual((queued[0]["seed"], queued[0]["genes"]), (7, [{"feature": "sigh", "position": 0.0}]))
            plain = self.client.post("/api/v1/genome/render", headers=auth,
                                     json={"dataset": "VCTK", "clip_id": "p250_001", "text": LINE, "seed": 7, "genes": []})
            self.assertEqual(plain.get_json()["preview"]["markup"], LINE)
            bad = self.client.post("/api/v1/genome/render", headers=auth,
                                   json={"dataset": "VCTK", "clip_id": "p250_001", "text": LINE, "genes": "sigh"})
            self.assertEqual(bad.status_code, 400)

    def test_a_listener_may_not_save_or_render_and_sees_no_mos(self):
        ratings.save_scores(self.runs / "r1", {"Ash": {"genome_01": 4}})
        auth = self.auth("listener")
        genes = [{"feature": "sigh", "position": 0.2}]
        self.assertEqual(self.client.post("/api/v1/genome/edited", headers=auth, json={"name": "X", "genes": genes}).status_code, 403)
        self.assertEqual(self.client.post("/api/v1/genome/render", headers=auth,
                                          json={"dataset": "VCTK", "clip_id": "x", "text": LINE, "genes": genes}).status_code, 403)
        library = self.client.get("/api/v1/genome/library", headers=auth).get_json()
        self.assertTrue(all(g["mos"] is None for g in library["genomes"]))
        self.assertNotIn("Ash", json.dumps(library))


if __name__ == "__main__":
    unittest.main()

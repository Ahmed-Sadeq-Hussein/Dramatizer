"""Tests of storage.py: the database of accounts, sessions and ratings (no models, no real data).

    python backend/test_storage.py
"""
import json
import sqlite3
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))    # run as a file by main_backend.py test

from backend import storage  # noqa: E402
from backend.api import accounts  # noqa: E402
from backend.api.test_accounts import fake_run  # noqa: E402
from backend.genome import ratings, search  # noqa: E402


class Database(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(folder.cleanup)
        self.dir = Path(folder.name)
        for p in (mock.patch.object(storage, "DB_PATH", self.dir / "test.db"),
                  mock.patch.object(accounts, "ACCOUNTS_PATH", self.dir / "accounts.json"),
                  mock.patch.object(accounts, "SESSIONS_PATH", self.dir / "sessions.json"),
                  mock.patch.object(accounts, "ROUNDS", 1000),
                  mock.patch.object(search, "RUNS_DIR", self.dir / "runs"),
                  mock.patch.object(ratings, "NAMED_PATH", self.dir / "named.json"),
                  mock.patch.object(ratings, "LOG_PATH", self.dir / "mos.jsonl")):   # who_rated_what.csv goes beside it
            p.start()
            self.addCleanup(p.stop)

    # --- the schema -------------------------------------------------------------------------------
    def test_a_new_database_gets_every_migration(self):
        found = storage.counts()
        self.assertEqual(found["schema_version"], storage.SCHEMA_VERSION)
        for table in ("accounts", "account_piles", "sessions", "rating_runs", "ratings", "rating_log", "meta"):
            self.assertEqual(found["tables"][table], 0)

    def test_a_newer_database_is_refused(self):
        db = sqlite3.connect(self.dir / "newer.db")
        db.execute(f"PRAGMA user_version = {storage.SCHEMA_VERSION + 1}")
        db.close()
        with self.assertRaisesRegex(RuntimeError, "newer than this code"):
            with storage.connect(self.dir / "newer.db"):
                pass

    def test_an_error_rolls_the_whole_change_back(self):
        with self.assertRaises(ZeroDivisionError):
            with storage.connect(write=True) as db:
                storage.set_meta(db, "half", "done")
                1 / 0
        with storage.connect() as db:
            self.assertIsNone(storage.get_meta(db, "half"))

    def test_a_change_cannot_start_inside_a_read(self):
        with storage.connect():
            with self.assertRaises(RuntimeError):
                with storage.connect(write=True):
                    pass

    def test_constraints_hold(self):
        with self.assertRaises(sqlite3.IntegrityError):              # a session needs a real account
            with storage.connect(write=True) as db:
                db.execute("INSERT INTO sessions VALUES ('h', 'nobody', 'x', 'y')")
        with self.assertRaises(sqlite3.IntegrityError):              # a score is 0 to 5
            with storage.connect(write=True) as db:
                db.execute("INSERT INTO rating_runs (run) VALUES ('r')")
                db.execute("INSERT INTO ratings (run, rater, clip, score) VALUES ('r', 'Ash', 'baseline', 7)")

    # --- accounts and sessions --------------------------------------------------------------------
    def test_two_writers_at_once_both_land(self):
        accounts.new_account("julius", "Julius")
        errors = []

        def assign(pile):
            try:
                accounts.assign("julius", "b", pile)
            except Exception as error:                  # noqa: BLE001 (reported below)
                errors.append(error)
        threads = [threading.Thread(target=assign, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertEqual(sorted(accounts.get("julius").assignments), sorted(f"b|{i}" for i in range(8)))

    def test_removing_an_account_ends_its_sessions(self):
        account, _ = accounts.new_account("julius", "Julius")
        token = accounts.open_session(account)
        accounts.remove("julius")
        self.assertIsNone(accounts.session_account(token))
        self.assertEqual(storage.counts()["tables"]["sessions"], 0)

    def test_a_wrong_password_is_counted_even_though_it_is_refused(self):
        accounts.new_account("julius", "Julius")
        with self.assertRaises(accounts.LoginRefused):
            accounts.authenticate("julius", "wrong password!!")
        self.assertEqual(accounts.get("julius").failed_logins, 1)

    def test_old_files_not_yet_imported_give_a_clear_message(self):
        (self.dir / "accounts.json").write_text('{"accounts": {}}', encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "storage migrate"):
            accounts.load()

    # --- ratings ----------------------------------------------------------------------------------
    def test_ratings_round_trip_and_the_folder_keeps_a_copy(self):
        fake_run(self.dir / "runs" / "r1", "r1", 1)
        folder = self.dir / "runs" / "r1"
        ratings.save_scores(folder, {"Ash": {"baseline": 3, "genome_01": 4.5}}, account="ash",
                            artifacts={"Ash": {"genome_01": True}})
        ratings.save_scores(folder, {"Ash": {"genome_01": 4}})
        stored = ratings.load(folder)
        self.assertEqual(stored["scores"], {"Ash": {"baseline": 3.0, "genome_01": 4.0}})
        self.assertEqual((stored["given"]["Ash"]["genome_01"]["changes"], stored["artifacts"]), (1, {"Ash": {"genome_01": True}}))
        copy = json.loads((folder / ratings.RATINGS_FILE).read_text(encoding="utf-8"))
        self.assertEqual(copy["scores"], stored["scores"])
        log = ratings.read_log()
        self.assertEqual([(e["clip"], e["score"], e["previous"]) for e in log],
                         [("baseline", 3.0, None), ("genome_01", 4.5, None), ("genome_01", 4.0, 4.5)])

    def test_the_rating_log_cannot_be_changed_or_deleted(self):
        with storage.connect(write=True) as db:
            ratings._log(db, [{"time": "t", "run": "r", "rater": "Ash", "clip": "baseline", "score": 3}])
        for sql in ("UPDATE rating_log SET score = 5", "DELETE FROM rating_log"):
            with self.subTest(sql), self.assertRaisesRegex(sqlite3.DatabaseError, "append-only"):
                with storage.connect(write=True) as db:
                    db.execute(sql)

    def test_a_run_copied_in_is_read_from_its_file_until_its_first_change(self):
        fake_run(self.dir / "runs" / "r2", "r2", 1)
        folder = self.dir / "runs" / "r2"
        doc = {"schema_version": 1, "run": "r2", "scale": [0, 5], "raters": ["Ash"], "scores": {"Ash": {"baseline": 2}},
               "picked": [], "updated": "2026-11-11 10:00:00",
               "given": {"Ash": {"baseline": {"first_given": "2026-11-11 10:00:00", "changes": 0, "score": 2,
                                              "time": "2026-11-11 10:00:00", "account": "ash", "submitted_by": "Ash"}}}}
        (folder / ratings.RATINGS_FILE).write_text(json.dumps(doc), encoding="utf-8")
        self.assertEqual(ratings.load(folder)["scores"], {"Ash": {"baseline": 2}})
        self.assertEqual(storage.counts()["tables"]["rating_runs"], 0)      # reading changes nothing
        ratings.save_scores(folder, {"Ash": {"genome_01": 5}})
        self.assertEqual(ratings.load(folder)["scores"], {"Ash": {"baseline": 2.0, "genome_01": 5.0}})

    # --- the one-time import and backups -----------------------------------------------------------
    def test_migrate_imports_once_checks_and_renames(self):
        fake_run(self.dir / "runs" / "r3", "r3", 1)
        folder = self.dir / "runs" / "r3"
        doc = {"schema_version": 1, "run": "r3", "scale": [0, 5], "raters": ["Ash", "Rama"],
               "scores": {"Ash": {"baseline": 3, "genome_01": 4}, "Rama": {"genome_01": 2.5}},
               "picked": [{"genome": 1, "name": "Proud", "time": "2026-11-11 11:00:00"}], "updated": "2026-11-11 11:00:00",
               "given": {r: {c: {"first_given": "t0", "changes": 0, "score": s, "time": "t1", "account": None,
                                 "submitted_by": r} for c, s in clips.items()}
                         for r, clips in {"Ash": {"baseline": 3, "genome_01": 4}, "Rama": {"genome_01": 2.5}}.items()},
               "artifacts": {"Rama": {"genome_01": False}}}
        doc["given"]["Rama"]["genome_01"]["artifact"] = False
        (folder / ratings.RATINGS_FILE).write_text(json.dumps(doc), encoding="utf-8")
        account = {"username": "ash", "display_name": "Ash", "role": "admin", "password": accounts.hash_password("x" * 12),
                   "disabled": False, "created": "c", "password_changed": "c", "last_login": None, "failed_logins": 0,
                   "locked_until": None, "assignments": [], "schema_version": 1}
        old = {"accounts": self.dir / "accounts.json", "sessions": self.dir / "sessions.json",
               "rating_log": self.dir / "mos.jsonl"}
        old["accounts"].write_text(json.dumps({"accounts": {"ash": account}}), encoding="utf-8")
        old["sessions"].write_text(json.dumps({"h1": {"username": "ash", "created": "2026-11-16 10:00:00",
                                                      "expires": "2099-01-01 00:00:00"},
                                               "h2": {"username": "ash", "created": "x", "expires": "2020-01-01 00:00:00"}}),
                                   encoding="utf-8")
        old["rating_log"].write_text(json.dumps({"time": "t", "run": "r3", "rater": "Ash", "clip": "baseline",
                                                 "score": 3}) + "\n", encoding="utf-8")
        done = storage.migrate(rating_folders=[folder], legacy=old, backup_folder=self.dir / "backups")
        self.assertEqual({k: done[k] for k in ("accounts", "sessions", "rating_log", "runs")},
                         {"accounts": 1, "sessions": 1, "rating_log": 1, "runs": 1})     # the expired session is dropped
        self.assertEqual(len(list(Path(done["backup"]).iterdir())), 4)                   # every file read, copied first
        self.assertTrue(all(not p.exists() and p.with_name(p.name + ".migrated").exists() for p in old.values()))
        self.assertEqual(ratings.load(folder)["picked"][0]["name"], "Proud")
        self.assertEqual(accounts.get("ash").role, "admin")
        again = storage.migrate(rating_folders=[folder], legacy=old, backup_folder=self.dir / "backups")
        self.assertEqual(again["runs"], 0)                                                # once only

    def test_backup_is_a_working_copy_and_old_ones_are_pruned(self):
        accounts.new_account("julius", "Julius")
        for _ in range(3):
            target = storage.backup(folder=self.dir / "backups", keep=2)
        copy = sqlite3.connect(target)
        self.assertEqual(copy.execute("SELECT username FROM accounts").fetchall(), [("julius",)])
        copy.close()
        self.assertLessEqual(len(list((self.dir / "backups").glob("*.db"))), 2)


if __name__ == "__main__":
    unittest.main()

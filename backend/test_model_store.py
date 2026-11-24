"""test_model_store.py | The model store: only files that match their pin go in, and a missing model stops the server.

    python backend/test_model_store.py

Hugging Face is replaced by a stand-in that lists files with their hashes, so this runs offline (and on CI).
"""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend import model_store as ms  # noqa: E402

LARGE = b"weights " * 1000
SMALL = b'{"hidden_size": 8}'
PIN = "a" * 40


def blob(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sibling(name, data, lfs=True):
    return SimpleNamespace(rfilename=name, size=len(data), blob_id=blob(data),
                           lfs={"sha256": hashlib.sha256(data).hexdigest()} if lfs else None)


class FakeHub:
    """model_info and list_repo_commits for one repo: the files at PIN are LARGE and SMALL."""

    def __init__(self, files=None):
        self.files = files or [sibling("model.safetensors", LARGE), sibling("config.json", SMALL, lfs=False),
                               sibling("README.md", b"readme", lfs=False)]

    def model_info(self, repo, revision, files_metadata):
        return SimpleNamespace(siblings=self.files if revision == PIN else
                               [sibling("model.safetensors", b"older"), sibling("config.json", SMALL, lfs=False)])

    def list_repo_commits(self, repo):
        return [SimpleNamespace(commit_id="b" * 40), SimpleNamespace(commit_id=PIN)]


class Store(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.local = self.root / "local" / "Tiny"
        self.local.mkdir(parents=True)
        (self.local / "model.safetensors").write_bytes(LARGE)
        (self.local / "config.json").write_bytes(SMALL)
        self.store = self.root / "store"
        self.spec = {"repo": "someone/Tiny", "revision": PIN, "folder": "llm/Tiny", "local": ("TINY_DIR", None),
                     "why": "a test"}
        for patch in (mock.patch.dict(ms.MODELS, {"Tiny": self.spec}, clear=True),
                      mock.patch.dict(os.environ, {"TINY_DIR": str(self.local), "MODELS_DIR": str(self.store)})):
            patch.start()
            self.addCleanup(patch.stop)
        self.downloads = []

    def download(self, name, spec, relative, here):
        self.downloads.append(relative)
        here.parent.mkdir(parents=True, exist_ok=True)
        here.write_bytes({"model.safetensors": LARGE, "config.json": SMALL}[relative])

    def fetch(self, hub=None):
        return ms.fetch(self.store, api=hub or FakeHub(), download=self.download, say=lambda *_: None)

    def test_a_matching_local_copy_is_linked_not_downloaded(self):
        data = self.fetch()
        self.assertEqual(self.downloads, [])
        entry = data["models"]["Tiny"]
        self.assertEqual(entry["revision"], PIN)
        self.assertEqual(set(entry["files"]), {"model.safetensors", "config.json"})     # README left out
        self.assertEqual((self.store / "llm/Tiny/model.safetensors").read_bytes(), LARGE)
        self.assertEqual(ms.check(self.store, full=True), [])

    def test_a_changed_local_file_is_downloaded_instead(self):
        (self.local / "model.safetensors").write_bytes(b"x" * len(LARGE))       # same size, other bytes
        self.fetch()
        self.assertEqual(self.downloads, ["model.safetensors"])
        self.assertEqual(ms.check(self.store, full=True), [])

    def test_a_bad_download_is_refused_and_removed(self):
        (self.local / "config.json").unlink()

        def bad(name, spec, relative, here):
            here.parent.mkdir(parents=True, exist_ok=True)
            here.write_bytes(b"not the config")

        with self.assertRaises(ms.StoreError):
            ms.fetch(self.store, api=FakeHub(), download=bad, say=lambda *_: None)
        self.assertFalse((self.store / "llm/Tiny/config.json").exists())

    def test_a_second_fetch_keeps_what_is_there(self):
        self.fetch()
        (self.local / "model.safetensors").unlink()
        self.fetch()
        self.assertEqual(self.downloads, [])

    def test_check_finds_missing_changed_and_unpinned(self):
        self.fetch()
        (self.store / "llm/Tiny/config.json").unlink()
        self.assertIn("Tiny/config.json is missing.", ms.check(self.store))
        self.fetch()
        with mock.patch.dict(self.spec, {"revision": "c" * 40}):
            self.assertTrue(any("the pin is" in p for p in ms.check(self.store)))
        with open(self.store / "llm/Tiny/model.safetensors", "r+b") as file:
            file.write(b"W")
        self.assertEqual(ms.check(self.store), [])                       # same size: only --full sees it
        self.assertTrue(any("changed" in p for p in ms.check(self.store, full=True)))

    def test_no_store_is_one_clear_problem(self):
        self.assertEqual(len(ms.check(self.root / "nothing")), 1)

    def test_the_server_will_not_start_without_its_models(self):
        with self.assertRaises(SystemExit) as caught:
            ms.check_at_start()
        self.assertIn("does not start", str(caught.exception))
        self.fetch()
        ms.check_at_start()                                              # complete: no error
        with mock.patch.dict(os.environ, {"MODELS_DIR": ""}):
            (self.store / "models.json").unlink()
            ms.check_at_start()                                          # no store in use: nothing checked

    def test_pin_finds_the_revision_the_local_copy_matches(self):
        self.assertEqual(ms.pin("Tiny", api=FakeHub()), PIN)
        (self.local / "config.json").write_bytes(b'{"hidden_size": 9}')
        with self.assertRaises(ms.StoreError):
            ms.pin("Tiny", api=FakeHub())

    def test_a_file_pinned_by_sha256_is_checked_too(self):
        weights = self.root / "cache" / "whisper"
        weights.mkdir(parents=True)
        (weights / "base.pt").write_bytes(LARGE)
        spec = {"url": "https://example.invalid/base.pt", "sha256": hashlib.sha256(LARGE).hexdigest(),
                "folder": "cache/whisper", "file": "base.pt", "local": ("HOME_CACHE", "whisper"), "why": "a test"}
        with mock.patch.dict(ms.MODELS, {"whisper-base": spec}, clear=True), \
                mock.patch.dict(os.environ, {"XDG_CACHE_HOME": str(self.root / "cache")}):
            data = self.fetch()
            self.assertEqual(data["models"]["whisper-base"]["revision"], spec["sha256"])
            self.assertEqual(self.downloads, [])
            with mock.patch.dict(spec, {"sha256": "0" * 64}):
                self.assertTrue(any("the pin is" in p for p in ms.check(self.store)))

    def test_a_model_pinned_file_by_file_is_linked_from_the_modelscope_cache(self):
        hub = self.root / "modelscope"
        for relative, data in (("en/tn/tagger.fst", LARGE), ("zh/tn/tagger.fst", SMALL)):
            (hub / "someone/wetext" / relative).parent.mkdir(parents=True, exist_ok=True)
            (hub / "someone/wetext" / relative).write_bytes(data)
        spec = {"modelscope": "someone/wetext", "folder": "cache/wetext", "local": ("MODELSCOPE_CACHE", "someone/wetext"),
                "files": {"en/tn/tagger.fst": {"sha256": hashlib.sha256(LARGE).hexdigest(), "commit": "c" * 40},
                          "zh/tn/tagger.fst": {"sha256": hashlib.sha256(SMALL).hexdigest(), "commit": "d" * 40}},
                "why": "a test"}
        with mock.patch.dict(ms.MODELS, {"wetext": spec}, clear=True), \
                mock.patch.dict(os.environ, {"MODELSCOPE_CACHE": str(hub)}):
            data = self.fetch()
            entry = data["models"]["wetext"]
            self.assertEqual(self.downloads, [])
            self.assertEqual(entry["source"], "modelscope")
            self.assertEqual(entry["revision"], ms.pinned_revision(spec))
            self.assertEqual(entry["commits"]["zh/tn/tagger.fst"], "d" * 40)
            self.assertEqual(ms.check(self.store, full=True), [])
            with mock.patch.dict(spec["files"], {"zh/tn/tagger.fst": {"sha256": "0" * 64, "commit": "d" * 40}}):
                self.assertTrue(any("the pin is" in p for p in ms.check(self.store)))   # a new pin changes the revision

    def test_wetext_reads_the_store_and_never_downloads(self):
        from backend import model

        calls = []
        fake = SimpleNamespace(snapshot_download=lambda *a, **k: calls.append(a) or "downloaded")
        with mock.patch.dict(sys.modules, {"wetext": SimpleNamespace(wetext=fake), "wetext.wetext": fake}), \
                mock.patch.dict(os.environ, {"WETEXT_DIR": str(self.store / "cache/wetext")}):
            model._wetext_from_cache()
            self.assertEqual(fake.snapshot_download("pengzhendong/wetext"), str(self.store / "cache/wetext"))
        self.assertEqual(calls, [])

    def test_release_info_reads_the_store(self):
        from backend import release_info

        self.fetch()
        self.assertEqual(release_info.models(self.store), {"Tiny": {"repo": "someone/Tiny", "revision": PIN}})
        self.assertEqual(json.loads((self.store / "models.json").read_text())["models"]["Tiny"]["folder"], "llm/Tiny")

    def test_the_settings_point_every_loader_at_the_store(self):
        lines = dict(line.split("=", 1) for line in ms.settings_for(self.store))
        self.assertEqual(Path(lines["LLM_LIBRARY_DIR"]), self.store / "llm")
        self.assertEqual(Path(lines["TORCH_HOME"]), self.store / "cache" / "torch")


if __name__ == "__main__":
    unittest.main()

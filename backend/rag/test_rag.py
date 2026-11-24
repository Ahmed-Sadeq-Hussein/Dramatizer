"""Tests of the evidence corpus and the BM25 baseline (no models, no real ratings).

    python -m unittest backend.rag.test_rag
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))    # run as a file by main_backend.py test

from backend.rag import baseline, corpus  # noqa: E402


def record(id_, text, kind="clip", revision="2026-11-11 10:00:00"):
    return {"id": id_, "kind": kind, "revision": revision, "title": id_, "text": text, "fields": {}}


def run(voice="CREMA-D/1014", run_id="mos-diverse-v1_p1_CREMA-D_1014"):
    genes = [{"feature": "vocalized_noise", "position": 0.2}, {"feature": "pause", "position": 0.5, "kind": "dramatic"}]
    return {"id": run_id, "batch": "mos-diverse-v1", "pile": 1, "voice": voice, "text": "Well, you really did it.",
            "genomes": [{"number": 1, "genes": genes, "described": ["vocalized noise after 'Well'", "dramatic pause after 'did'"],
                         "speed": 1.0, "verdict": "fine", "markup": "Well [vocalized_noise] ..."},
                        {"number": 2, "genes": genes[:1], "described": ["vocalized noise after 'Well'"],
                         "speed": 1.0, "verdict": "fine", "markup": "Well [vocalized_noise]"}]}


def row(clip, genome, score, above, rated_on="2026-11-11 10:00:00", artifact=""):
    return {"rater": "Ash", "account": "ash", "batch": "mos-diverse-v1", "pile": "1", "voice": "CREMA-D/1014",
            "run": "mos-diverse-v1_p1_CREMA-D_1014", "clip": clip, "genome": genome, "score": score, "artifact": artifact,
            "above_plain_line": above, "rated_on": rated_on, "first_given": rated_on, "changes": "0", "verdict": "fine",
            "markup": "", "text": ""}



_scratch = None


def setUpModule():
    """Every test here reads ratings through a temporary database, never the real one."""
    global _scratch, _patch
    from backend import storage
    _scratch = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    _patch = mock.patch.object(storage, "DB_PATH", Path(_scratch.name) / "test.db")
    _patch.start()


def tearDownModule():
    _patch.stop()
    _scratch.cleanup()

class Words(unittest.TestCase):
    def test_split_lower_and_voice_names(self):
        got = baseline.words("Which genome of CREMA-D/1014 has vocalized_noise?")
        self.assertIn("vocalized", got)
        self.assertIn("noise", got)
        self.assertIn("cremad1014", got)        # the voice as one word
        self.assertNotIn("which", got)          # a stop word

    def test_vctk_voice(self):
        self.assertIn("vctkp250", baseline.words("voice VCTK/p250"))


class Bm25(unittest.TestCase):
    def setUp(self):
        self.index = baseline.BM25([
            record("clip:a/genome_01", "Voice CREMA-D/1014 genome 1 has laughter and a sigh."),
            record("clip:a/genome_02", "Voice CREMA-D/1014 genome 2 has hissing."),
            record("clip:b/genome_01", "Voice VCTK/p250 genome 1 has hissing and hissing again."),
            record("sanity:x", "Sanity check failed: words dropped.", kind="sanity"),
        ])

    def test_the_record_with_the_rare_words_ranks_first(self):
        hits = self.index.search("Which genome of CREMA-D/1014 has hissing?")
        self.assertEqual(hits[0]["id"], "clip:a/genome_02")

    def test_hits_carry_id_kind_revision_and_rank(self):
        hit = self.index.search("sanity failed")[0]
        self.assertEqual((hit["rank"], hit["id"], hit["kind"], hit["revision"]), (1, "sanity:x", "sanity", "2026-11-11 10:00:00"))

    def test_kind_filter_and_k(self):
        self.assertEqual([h["kind"] for h in self.index.search("genome", kind="clip")], ["clip"] * 3)
        self.assertEqual(len(self.index.search("genome", k=2)), 2)

    def test_no_shared_word_finds_nothing(self):
        self.assertEqual(self.index.search("angry pirate"), [])

    def test_same_question_same_ranking(self):
        self.assertEqual(self.index.search("genome hissing"), self.index.search("genome hissing"))


class Corpus(unittest.TestCase):
    def test_clip_and_run_records(self):
        runs = {"mos-diverse-v1_p1_CREMA-D_1014": run()}
        rows = [row("baseline", "", "3.0", ""), row("genome_01", "1", "4.5", "1.5", artifact="yes"),
                row("genome_02", "2", "2.0", "-1.0", rated_on="2026-11-11 10:05:00")]
        clips = corpus.clip_records(rows, runs)
        self.assertEqual([c["id"] for c in clips], ["clip:mos-diverse-v1_p1_CREMA-D_1014/baseline@ash",
                                                    "clip:mos-diverse-v1_p1_CREMA-D_1014/genome_01@ash",
                                                    "clip:mos-diverse-v1_p1_CREMA-D_1014/genome_02@ash"])
        one = clips[1]
        self.assertEqual(one["revision"], "2026-11-11 10:00:00")
        self.assertIn("better than the plain line", one["text"])
        self.assertIn("pause (dramatic)", one["text"])
        self.assertEqual((one["fields"]["features"], one["fields"]["artifact"]), (2, True))
        self.assertIn("worse than the plain line", clips[2]["text"])

        runs_out = corpus.run_records(clips, runs)
        self.assertEqual(runs_out[0]["id"], "run:mos-diverse-v1_p1_CREMA-D_1014@ash")
        self.assertEqual(runs_out[0]["revision"], "2026-11-11 10:05:00")      # the newest score in the run
        self.assertEqual(runs_out[0]["fields"]["best_genomes"], [1])
        self.assertEqual(runs_out[0]["fields"]["better_than_plain"], 1)

    def test_a_new_site_without_ratings_builds_an_empty_corpus(self):
        empty = Path(tempfile.mkdtemp())          # found by the Milestone 4 smoke test on a fresh container
        self.assertEqual(corpus.build(who_csv=empty / "who_rated_what.csv", runs_dir=empty, random_v2=None,
                                      sanity_all=None), [])

    def test_sanity_records_and_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "record.json").write_text(json.dumps({"name": "random-v2", "made": "2026-11-08 03:13:11"}), encoding="utf-8")
            (folder / "rows.json").write_text(json.dumps([{
                "voice": "VCTK_p231", "dataset": "VCTK", "line": 0, "genome": 2, "verdict": "terrible",
                "features": ["clucking", "noise", "sigh"], "markup": "x", "wer_ok": False, "similarity_ok": True,
                "pace_ok": True, "wer": 0.4, "similarity": 0.91, "heard": "y"}]), encoding="utf-8")
            records = corpus.sanity_records(folder)
            self.assertEqual(records[0]["id"], "sanity:random-v2/VCTK_p231/line0/genome02")
            self.assertEqual(records[0]["revision"], "2026-11-08 03:13:11")
            self.assertIn("words dropped or changed", records[0]["text"])

            manifest = corpus.write(records, folder / "corpus.jsonl", sources=[folder / "rows.json"])
            self.assertEqual((manifest["records"], manifest["kinds"]), (1, {"sanity": 1}))
            self.assertEqual(corpus.load(folder / "corpus.jsonl")[0]["id"], records[0]["id"])
            self.assertTrue((folder / "corpus_manifest.json").is_file())


def clip_record(run_id, clip, genome, score, above, voice="VCTK/p250", types=(), artifact=None, revision="2026-11-11 10:00:00"):
    return {"id": f"clip:{run_id}/{clip}", "kind": "clip", "revision": revision, "title": clip, "text": f"{clip} {score}",
            "fields": {"voice": voice, "run": run_id, "clip": clip, "genome": genome, "score": score,
                       "above_plain": above, "gene_types": list(types), "artifact": artifact}}


def small_corpus():
    r = "mos-diverse-v1_p1_VCTK_p250"
    return [clip_record(r, "baseline", None, 3.0, None),
            clip_record(r, "genome_01", 1, 4.0, 1.0, types=["sigh"]),
            clip_record(r, "genome_02", 2, 4.0, 1.0, types=["speed"]),
            clip_record(r, "genome_03", 3, 1.0, -2.0, types=["hissing"], artifact=True),
            {"id": f"run:{r}", "kind": "run", "revision": "2026-11-11 10:00:00", "title": "", "text": "",
             "fields": {"voice": "VCTK/p250", "run": r, "better_than_plain": 2, "best_genomes": [1, 2]}},
            {"id": "sanity:random-v2/a", "kind": "sanity", "revision": "2026-11-08 03:13:11", "title": "", "text": "",
             "fields": {"verdict": "failed", "gene_types": ["speed", "sigh"]}},
            {"id": "sanity:random-v2/b", "kind": "sanity", "revision": "2026-11-08 03:13:11", "title": "", "text": "",
             "fields": {"verdict": "passed", "gene_types": ["mn"]}}]


class Labels(unittest.TestCase):
    def test_rules(self):
        from backend.rag import labels
        rs, r = small_corpus(), "mos-diverse-v1_p1_VCTK_p250"
        self.assertEqual(labels.best_per_voice(rs), {f"run:{r}": 2, f"clip:{r}/genome_01": 1, f"clip:{r}/genome_02": 1})
        self.assertEqual(labels.bad_mos(rs), {f"clip:{r}/genome_03": 2})
        self.assertEqual(labels.artifacts(rs), {f"clip:{r}/genome_03": 2, "sanity:random-v2/a": 1})
        self.assertEqual(labels.better_than_plain(rs), {f"clip:{r}/genome_01": 2, f"clip:{r}/genome_02": 2, f"run:{r}": 1})
        self.assertEqual(labels.exclude(rs), {f"clip:{r}/genome_02": 1, f"clip:{r}/genome_03": 2, "sanity:random-v2/a": 1})
        self.assertEqual(labels.voice_better("VCTK/p250")(rs)[f"clip:{r}/baseline"], 1)
        self.assertEqual(labels.nothing(rs), {})

    def test_label_keeps_revisions_and_what_to_expect(self):
        from backend.rag import labels
        questions = [{"id": "q03", "question": "bad?"}, {"id": "q09", "question": "angry?"}]
        got = labels.label(small_corpus(), questions)
        self.assertEqual(got["q03"]["expect"], "answer")
        self.assertEqual(list(got["q03"]["relevant"].values()), [{"grade": 2, "revision": "2026-11-11 10:00:00"}])
        self.assertEqual((got["q09"]["expect"], got["q09"]["relevant"]), ("decline", {}))
        self.assertIn("worse than the plain line", got["q03"]["rule"])


class Refresh(unittest.TestCase):
    def test_compare_finds_added_removed_and_changed(self):
        from backend.rag import refresh
        old = small_corpus()
        new = small_corpus()[1:]                                         # the plain line is gone
        new[0] = {**new[0], "revision": "2026-11-12 09:00:00"}          # genome 1 scored again
        new.append(clip_record("x", "genome_01", 1, 2.0, 0.0))
        diff = refresh.compare(old, new)
        self.assertEqual(diff["removed"], ["clip:mos-diverse-v1_p1_VCTK_p250/baseline"])
        self.assertEqual(diff["changed"], ["clip:mos-diverse-v1_p1_VCTK_p250/genome_01"])
        self.assertEqual(diff["added"], ["clip:x/genome_01"])

    def test_stale_labels(self):
        from backend.rag import labels, refresh
        questions = [{"id": "q03", "question": "bad?"}, {"id": "q04", "question": "better?"}]
        saved = {"questions": labels.label(small_corpus(), questions)}
        self.assertEqual(refresh.stale_labels(saved, small_corpus(), questions)[0], {})     # nothing moved
        later = small_corpus()
        later[1] = clip_record("mos-diverse-v1_p1_VCTK_p250", "genome_01", 1, 2.0, -1.0, revision="2026-11-12 09:00:00")
        stale, _ = refresh.stale_labels(saved, later, questions)
        self.assertEqual(stale["q03"]["added"], ["clip:mos-diverse-v1_p1_VCTK_p250/genome_01"])     # now worse than plain
        self.assertEqual(stale["q04"]["dropped"], ["clip:mos-diverse-v1_p1_VCTK_p250/genome_01"])
        self.assertEqual(stale["q04"]["revised"], ["clip:mos-diverse-v1_p1_VCTK_p250/genome_01"])

    def test_check_sees_a_changed_source(self):
        from backend.rag import refresh
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            source = folder / "who.csv"
            source.write_text("a", encoding="utf-8")
            manifest = corpus.write([record("clip:x", "x")], folder / "corpus.jsonl", sources=[source])
            self.assertTrue(manifest["sources"])
            state = refresh.check(folder / "corpus_manifest.json", [source])
            self.assertEqual((state["changed"], state["missing"]), ([], []))
            source.write_text("b", encoding="utf-8")
            self.assertEqual(refresh.check(folder / "corpus_manifest.json", [source])["changed"], [str(source)])
            self.assertEqual(refresh.check(folder / "corpus_manifest.json", [folder / "gone.csv"])["missing"],
                             [str(folder / "gone.csv")])



class FakeEncoder:
    """Texts -> unit vectors over a few meaning groups, so dense search is tested without MiniLM."""
    GROUPS = [("artifact", "artifacts", "glitch", "noise"), ("worse", "bad", "low"), ("sigh", "breath"), ("hissing",)]

    def __init__(self):
        self.calls = 0

    def __call__(self, texts):
        import numpy as np
        self.calls += 1
        out = np.zeros((len(texts), len(self.GROUPS) + 1), dtype=np.float32)
        for row, text in enumerate(texts):
            words = set(baseline.words(text))
            for col, group in enumerate(self.GROUPS):
                out[row, col] = len(words & set(group))
            out[row, -1] = 0.1
        return out / np.linalg.norm(out, axis=1, keepdims=True)


class Variant(unittest.TestCase):
    def setUp(self):
        self.records = [
            record("clip:a/genome_01@ash", "Genome 1 of CREMA-D/1014 has a sigh."),
            record("clip:a/genome_02@ash", "Genome 2 had a glitch: an artifact was heard."),
            record("clip:b/genome_01@ash", "Genome 1 of VCTK/p250 scored worse than the plain line, it was bad."),
            record("sanity:x", "Sanity check failed: hissing.", kind="sanity"),
        ]

    def test_dense_matches_meaning_not_spelling(self):
        from backend.rag import dense
        index = dense.Dense(self.records, encode=FakeEncoder(), cache_dir=None)
        self.assertEqual(index.search("Which genomes produce artifacts?")[0]["id"], "clip:a/genome_02@ash")
        self.assertEqual(index.search("which were low", kind="clip")[0]["id"], "clip:b/genome_01@ash")
        self.assertEqual(len(index.search("anything", k=2)), 2)

    def test_dense_vectors_are_kept_until_the_corpus_changes(self):
        from backend.rag import dense
        with tempfile.TemporaryDirectory() as folder:
            first = FakeEncoder()
            dense.Dense(self.records, encode=first, cache_dir=folder)
            second = FakeEncoder()
            dense.Dense(self.records, encode=second, cache_dir=folder)
            self.assertEqual((first.calls, second.calls), (1, 0))          # the records were not encoded again
            changed = self.records[:-1] + [record("sanity:x", "Sanity check passed.", kind="sanity")]
            dense.Dense(changed, encode=second, cache_dir=folder)
            self.assertEqual(second.calls, 1)

    def test_hybrid_fuses_both_rankings(self):
        from backend.rag import dense
        hybrid = dense.Hybrid(self.records, dense=dense.Dense(self.records, encode=FakeEncoder(), cache_dir=None))
        ids = [h["id"] for h in hybrid.search("Which genome of CREMA-D/1014 has artifacts?")]
        self.assertIn("clip:a/genome_01@ash", ids[:2])       # BM25: the voice name
        self.assertIn("clip:a/genome_02@ash", ids[:2])       # dense: "artifacts" means the glitch
        top = hybrid.search("hissing")[0]
        self.assertEqual(top["id"], "sanity:x")
        self.assertAlmostEqual(top["score"], round(2 / 61, 4))   # first in both lists: 1/(60+1) twice


class Comparison(unittest.TestCase):
    def test_measures_of_one_ranking(self):
        from backend.rag import compare
        hits = [{"id": i} for i in ["x", "a", "y", "b"]]
        got = compare.scores(hits, {"a": 2, "b": 1, "c": 1})
        self.assertEqual((got["hit@1"], got["hit@3"], got["mrr"], got["first_relevant_rank"]), (0, 1, 0.5, 2))
        self.assertAlmostEqual(got["p@10"], 0.2)
        self.assertTrue(0 < got["ndcg@10"] < 1)
        perfect = compare.scores([{"id": "a"}, {"id": "b"}, {"id": "c"}], {"a": 2, "b": 1, "c": 1})
        self.assertAlmostEqual(perfect["ndcg@10"], 1.0)
        self.assertEqual(compare.scores([{"id": "x"}], {"a": 1})["mrr"], 0.0)

    def test_compare_means_paired_and_unsupported(self):
        from backend.rag import compare

        class Fixed:
            def __init__(self, order):
                self.order = order

            def search(self, question, k=10):
                return [{"rank": n, "id": i, "score": 1.0 / n, "title": i} for n, i in enumerate(self.order[question][:k], 1)]

        questions = [{"id": "q1", "question": "one", "kind": "lookup", "supported": True},
                     {"id": "q2", "question": "two", "kind": "summary", "supported": True},
                     {"id": "q3", "question": "none", "kind": "lookup", "supported": False}]
        key = {"q1": {"expect": "answer", "relevant": {"a": {"grade": 2}}},
               "q2": {"expect": "answer", "relevant": {"b": {"grade": 1}}},
               "q3": {"expect": "decline", "relevant": {}}}
        old = Fixed({"one": ["x", "a"], "two": ["b"], "none": ["z"]})
        new = Fixed({"one": ["a"], "two": ["b"], "none": []})
        got = compare.compare([], questions, key, {"bm25": old, "hybrid": new})
        self.assertEqual(got["means"]["bm25"]["all"]["mrr"], 0.75)
        self.assertEqual(got["means"]["hybrid"]["all"]["mrr"], 1.0)
        self.assertEqual(got["means"]["hybrid"]["lookup"]["questions"], 1)       # q3 is not scored
        self.assertEqual(got["paired_with_bm25"]["hybrid"], {"measure": "mrr", "better": 1, "same": 1, "worse": 0})
        unsupported = got["questions"][2]["by"]
        self.assertEqual((unsupported["bm25"]["top_score"], unsupported["hybrid"]["top_score"]), (1.0, 0.0))
        self.assertNotIn("mrr", unsupported["bm25"])
        self.assertEqual(got["questions"][0]["by"]["bm25"]["hits"][1]["grade"], 2)

    def test_refuses_labels_from_another_corpus(self):
        from backend.rag import compare
        compare.check_labels({"corpus_sha256": "a"}, {"corpus_sha256": "a"})
        with self.assertRaises(SystemExit):
            compare.check_labels({"corpus_sha256": "a"}, {"corpus_sha256": "b"})

    def test_two_listeners_of_one_pile_get_their_own_records(self):
        rows = [row("baseline", "", "3", ""), row("genome_01", "1", "4", "1"),
                {**row("baseline", "", "5", ""), "rater": "Julius", "account": "julius"}]
        clips = corpus.clip_records(rows, {run()["id"]: run()})
        self.assertEqual(len({c["id"] for c in clips}), 3)
        self.assertTrue(clips[2]["id"].endswith("@julius"))
        runs_out = corpus.run_records(clips, {run()["id"]: run()})
        self.assertEqual(sorted(r["id"].split("@")[1] for r in runs_out), ["ash", "julius"])

    def test_test_accounts_are_not_evidence(self):
        rows = [row("baseline", "", "3", ""),
                {**row("baseline", "", "5", ""), "rater": "Test user 3", "account": "test_user_3"},
                {**row("genome_01", "1", "2", "-1"), "rater": "Test user 12", "account": ""}]     # no login: by name
        clips = corpus.clip_records(rows, {run()["id"]: run()})
        self.assertEqual([c["fields"]["listener"] for c in clips], ["Ash"])
        self.assertFalse(corpus.is_test_account({"rater": "Tester", "account": "testuser"}))


def answer_corpus():
    """Two listeners on one voice, full fields, as the corpus writes them."""
    r = "mos-diverse-v1_p1_VCTK_p250"
    def clip(who, genome, score, above, types=(), artifact=None):
        name = f"genome_{genome:02d}" if genome else "baseline"
        return {"id": f"clip:{r}/{name}@{who.lower()}", "kind": "clip", "revision": "2026-11-11 10:00:00",
                "title": f"VCTK/p250 {name}, scored {score} by {who}",
                "text": f"Voice VCTK/p250, genome {genome}. {who} scored it {score}. Gene types: {', '.join(types)}.",
                "fields": {"voice": "VCTK/p250", "run": r, "clip": name, "genome": genome, "score": score,
                           "above_plain": above, "gene_types": list(types), "features": len(types),
                           "artifact": artifact, "listener": who, "listener_key": who.lower()}}
    return [clip("Ash", None, 3.0, None), clip("Ash", 1, 4.5, 1.5, ["sigh"]), clip("Ash", 2, 1.0, -2.0, ["hissing"]),
            clip("Julius", None, 2.0, None), clip("Julius", 1, 1.0, -1.0, ["sigh"]),
            {"id": f"run:{r}@ash", "kind": "run", "revision": "2026-11-11 10:00:00", "title": "VCTK/p250 run (Ash)",
             "text": "Listening run for voice VCTK/p250, scored by Ash.",
             "fields": {"voice": "VCTK/p250", "run": r, "listener": "Ash", "better_than_plain": 1, "best_genomes": [1]}},
            {"id": "sanity:random-v2/VCTK_p250/line0/genome01", "kind": "sanity", "revision": "2026-11-08 03:13:11",
             "title": "random-v2 VCTK_p250 genome 1: sanity failed",
             "text": "Random search genome on voice VCTK_p250, genome 1. Gene types: speed. Sanity check failed.",
             "fields": {"voice": "VCTK_p250", "verdict": "failed", "gene_types": ["speed"]}}]


class Answers(unittest.TestCase):
    def test_routes(self):
        from backend.rag import answer
        self.assertEqual(answer.route("Which genomes work best for each voice?"), ("summary", "best", None))
        self.assertEqual(answer.route("Which genomes of voice VCTK/p250 scored better than its plain line?"),
                         ("summary", "better", "VCTK/p250"))
        self.assertEqual(answer.route("How often does a genome beat its plain line, for each number of active features?")[1],
                         "by_features")
        self.assertEqual(answer.route("What did genome 5 of voice CREMA-D/1014 do?"), ("genome", 5, "CREMA-D/1014"))
        self.assertEqual(answer.route("Which genome suits an angry character?"), ("search", None, None))

    def test_a_summary_cites_the_records_it_counted(self):
        from backend.rag import answer
        a = answer.answer("Which genomes scored better than the plain line?", answer_corpus())
        self.assertFalse(a["absent"])
        self.assertIn("1 of 3 rated genomes", a["answer"])
        self.assertEqual(a["cited_ids"], ["clip:mos-diverse-v1_p1_VCTK_p250/genome_01@ash", "run:mos-diverse-v1_p1_VCTK_p250@ash"])
        self.assertEqual(a["citations"][0]["revision"], "2026-11-11 10:00:00")

    def test_one_genome_and_a_missing_one(self):
        from backend.rag import answer
        a = answer.answer("What did genome 1 of VCTK/p250 do?", answer_corpus())
        self.assertIn("clip:mos-diverse-v1_p1_VCTK_p250/genome_01@ash", a["cited_ids"])
        missing = answer.answer("What did genome 9 of VCTK/p250 do?", answer_corpus())
        self.assertTrue(missing["absent"])
        self.assertEqual(missing["citations"], [])

    def test_safe_absence_below_the_cut_off(self):
        from backend.rag import answer
        a = answer.answer("Which voice sounds like a pirate?", answer_corpus())
        self.assertTrue(a["absent"])
        self.assertIn("No record in the evidence answers this", a["answer"])
        found = answer.answer("Which random search genome on voice VCTK_p250 failed the sanity check?", answer_corpus(), cutoff=1.0)
        self.assertEqual((found["absent"], found["cited_ids"][0]), (False, "sanity:random-v2/VCTK_p250/line0/genome01"))
        self.assertTrue(answer.answer("Which random search genome on voice VCTK_p250 failed the sanity check?",
                                      answer_corpus(), cutoff=1e6)["absent"])

    def test_a_listener_sees_only_their_own_scores(self):
        from backend.rag import answer
        seen = answer.visible(answer_corpus(), "Julius")
        self.assertEqual({r["kind"] for r in seen}, {"clip", "sanity"})
        self.assertTrue(all(r["fields"]["listener"] == "Julius" for r in seen if r["kind"] == "clip"))
        a = answer.answer("Which genomes scored better than the plain line?", answer_corpus(), listener="Julius")
        self.assertIn("0 of 1 rated genomes", a["answer"])
        self.assertTrue(all("@julius" in i for i in a["cited_ids"]))

    def test_check_flags_a_wrong_decline(self):
        from backend.rag import answer
        rs = answer_corpus()
        key = {"questions": {"q1": {"expect": "answer", "relevant": {"clip:mos-diverse-v1_p1_VCTK_p250/genome_01@ash": {}}},
                             "q2": {"expect": "decline", "relevant": {}}}}
        qs = [{"id": "q1", "question": "Which genomes scored better than the plain line?"},
              {"id": "q2", "question": "Which genome suits an angry character?"}]
        out = answer.check(rs, qs, key, [{"question": "How much did the GPU cost?", "route": "search", "absent": True}])
        self.assertEqual([r["ok"] for r in out["questions"]], [False, True])   # q1 also cites the run, not in this key
        self.assertTrue(out["paraphrases"][0]["ok"])


if __name__ == "__main__":
    unittest.main()

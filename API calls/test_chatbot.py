"""test_chatbot.py | The chatbot's parts that need no model or API: reading replies, the loop, and the scoring.

    python "API calls/test_chatbot.py"
"""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chatbot  # noqa: E402
import chatbot_eval  # noqa: E402


class Replies(unittest.TestCase):
    def test_first_json_finds_the_object_in_words_and_fences(self):
        self.assertEqual(chatbot.first_json('Sure! ```json\n{"tool": "overview", "arguments": {}}\n```'),
                         {"tool": "overview", "arguments": {}})
        self.assertEqual(chatbot.first_json('{"answer": "Ash, 3.0 {not json"}'), {"answer": "Ash, 3.0 {not json"})
        self.assertIsNone(chatbot.first_json("no json here"))

    def test_the_loop_calls_a_tool_then_answers(self):
        class Model:
            replies = iter(['{"tool": "query_scores", "arguments": {"listener": "Rama"}}',
                            '{"answer": "Rama: 5.0 for p250"}'])

            def chat(self, messages):
                self.seen = messages
                return next(self.replies)

        class Results:
            def run_tool(self, name, arguments):
                return json.dumps({"rows": [{"voice": "VCTK/p250", "score": 5.0}], "asked": [name, arguments]})

        model = Model()
        out = chatbot.Chatbot(Results(), model).ask("Which genomes did Rama rate highest?")
        self.assertEqual(out["answer"], "Rama: 5.0 for p250")
        self.assertEqual([(s["tool"], s["arguments"]) for s in out["steps"]], [("query_scores", {"listener": "Rama"})])
        self.assertIn("VCTK/p250", model.seen[-1]["content"])           # the tool's result went back to the model
        self.assertIn("query_scores", model.seen[0]["content"])         # the tools are in the instructions

    def test_a_reply_without_json_is_the_answer_in_a_conversation(self):
        class Model:
            def chat(self, messages):
                return "There is no data."
        earlier = [{"role": "user", "content": "Rama's scores?"}, {"role": "assistant", "content": "None yet."}]
        self.assertEqual(chatbot.Chatbot(None, Model()).ask("?", earlier)["answer"], "There is no data.")

    def test_it_declines_questions_that_are_not_about_the_dramatizer(self):
        class Says:
            def __init__(self, reply):
                self.reply = reply

            def chat(self, messages):
                return self.reply
        bot = lambda reply: chatbot.Chatbot(None, Says(reply))                         # noqa: E731
        self.assertEqual(bot('{"answer": "OFF_TOPIC"}').ask("What is the capital of France?")["answer"], chatbot.OFF_TOPIC)
        # an answer made without any tool, at the start of a conversation, is not from the project's data
        self.assertEqual(bot('{"answer": "Paris."}').ask("What is the capital of France?")["answer"], chatbot.OFF_TOPIC)
        self.assertIn("Only answer questions about the Dramatizer", chatbot.INSTRUCTIONS)


class Serve(unittest.TestCase):
    """--serve: what the API server's worker reads and writes (backend/api/assistant.py)."""

    def test_each_question_uses_its_own_token_and_gets_one_reply_line(self):
        import io
        from unittest import mock

        class Api:
            token = None

        api = Api()
        used = []

        class Model:
            name = "fake"
            replies = iter(['{"tool": "overview", "arguments": {}}', '{"answer": "Two piles."}', "boom"])

            def chat(self, messages):
                reply = next(self.replies)
                if reply == "boom":
                    raise ValueError("model failed")
                return reply

        def run_tool(self, name, arguments):
            used.append(api.token)
            return "{}"

        first = {"id": 1, "question": "How many piles?", "token": "tok-A", "history": [{"role": "system", "content": "x"}]}
        second = {"id": 2, "question": "Again?", "token": "tok-B"}
        stdin = io.StringIO(json.dumps(first) + "\n\n" + json.dumps(second) + "\n")
        stdout = io.StringIO()
        with mock.patch.object(chatbot.Results, "run_tool", run_tool):
            chatbot.serve(Model(), api, stdin, stdout)
        lines = [json.loads(x[len(chatbot.REPLY):]) for x in stdout.getvalue().splitlines()]
        self.assertEqual(lines[0], {"ready": True, "model": "fake"})
        self.assertEqual((lines[1]["id"], lines[1]["answer"]), (1, "Two piles."))
        self.assertEqual(lines[1]["tools"], [{"tool": "overview", "arguments": {}}])
        self.assertEqual(used, ["tok-A"])                                       # the tool call went as the asker
        self.assertEqual(lines[2]["id"], 2)
        self.assertIn("model failed", lines[2]["error"])                       # an error is a reply, not a crash
        self.assertIsNone(api.token)                                            # no token kept between questions


class Scoring(unittest.TestCase):
    def test_numbers_match_with_rounding_and_percent(self):
        self.assertTrue(chatbot_eval.has(" passed 67% of them", 0.671))
        self.assertTrue(chatbot_eval.has("a mean of 2.16", 2.163))
        self.assertTrue(chatbot_eval.has("80 scores", 80))
        self.assertFalse(chatbot_eval.has("81 scores", 80))
        self.assertTrue(chatbot_eval.has("voice 1014 scored best", "CREMA-D/1014"))

    def test_judge_paths_ties_and_claims(self):
        truth = {"rows": [{"voice": "VCTK/p250", "score": 5.0}, {"voice": "VCTK/p310", "score": 5.0},
                          {"voice": "VCTK/p248", "score": 4.0}], "sort": "score"}
        q = {"truth": {"must": ["rows.0.score"], "any_tied": "voice"}}
        self.assertEqual(chatbot_eval.judge(q, "p310 got 5.0", truth), [])
        self.assertTrue(chatbot_eval.judge(q, "p248 got 5.0", truth))     # p248 is not among the tied
        claim = {"truth": {"claim": "answer"}}
        self.assertEqual(chatbot_eval.judge(claim, "13 of 24 voices; 16 above", {"answer": "13 of 24 voices, 16 of 24 above"}), [])
        self.assertTrue(chatbot_eval.judge(claim, "most voices", {"answer": "13 of 24 voices, 16 of 24 above"}))
        # A chart with no data ("No data"): the right answer says so, one naming results anyway is wrong
        empty = {"skipped": "no listener answered the artifact question in this range"}
        self.assertEqual(chatbot_eval.judge(claim, "No listener has answered the artifact question yet.", empty), [])
        self.assertEqual(chatbot_eval.judge(claim, "There is no data for this yet.", empty), [])
        self.assertTrue(chatbot_eval.judge(claim, "Genome 3 of p250 produces artifacts.", empty))
        self.assertTrue(chatbot_eval.judge(claim, "anything", {}))                     # no claim and no reason: wrong
        self.assertEqual(chatbot_eval.values_at({"g": [{"l": "A"}, {"l": "B"}]}, "g.*.l"), ["A", "B"])


if __name__ == "__main__":
    unittest.main()

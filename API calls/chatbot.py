"""chatbot.py | A chatbot that answers questions about the Dramatizer's results, with Gemma 4 and the API's tools.

The model never computes or remembers a number: it picks a tool (API_REQUEST.TOOLS), the tool asks the API,
and the model words the answer from what came back. Exact questions ("Which genomes did Rama rate highest?")
go to query_scores, the six research questions to evidence_question (the same numbers as their charts),
questions about the evidence that the others do not fit to answer_evidence (it cites record IDs, or says nothing
answers them), and anything else to search_evidence or the other tools.

    python chatbot.py                                   # chat in the terminal
    python chatbot.py --ask "Which genomes did Rama rate highest?"
    python chatbot.py --model gemma-4-E2B-it            # the smaller Gemma 4 (default gemma-4-E4B-it)
    python chatbot.py --server http://127.0.0.1:11434/v1 --model gemma4   # an Ollama / llama.cpp / vLLM server
    python chatbot.py --serve --url https://127.0.0.1:5005 --fingerprint AB:CD:...   # for the website (below)

--serve is how the website's "Ask the Dramatizer" uses it: the API server starts this script in the LLM
environment (backend/api/assistant.py) and sends one question per line on stdin, as JSON with the asker's
own session token. Every tool call goes to the API with that token, so the model sees exactly what the
asker may see. Replies are lines on stdout starting with REPLY; anything else (loading messages) is ignored.

The local model is loaded from LLM_LIBRARY_DIR (.env) by backend/model.py, in 4-bit, so run it with the
environment that has Gemma 4's transformers:  D:/envs/llm/python.exe chatbot.py
The API's address and login come from DRAMATIZER_URL, DRAMATIZER_FINGERPRINT and DRAMATIZER_USER or
DRAMATIZER_KEY (README.md). Use an admin account or an API key to ask about other listeners' scores:
a listener or researcher sees only their own.

How the model calls a tool: the instructions list the tools, and the model answers with one JSON object,
either {"tool": NAME, "arguments": {...}} or {"answer": "..."}. That works with any chat model and
template, local or on a server, without relying on a template's own tool format.
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from API_REQUEST import TOOLS, Results  # noqa: E402

MAX_STEPS = 6                 # tool calls per question, at most
RESULT_CHARS = 6000           # a tool's answer is cut to this many characters for the prompt
ANSWER_TOKENS = 600

INSTRUCTIONS = """You answer questions about the Dramatizer, a project that clones voices and adds a delivery to a
line with a "genome": a few edits (genes) such as a sigh, a pause, a laugh, stress (accent) or a speed change.
Listeners scored each genome's clip from 0 (bad) to 5 (excellent); "above plain" is the score minus the same
listener's score for the plain line (no genome) of that voice. MOS is the mean of the listeners' scores.
Voices are written DATASET/SPEAKER, e.g. CREMA-D/1014, VCTK/p250, LibriTTS-R/6319.

You know no numbers yourself: every fact must come from a tool. Reply with exactly one JSON object and nothing else:
  to call a tool:      {"tool": "<name>", "arguments": {...}}
  to give the answer:  {"answer": "<your answer>"}

Rules:
- Call a tool first, then answer from its result. Call more tools if the first does not answer the question.
- If a tool returns an error, read it, fix the arguments and call the tool again.
- Who-rated-what, highest, lowest, averages, gene types, voices, datasets, counts: query_scores.
  Put every filter the question names into the arguments: a listener, a voice, a dataset, a gene type,
  a number of genes ("genomes with two genes" -> genes 2), a genome number.
  For "highest" use sort "score" (rows) or group_by with sort "mean"; for "lowest" add order "asc".
  "How often ... beats the plain line" is beat_plain in a group_by answer (a share: 0.26 = 26%).
  "Average per voice / dataset / gene type / listener": query_scores with group_by and sort "mean".
  "Genome scores" are clip "genome" (the default): leave clip out unless the question is about the plain line.
  "How many scores" is the count in a group_by answer.
- One genome of a voice ("genome 5 of CREMA-D/1014"): query_scores with voice and genome; its genes are in the rows.
- The six research questions, asked in these or similar words, are answered with evidence_question and the number:
  1 "Which genomes work best for each voice?"   2 "Which genomes produce artifacts?"
  3 "Which genome configurations give bad MOS?"  4 "Which configurations consistently give good genomes?"
  5 "How often does a genome beat the plain line, for each number of active features?"
  6 "Which genomes can we exclude?"
  Always use evidence_question for these six, even when query_scores could compute the same thing.
- How many genome searches, voices or renders there are: overview.
- What the Dramatizer is, how to use the website (how to rate clips, generate, listen, ask, research), who made
  it: about_dramatizer. These are questions about the Dramatizer: always call the tool for them.
- Baselines (WER, similarity, PESQ, STOI, SI-SDR): baseline_results. Milestone 2: milestone2_results.
- Other questions about genomes, scores or sanity checks: answer_evidence. Give its record IDs as the sources, and if
  it says nothing in the evidence answers the question, say that and do not guess.
- Quote the numbers you were given, with the voice and genome they belong to. Mention ties.
- If a tool returns nothing, an error or a note that the account cannot see the data, say so plainly.
- Keep the answer short: a few sentences or a short list.
- How many voices, speakers or datasets the project uses, or which datasets: voices (no arguments). Give the total,
  the validation and the locked test voices, and the voices per dataset.
- Only answer questions about the Dramatizer: its voices, genomes, scores, listeners, results, website and team.
  For anything else (general knowledge, other topics, jokes, code, advice, chit-chat) call no tool and reply
  exactly {"answer": "OFF_TOPIC"}.

Tools:
"""


def tool_list():
    lines = []
    for t in TOOLS:
        f = t["function"]
        params = []
        for name, spec in f["parameters"]["properties"].items():
            kind = spec.get("type", "string")
            extra = f" one of {spec['enum']}" if "enum" in spec else ""
            note = f": {spec['description']}" if "description" in spec else ""
            params.append(f"    {name} ({kind}{extra}{', required' if name in f['parameters']['required'] else ''}){note}")
        lines.append(f"- {f['name']}: {f['description']}" + ("\n" + "\n".join(params) if params else ""))
    return "\n".join(lines)


def first_json(text):
    """The first JSON object in a model's reply (it may wrap it in ``` or add words), or None."""
    text = re.sub(r"```(?:json)?", "", text)
    decoder = json.JSONDecoder()
    for start in (m.start() for m in re.finditer(r"\{", text)):
        try:
            value, _ = decoder.raw_decode(text[start:])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


# ------------------------------------------------------------------ the model --
class LocalModel:
    """A model from LLM_LIBRARY_DIR, loaded by backend/model.py (4-bit, on the GPU)."""

    def __init__(self, name):
        sys.path.insert(0, str(HERE.parent))
        from backend import model
        self.model = model
        self.name = name
        self.tokenizer, self.llm = model.emphasis_model(name)

    def chat(self, messages):
        import torch

        try:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=False)
        except Exception:                          # a template without a system role: fold it into the first turn
            folded = [{"role": "user", "content": messages[0]["content"] + "\n\n" + messages[1]["content"]}] + messages[2:]
            text = self.tokenizer.apply_chat_template(folded, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device())
        with torch.inference_mode():
            output = self.llm.generate(**inputs, max_new_tokens=ANSWER_TOKENS, do_sample=False, top_p=None,
                                       top_k=None, temperature=None,
                                       pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        reply = self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return reply.split("</think>")[-1].strip()


class ServerModel:
    """A model behind an OpenAI-compatible /chat/completions (Ollama, llama.cpp, vLLM, LM Studio)."""

    def __init__(self, url, name):
        self.url, self.name = url.rstrip("/") + "/chat/completions", name

    def chat(self, messages):
        body = json.dumps({"model": self.name, "messages": messages, "temperature": 0, "max_tokens": ANSWER_TOKENS})
        request = urllib.request.Request(self.url, data=body.encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as reply:
            return json.loads(reply.read())["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------- the chatbot --
class Chatbot:
    def __init__(self, results, model):
        self.results, self.model = results, model
        self.system = INSTRUCTIONS + tool_list()

    def ask(self, question, history=()):
        """Answer one question: {"answer", "steps": [{"tool", "arguments", "result"}], "seconds"}."""
        started = time.time()
        messages = [{"role": "system", "content": self.system}, *history, {"role": "user", "content": question}]
        steps = []
        for _ in range(MAX_STEPS + 1):
            reply = self.model.chat(messages)
            call = first_json(reply)
            if call and "tool" in call and len(steps) < MAX_STEPS:
                name, arguments = str(call["tool"]), call.get("arguments") or {}
                result = self.results.run_tool(name, arguments)
                steps.append({"tool": name, "arguments": arguments, "result": result})
                shown = result if len(result) <= RESULT_CHARS else result[:RESULT_CHARS] + " ... (cut)"
                messages += [{"role": "assistant", "content": json.dumps({"tool": name, "arguments": arguments})},
                             {"role": "user", "content": f"Result of {name}:\n{shown}\n\nCall another tool, or reply "
                                                         '{"answer": "..."} using only these results.'}]
                continue
            answer = call.get("answer") if call and "answer" in call else reply
            if call and "tool" in call:            # out of steps
                answer = "I could not finish: too many tool calls for one question."
            elif "OFF_TOPIC" in str(answer) or not steps and not history:
                answer = OFF_TOPIC                 # every fact comes from a tool: an answer without one is not about us
            return {"answer": str(answer).strip(), "steps": steps, "seconds": round(time.time() - started, 1)}
        return {"answer": "I could not finish.", "steps": steps, "seconds": round(time.time() - started, 1)}


REPLY = "@@reply "
OFF_TOPIC = ("I only answer questions about the Dramatizer: its voices, genomes, listening scores, results, the website "
             "and the team. Try for example \"Which genomes work best for each voice?\" or \"How do I rate clips?\"")
HISTORY_TURNS = 6             # earlier messages of the conversation sent with a question, at most


def serve(model, api, stdin=None, stdout=None):
    """Answer questions from stdin, one JSON line each: {"id", "question", "token", "history"}. Each reply is one
    line on stdout: REPLY + {"id", "answer", "tools": [{"tool", "arguments"}], "seconds"} or {"id", "error"}."""
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    bot = Chatbot(Results(api), model)
    stdout.write(REPLY + json.dumps({"ready": True, "model": getattr(model, "name", None)}) + "\n")
    stdout.flush()
    for line in stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        api.token = request.get("token")
        try:
            history = [m for m in (request.get("history") or []) if m.get("role") in ("user", "assistant")
                       and isinstance(m.get("content"), str)][-HISTORY_TURNS:]
            out = bot.ask(str(request["question"]), history)
            reply = {"id": request.get("id"), "answer": out["answer"], "seconds": out["seconds"],
                     "tools": [{"tool": s["tool"], "arguments": s["arguments"]} for s in out["steps"]]}
        except Exception as error:                 # one bad question must not stop the service
            reply = {"id": request.get("id"), "error": f"{type(error).__name__}: {error}"}
        finally:
            api.token = None
        stdout.write(REPLY + json.dumps(reply, ensure_ascii=False) + "\n")
        stdout.flush()


def make_model(args):
    return ServerModel(args.server, args.model) if args.server else LocalModel(args.model)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ask the Dramatizer's results in plain words (Gemma 4 + the API).")
    parser.add_argument("--ask", help="one question; without it, a chat in the terminal")
    parser.add_argument("--model", default="gemma-4-E4B-it", help="a folder in LLM_LIBRARY_DIR, or the server's model name")
    parser.add_argument("--server", help="an OpenAI-compatible address instead of a local model, e.g. http://127.0.0.1:11434/v1")
    parser.add_argument("--steps", action="store_true", help="show the tool calls")
    parser.add_argument("--serve", action="store_true", help="answer JSON lines on stdin for the API server")
    parser.add_argument("--url", help="with --serve: the API's address")
    parser.add_argument("--fingerprint", help="with --serve: the API certificate's fingerprint")
    args = parser.parse_args(argv)
    if args.serve:
        from dramatizer_api import Api
        api = Api(args.url or "https://127.0.0.1:5005", args.fingerprint)
        serve(make_model(args), api)
        return 0
    with Results.connect() as results:
        bot = Chatbot(results, make_model(args))
        questions = [args.ask] if args.ask else iter(lambda: input("\nYou: ").strip(), "quit")
        if not args.ask:
            print("Ask about the ratings, genomes and results. 'quit' ends.")
        for question in questions:
            if not question:
                continue
            out = bot.ask(question)
            if args.steps:
                for s in out["steps"]:
                    print(f"  [{s['tool']} {json.dumps(s['arguments'])}] -> {len(s['result'])} characters")
            print(f"\nDramatizer: {out['answer']}\n  ({out['seconds']} s, {len(out['steps'])} tool calls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

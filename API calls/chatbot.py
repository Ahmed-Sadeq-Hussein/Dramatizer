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
sys.path.insert(1, str(HERE.parent))                  # the project: backend/api/small_talk.py
from API_REQUEST import TOOLS, Results  # noqa: E402
from backend.api import small_talk  # noqa: E402

MAX_STEPS = 6                 # tool calls per question, at most
RESULT_CHARS = 6000           # a tool's answer is cut to this many characters for the prompt
SHOWN_CHARS = 8000            # and to this many in the reply to the page (the Ask tab's look-up details)
MAX_LINKS = 25                # genomes and voices the page may link to, per answer
ANSWER_TOKENS = 600
# A question that asks what something means: the glossary is looked up before the model answers, so the model
# explains the program's own definition instead of guessing one (it once called STOI "Speech Transmission Index")
ASKS_MEANING = re.compile(r"\b(what (is|are|does|do)|what's|whats|mean|means|meaning|measure|measures|stand for|"
                          r"stands for|define|definition|explain)\b", re.IGNORECASE)

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
- What a word, measurement or idea of the program means ("what is STOI", "what does WER measure", "explain the
  sanity check", "what is a genome"): define with that term. For a follow-up ("what does it measure?") use the
  term from the conversation. Answer from its meaning and scale in your own short words; never define a term
  without calling define. For the project's own numbers of that measurement, also call the tool that has them.
- A render's ID (from Copy ID: "render:" and digits, or a long name): render_info for what it is (voice, line,
  genome, genes); measure_render for its similarity to its reference (original recording), its WER or quality, or
  to compare two renders (compare_with). Pass the ID exactly as the question gives it.
- Baselines (the measured WER, similarity, PESQ, STOI, SI-SDR): baseline_results. Milestone 2: milestone2_results.
- Other questions about genomes, scores or sanity checks: answer_evidence. Give its record IDs as the sources, and if
  it says nothing in the evidence answers the question, say that and do not guess.
- You explain the analysis; you never decide. Recommending a genome, naming a winner, "which should I use", "how
  sure are we" or "which ratings is this based on": evidence_status (with the voice if one is named).
  Name a genome as the best or recommend it ONLY if evidence_status gives that voice status "winner". For any
  other status say plainly there is not enough evidence (or no clear winner), give its reason, and you may list
  the highest-scored safe genomes as candidates, each with its MOS and number of listeners.
- Every answer about scores says what it rests on: the number of listeners (n) behind each number you quote,
  and the snapshot_id and ratings_up_to when the tool gives them. A score from one listener is that listener's
  opinion: say "1 listener", never present it as a result.
- Quote the numbers you were given, with the voice and genome they belong to. Mention ties.
- If a tool returns nothing, an error or a note that the account cannot see the data, say so plainly.
- Keep the answer short: a few sentences or a short list.
- Lists of genomes ("list", "show", "which genomes", "genomes that performed well", "good genomes", "did X rate any
  genome 5"): query_scores with limit 20 (or more when asked), sort "score". "Rated 5" or "a score of N" is
  min_score N (and max_score N for exactly N); "performed well" is min_score 4. Give at least 10 when there are
  that many: one line each with the voice, genome number, score and listener. The page adds a link to each genome
  and voice in the tool results, so do not write links or URLs yourself.
- Lists of voices: voices with a dataset (and query for words or a speaker); without a dataset first to see them.
- A greeting, thanks or "what can you do": call no tool and reply exactly {"answer": "GREETING"}.
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
        hello = small_talk.greeting(question)
        if hello:                                  # small talk: answered at once, no model, no look-up
            return {"answer": hello, "steps": [], "seconds": 0.0, "small_talk": True}
        messages = [{"role": "system", "content": self.system}, *history, {"role": "user", "content": question}]
        steps = []
        if ASKS_MEANING.search(question):
            self._define_first(question, messages, steps)
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
            elif "GREETING" in str(answer) and not steps:
                answer = small_talk.HELLO_REPLY
            elif "OFF_TOPIC" in str(answer) or not steps and not history:
                answer = OFF_TOPIC                 # every fact comes from a tool: an answer without one is not about us
            return {"answer": str(answer).strip(), "steps": steps, "seconds": round(time.time() - started, 1)}
        return {"answer": "I could not finish.", "steps": steps, "seconds": round(time.time() - started, 1)}

    def _define_first(self, question, messages, steps):
        """Look the question's terms up in the glossary (define) and hand the model what it found, as if it had
        called define itself. Nothing is added when no term matches."""
        if self.results is None:
            return
        result = self.results.run_tool("define", {"term": question})
        try:
            found = json.loads(result).get("matches")
        except (ValueError, AttributeError):
            return
        if not found:
            return
        steps.append({"tool": "define", "arguments": {"term": question}, "result": result})
        shown = result if len(result) <= RESULT_CHARS else result[:RESULT_CHARS] + " ... (cut)"
        messages += [{"role": "assistant", "content": json.dumps({"tool": "define", "arguments": {"term": question}})},
                     {"role": "user", "content": f"Result of define:\n{shown}\n\nThese are the program's own "
                                                 "definitions: use them for what the terms mean (never another meaning "
                                                 "of the same letters). Call another tool for numbers, or reply "
                                                 '{"answer": "..."} using only these results.'}]


REPLY = "@@reply "
OFF_TOPIC = small_talk.OFF_TOPIC


def links(steps):
    """
    The genomes and voices in the tool results, for the page to link to (it opens the genome in the studio, or
    the voice in Voices): [{"kind": "genome", "key", "voice", "genome", "score"?, "listener"?, "genes"?} |
    {"kind": "voice", "dataset", "speaker"}], each once, in the order the results gave them. The model does not
    write links: they come from the data, so a link always points at something the asker may see.
    """
    out, seen = [], set()

    def add(item, key):
        if key not in seen and len(out) < MAX_LINKS:
            seen.add(key)
            out.append(item)

    def voice(name):
        if name and "/" in str(name):
            dataset, speaker = str(name).split("/", 1)
            add({"kind": "voice", "dataset": dataset, "speaker": speaker}, ("voice", dataset, speaker))

    def genome(search, number, voice_name, **more):
        if search and number is not None:
            key = f"search:{search}/{number}"
            add({"kind": "genome", "key": key, "voice": voice_name, "genome": number,
                 **{k: v for k, v in more.items() if v is not None}}, key)

    for step in steps:
        try:
            data = json.loads(step["result"])
        except (TypeError, ValueError):
            continue
        if not isinstance(data, dict) or data.get("error"):
            continue
        tool = step["tool"]
        if tool == "query_scores":
            for row in data.get("rows") or []:
                genome(row.get("search"), row.get("genome"), row.get("voice"), score=row.get("score"),
                       listener=row.get("listener"), genes=row.get("genes_described") or None)
            for row in data.get("rows") or []:
                voice(row.get("voice"))
            for grp in data.get("groups") or []:
                voice(grp.get("voice"))
        elif tool == "top_genomes":
            for g in data.get("genomes") or []:
                genome(g.get("search"), g.get("number"), g.get("voice"), score=(g.get("mos") or {}).get("mos")
                       if isinstance(g.get("mos"), dict) else g.get("mos"), genes=g.get("genes") or None)
        elif tool == "search_details":
            for g in data.get("genomes") or []:
                genome(data.get("id"), g.get("number"), data.get("voice"), genes=g.get("genes") or None)
            voice(data.get("voice"))
        elif tool == "named_genomes":
            for g in data.get("named") or []:
                if g.get("from_search"):
                    genome(g["from_search"], g.get("genome"), g.get("voice"), name=g.get("name"))
                elif g.get("name"):
                    add({"kind": "genome", "key": "named:" + g["name"], "name": g["name"]}, "named:" + g["name"])
        elif tool == "voices":
            for v in data.get("voices") or []:
                add({"kind": "voice", "dataset": v.get("dataset"), "speaker": v.get("speaker")},
                    ("voice", v.get("dataset"), v.get("speaker")))
        elif tool == "pile":
            for v in data.get("voices") or []:
                voice(v.get("voice"))
        elif tool in ("render_info", "measure_render"):
            render_id = data.get("id") or data.get("render_id")
            if render_id:
                add({"kind": "render", "id": render_id, "voice": data.get("voice"), "genome": data.get("genome")},
                    ("render", render_id))
            voice(data.get("voice"))
    return out
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
                     "tools": [{"tool": s["tool"], "arguments": s["arguments"],
                                "result": s["result"] if len(s["result"]) <= SHOWN_CHARS
                                else s["result"][:SHOWN_CHARS] + " ... (cut)"} for s in out["steps"]],
                     "links": links(out["steps"]), "small_talk": bool(out.get("small_talk"))}
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

"""client_loop.py | Step 5: Gemma 4 picks MCP tools from dramatizer_mcp.py, one call at a time.

    python client_loop.py --ask "Which genomes did Rama rate highest?" --server http://127.0.0.1:11434/v1 --model gemma4-8k --steps

Stops when: the model answers, MAX_STEPS tool calls are used, the same call is repeated, TIME_BUDGET runs out,
or the model says a person is needed. Every run is appended to history/mcp_traces/runs.jsonl (private, git-ignored).
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "API calls"))
from chatbot import OFF_TOPIC, ServerModel  # noqa: E402

import chatbot  # noqa: E402
chatbot.ANSWER_TOKENS = 1500   # room for the whole {"answer": ...} object

SERVER = HERE / "dramatizer_mcp.py"
TRACES = ROOT / "history" / "mcp_traces"
MAX_STEPS = 6
TIME_BUDGET = 600          # seconds per question
RESULT_CHARS = 6000

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
- Who-rated-what, highest, lowest, averages, gene types, voices, counts: query_scores. Put every filter the
  question names into the arguments (listener, voice, dataset, gene_type, genes, genome). For "highest" use
  sort "score"; for "lowest" add order "asc"; for averages per group use group_by with sort "mean".
- Genome scores are clip "genome" (the default): leave clip out unless the question is about the plain line. "How many scores" is the count in a group_by answer.
- The six research questions, asked in these or similar words, are answered with evidence_question and the number:
  1 best genomes for each voice, 2 genomes that produce artifacts, 3 configurations that give bad MOS,
  4 configurations that consistently give good genomes, 5 how often a genome beats the plain line for each
  number of active features, 6 which genomes can be excluded. Always use evidence_question for these six.
- One listener's average or count: query_scores with listener and group_by listener, then read groups.0.mean or groups.0.count. Never group by genome for this.
- Example, one listener's average: {"tool": "query_scores", "arguments": {"listener": "Julius", "group_by": "listener", "sort": "mean"}}, then answer with groups.0.mean.
- The single lowest or highest score: query_scores with listener, sort score and order asc (or desc), and no group_by. Name the voice from the row.
- A gene type such as sigh, laughter or speed is gene_type (text). genes is only a whole number of genes, never a gene type.
- Highest score for one gene type: query_scores with gene_type, sort "score" and order "desc", and no group_by. Name the voice from the row.
- How often genomes of a dataset, voice or gene type beat the plain line: query_scores with group_by and sort beat_plain. Only the question about number of active features uses evidence_question.
- Baseline results: call results with name "v4" (or the baseline's name), and read the mean word error rate from its summary: metrics.wer.mean.
- Milestone 2 random search: call results with name "random-v2". The share of genomes that passed the checks is summary.shares.all.fine.
- group_by is one of: listener, voice, dataset, gene_type, genes, verdict, genome, pile. The number of scores is group_by listener with sort "count", never group_by "count".
- Other questions about genomes, scores or sanity checks: answer_evidence. Give its record IDs as sources. If it
  says nothing in the evidence answers the question, say so and do not guess. search_evidence lists matching records.
- Voices and datasets: datasets for counts and the voice split; voices to list validation voices.
- Searches: list_searches; one search: search_details; named genomes: named_genomes; renders: renders;
  baseline and Milestone 2 results: results; what the Dramatizer is and how to use it: about.
- How many genome searches there are, or how many are rated: overview.
- You explain the analysis; you never decide. Do not recommend a genome or name a winner: no tool decides that yet.
  Say so plainly.
- Every answer about scores says what it rests on: the number of listeners behind each number. A score from one
  listener is that listener's opinion: say "1 listener".
- Quote the numbers you were given, with the voice and genome they belong to.
- If a tool returns nothing, an error or a note that the account cannot see the data, say so plainly.
- Keep the answer short: a few sentences or a short list.
- Only answer questions about the Dramatizer. For anything else call no tool and reply exactly {"answer": "OFF_TOPIC"}.
- If the question needs a person (a choice, a recommendation, or information no tool has), reply exactly {"needs_person": "<what you need from a person>"}.

Tools:
"""


def tool_text(tools):
    lines = []
    for t in tools:
        schema = t.inputSchema or {}
        required = set(schema.get("required", []))
        params = [f"    {name} ({spec.get('type', 'string')}{', required' if name in required else ''}): "
                  f"{spec.get('description', '')}" for name, spec in schema.get("properties", {}).items()]
        lines.append(f"- {t.name}: {t.description}" + ("\n" + "\n".join(params) if params else ""))
    return "\n".join(lines)


def tool_text_of_result(result):
    return "\n".join(getattr(c, "text", "") for c in result.content if getattr(c, "text", None))


def first_json(text):
    """The first JSON object in the reply; accepts raw line breaks inside strings."""
    text = re.sub(r"```(?:json)?", "", text)
    decoder = json.JSONDecoder(strict=False)
    for start in (m.start() for m in re.finditer(r"\{", text)):
        try:
            value, _ = decoder.raw_decode(text[start:])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


async def ask(session, model, tools, question):
    names = {t.name for t in tools}
    messages = [{"role": "system", "content": INSTRUCTIONS + tool_text(tools)},
                {"role": "user", "content": question}]
    started, steps, seen = time.time(), [], set()

    def done(answer, stop):
        answer = OFF_TOPIC if str(answer).strip() == "OFF_TOPIC" else str(answer).strip()
        return {"answer": answer, "steps": steps, "stop": stop, "seconds": round(time.time() - started, 1)}

    for _ in range(MAX_STEPS + 1):
        reply = model.chat(messages)
        call = first_json(reply)
        if call and "tool" in call:
            name, arguments = str(call["tool"]), call.get("arguments") or {}
            key = (name, json.dumps(arguments, sort_keys=True))
            if len(steps) >= MAX_STEPS:
                return done("I could not finish: too many tool calls for one question.", "step budget")
            if key in seen:
                return done("I could not finish: the same tool call was repeated.", "repeated call")
            if time.time() - started > TIME_BUDGET:
                return done("I could not finish: the time limit was reached.", "time budget")
            if name not in names:
                text, error = f"Error: there is no tool named {name}. Tools: {', '.join(sorted(names))}", True
            else:
                seen.add(key)
                result = await session.call_tool(name, arguments)
                text, error = tool_text_of_result(result), bool(result.isError)
            steps.append({"tool": name, "arguments": arguments, "result": text, "error": error})
            shown = text[:RESULT_CHARS] + (" ... (cut)" if len(text) > RESULT_CHARS else "")
            messages += [{"role": "assistant", "content": json.dumps({"tool": name, "arguments": arguments})},
                         {"role": "user", "content": f"Result of {name}:\n{shown}\n\nCall another tool, or reply "
                                                     '{"answer": "..."} using only these results.'}]
            continue
        if call and "needs_person" in call:
            return done("Needs a person: " + str(call["needs_person"]), "needs a person")
        return done(call.get("answer") if call and "answer" in call else reply, "answer")
    return done("I could not finish.", "step budget")


async def run(args):
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)], env=os.environ.copy())
    model = ServerModel(args.server, args.model)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = (await session.list_tools()).tools
            return await ask(session, model, tools, args.ask)


def write_trace(args, out):
    TRACES.mkdir(parents=True, exist_ok=True)
    entry = {"time": datetime.now().isoformat(timespec="seconds"), "model": args.model, "question": args.ask, **out}
    with (TRACES / "runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ask the Dramatizer through its MCP tools (Gemma 4 chooses).")
    parser.add_argument("--ask", required=True)
    parser.add_argument("--server", required=True, help="an OpenAI-compatible address, e.g. http://127.0.0.1:11434/v1")
    parser.add_argument("--model", default="gemma4-8k")
    parser.add_argument("--steps", action="store_true", help="show the tool calls")
    args = parser.parse_args(argv)
    out = asyncio.run(run(args))
    write_trace(args, out)
    if args.steps:
        for s in out["steps"]:
            flag = " (error)" if s["error"] else ""
            print(f"  [{s['tool']} {json.dumps(s['arguments'])}] -> {len(s['result'])} characters{flag}")
    print(f"\nDramatizer: {out['answer']}\n  ({out['seconds']} s, {len(out['steps'])} tool calls, stopped: {out['stop']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
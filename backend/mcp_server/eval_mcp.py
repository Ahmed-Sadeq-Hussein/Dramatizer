"""eval_mcp.py | Step 6: chatbot_eval's questions and scoring, answered through the MCP tools (client_loop.py).

    "$D" backend/mcp_server/eval_mcp.py --server http://127.0.0.1:11434/v1 --model gemma4-8k
    "$D" backend/mcp_server/eval_mcp.py --server http://127.0.0.1:11434/v1 --model gemma4-8k --only s01,q5

The truth is computed from the API, as in chatbot_eval.py. The report goes to history/mcp_evals/ (git-ignored).
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "API calls"))
from API_REQUEST import Results  # noqa: E402
from chatbot import ServerModel  # noqa: E402
from chatbot_eval import judge  # noqa: E402
from client_loop import SERVER, ask  # noqa: E402

QUESTIONS = ROOT / "API calls" / "chatbot_questions.json"
REPORTS = ROOT / "history" / "mcp_evals"

# The MCP server answers baseline and Milestone 2 questions with one tool, results
EQUIVALENT = {"baseline_results": "results", "milestone2_results": "results"}


async def evaluate(args, questions):
    params = StdioServerParameters(command=sys.executable, args=[str(SERVER)], env=os.environ.copy())
    model = ServerModel(args.server, args.model)
    rows = []
    with Results.connect() as results:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
                names = {t.name for t in tools}
                for q in questions:
                    truth_data = json.loads(results.run_tool(q["truth"]["tool"], q["truth"]["arguments"]))
                    out = await ask(session, model, tools, q["question"])
                    used = [s["tool"] for s in out["steps"]]
                    missing = judge(q, out["answer"], truth_data)
                    expected = EQUIVALENT.get(q["tool"], q["tool"])
                    row = {"id": q["id"], "kind": q["kind"], "question": q["question"],
                           "tool_ok": expected in used, "tool_in_mcp": expected in names,
                           "answer_ok": not missing and "error" not in truth_data,
                           "missing": missing, "tools": used, "stop": out["stop"],
                           "calls": [{"tool": s["tool"], "arguments": s["arguments"]} for s in out["steps"]],
                           "steps": [{"tool": s["tool"], "result": s["result"][:2000]} for s in out["steps"]],
                           "answer": out["answer"], "seconds": out["seconds"]}
                    rows.append(row)
                    print(f"{q['id']:4s} tool {'ok ' if row['tool_ok'] else 'NO '} "
                          f"answer {'ok ' if row['answer_ok'] else 'NO '}{out['seconds']:7.1f}s  {q['question']}"
                          + (f"\n       missing: {'; '.join(missing)}" if missing else ""), flush=True)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score the MCP chatbot on chatbot_questions.json.")
    parser.add_argument("--server", required=True, help="e.g. http://127.0.0.1:11434/v1")
    parser.add_argument("--model", default="gemma4-8k")
    parser.add_argument("--only", help="question ids, comma-separated")
    args = parser.parse_args(argv)
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    rows = asyncio.run(evaluate(args, questions))
    kinds = sorted({r["kind"] for r in rows})
    summary = {"model": args.model, "made": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "questions": len(rows), "tool_ok": sum(r["tool_ok"] for r in rows),
               "answer_ok": sum(r["answer_ok"] for r in rows),
               "not_in_mcp": sum(not r["tool_in_mcp"] for r in rows),
               "by_kind": {k: {"questions": sum(r["kind"] == k for r in rows),
                               "tool_ok": sum(r["tool_ok"] for r in rows if r["kind"] == k),
                               "answer_ok": sum(r["answer_ok"] for r in rows if r["kind"] == k)} for k in kinds},
               "mean_seconds": round(sum(r["seconds"] for r in rows) / max(1, len(rows)), 1)}
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"mcp_eval_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{summary['tool_ok']}/{summary['questions']} right tool, {summary['answer_ok']}/{summary['questions']} "
          f"right answer, {summary['not_in_mcp']} questions need a tool MCP does not have yet, "
          f"{summary['mean_seconds']} s a question.\nBy kind: {summary['by_kind']}\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
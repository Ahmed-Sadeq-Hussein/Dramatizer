"""chatbot_eval.py | How well the chatbot answers: the right tool, and the right numbers in its answer.

    D:/envs/llm/python.exe chatbot_eval.py                          # every question, with gemma-4-E4B-it
    D:/envs/llm/python.exe chatbot_eval.py --model gemma-4-E2B-it   # compare a smaller model
    D:/envs/llm/python.exe chatbot_eval.py --only s01,q6            # some questions

The questions are in chatbot_questions.json. For each one the right answer is computed at test time from the
data, by calling the tool that holds it (truth), so the test stays right as listeners add scores. Two marks:
    tool    the chatbot called the expected tool
    answer  its answer holds every value the truth asks for (numbers within rounding: 0.671 = 67% = 67.1%),
            or, for the six questions, at least half of the numbers in the chart's own answer
The report goes to history/chatbot/ (git-ignored: it holds listeners' scores). Needs an admin account or an
API key, like the chatbot.
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from API_REQUEST import Results  # noqa: E402
from chatbot import Chatbot, make_model  # noqa: E402

QUESTIONS = HERE / "chatbot_questions.json"
REPORTS = HERE.parent / "history" / "chatbot"
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def values_at(data, path):
    """The values at a dotted path; '*' takes every item of a list."""
    found = [data]
    for part in path.split("."):
        nxt = []
        for value in found:
            if part == "*" and isinstance(value, list):
                nxt += value
            elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
                nxt.append(value[int(part)])
            elif isinstance(value, dict) and part in value:
                nxt.append(value[part])
        found = nxt
    return found


def numbers_in(text):
    return [float(n) for n in NUMBER.findall(text.replace(",", ""))]


def has_number(answer, value):
    """The number is in the answer, allowing rounding and percent (0.671 ~ 67%, 67.1%, 0.67)."""
    targets = [value, value * 100] if abs(value) <= 1 else [value]
    for n in numbers_in(answer):
        for t in targets:
            if abs(n - t) <= max(0.051 * (1 if abs(t) < 10 else 10), 0.006 * abs(t)):
                return True
    return False


def has_text(answer, value):
    a, v = answer.lower(), str(value).lower()
    return v in a or ("/" in v and v.split("/")[-1] in a)


def has(answer, value):
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return has_number(answer, float(value))
    return has_text(answer, value)


def judge(question, answer, truth_data):
    truth = question["truth"]
    missing = []
    for path in truth.get("must", []):
        values = values_at(truth_data, path)
        if not values:
            missing.append(f"{path}: no data")
        missing += [f"{path}={v}" for v in values if not has(answer, v)]
    if truth.get("any_tied"):
        listing = truth_data.get(truth.get("list", "rows")) or []
        key = truth.get("tie_key", (truth_data.get("sort") or "score"))
        top = listing[0].get(key) if listing else None
        tied = [r[truth["any_tied"]] for r in listing if r.get(key) == top]
        if tied and not any(has(answer, v) for v in tied):
            missing.append(f"one of the tied {truth['any_tied']}s {sorted(set(map(str, tied)))[:6]}")
    if truth.get("claim"):
        claim = str(truth_data.get(truth["claim"]) or "")
        wanted = list(dict.fromkeys(numbers_in(claim)))
        if wanted:
            hit = sum(has_number(answer, n) for n in wanted)
            if hit * 2 < len(wanted):
                missing.append(f"{hit} of {len(wanted)} numbers of the chart's answer")
        elif not claim:
            missing.append("the chart gave no answer")
    return missing


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score the chatbot on chatbot_questions.json.")
    parser.add_argument("--model", default="gemma-4-E4B-it")
    parser.add_argument("--server")
    parser.add_argument("--only", help="question ids, comma-separated")
    args = parser.parse_args(argv)
    questions = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    if args.only:
        wanted = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    with Results.connect() as results:
        bot = Chatbot(results, make_model(args))
        rows = []
        for q in questions:
            truth_data = json.loads(results.run_tool(q["truth"]["tool"], q["truth"]["arguments"]))
            out = bot.ask(q["question"])
            used = [s["tool"] for s in out["steps"]]
            missing = judge(q, out["answer"], truth_data)
            row = {"id": q["id"], "kind": q["kind"], "question": q["question"], "tool_ok": q["tool"] in used,
                   "answer_ok": not missing and "error" not in truth_data, "missing": missing, "tools": used,
                   "calls": [{"tool": s["tool"], "arguments": s["arguments"]} for s in out["steps"]],
                   "answer": out["answer"], "seconds": out["seconds"]}
            rows.append(row)
            print(f"{q['id']:4s} tool {'ok ' if row['tool_ok'] else 'NO '} answer {'ok ' if row['answer_ok'] else 'NO '}"
                  f"{out['seconds']:6.1f}s  {q['question']}" + (f"\n       missing: {'; '.join(missing)}" if missing else ""),
                  flush=True)
    kinds = sorted({r["kind"] for r in rows})
    summary = {"model": args.model, "server": args.server, "made": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
               "questions": len(rows), "tool_ok": sum(r["tool_ok"] for r in rows),
               "answer_ok": sum(r["answer_ok"] for r in rows),
               "by_kind": {k: {"questions": sum(r["kind"] == k for r in rows),
                               "tool_ok": sum(r["tool_ok"] for r in rows if r["kind"] == k),
                               "answer_ok": sum(r["answer_ok"] for r in rows if r["kind"] == k)} for k in kinds},
               "mean_seconds": round(sum(r["seconds"] for r in rows) / max(1, len(rows)), 1)}
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"eval_{args.model}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n{summary['tool_ok']}/{summary['questions']} right tool, {summary['answer_ok']}/{summary['questions']} right "
          f"answer, {summary['mean_seconds']} s a question. By kind: {summary['by_kind']}\nReport: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

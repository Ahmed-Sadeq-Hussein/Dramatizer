"""baseline.py | The lexical baseline: BM25 over the evidence corpus, returning source IDs with their revisions.

    python main_backend.py rag ask "What did genome 5 of voice CREMA-D/1014 do?"
    python main_backend.py rag ask "..." --k 5 --kind clip     # only one kind of record
    python main_backend.py rag run                              # every question in questions.json -> history/rag/

BM25 (Okapi, k1 = 1.5, b = 0.75) scores each record by the question's words it
contains: a rare word counts more than a common one, a word repeated in a record
counts less each time, and a long record is not favoured for being long. Nothing
is learned and nothing is sent anywhere: the same question on the same corpus
always gives the same ranking, so it is the baseline the variant is compared to.

Words are lower-cased and split on anything that is not a letter or digit, so
"vocalized_noise", "CREMA-D/1014" and "genome 5" become searchable words. A few
very common English words are left out. Voice names keep their number together
with their dataset: "CREMA-D/1014" also gives the word "cremad1014", so the right
voice ranks above one that only shares the number.

The baseline only retrieves. Deciding that nothing found is good enough (safe
absence) and writing an answer from the records come in later steps.
"""
import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from backend import safe_files
from backend.rag import corpus as evidence

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"
RESULTS_PATH = evidence.RAG_DIR / "baseline_results.jsonl"
K1, B = 1.5, 0.75
TOP_K = 10
STOP = set("a an and are as at be by did do does for from has have how i in is it its of on or so that the their "
           "them they this to was were what when which who why with".split())
VOICE = re.compile(r"\b(crema-d|libritts-r|vctk)[/_ ]?(p?\d+)\b", re.IGNORECASE)


def words(text):
    """The searchable words of a text: lower case, letters and digits, no stop words, plus joined voice names."""
    text = str(text)
    out = [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP]
    out += [(m.group(1) + m.group(2)).replace("-", "").lower() for m in VOICE.finditer(text)]
    return out


class BM25:
    """An index of records, searched by BM25."""

    def __init__(self, records, k1=K1, b=B):
        self.records, self.k1, self.b = records, k1, b
        self.docs = [Counter(words(r["title"] + " " + r["text"])) for r in records]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.average = sum(self.lengths) / max(1, len(self.lengths))
        frequency = Counter(w for d in self.docs for w in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in frequency.items()}

    def score(self, query_words, i):
        doc, length, total = self.docs[i], self.lengths[i], 0.0
        for w in query_words:
            tf = doc.get(w)
            if tf:
                total += self.idf[w] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / self.average))
        return total

    def search(self, question, k=TOP_K, kind=None):
        """The k best records for a question: [{"rank", "id", "kind", "revision", "score", "title"}], best first."""
        query = list(dict.fromkeys(words(question)))       # each word of the question counts once
        scored = [(self.score(query, i), i) for i, r in enumerate(self.records) if kind in (None, r["kind"])]
        scored = [(s, i) for s, i in scored if s > 0]
        scored.sort(key=lambda x: (-x[0], self.records[x[1]]["id"]))   # ties broken by ID, so runs repeat exactly
        return [{"rank": rank, "id": self.records[i]["id"], "kind": self.records[i]["kind"],
                 "revision": self.records[i]["revision"], "score": round(s, 4), "title": self.records[i]["title"]}
                for rank, (s, i) in enumerate(scored[:k], start=1)]


def load_questions(path=QUESTIONS_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))["questions"]


def run_all(index, questions, k=TOP_K, path=RESULTS_PATH):
    """Every question through the baseline; one line per question in history/rag/baseline_results.jsonl."""
    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    for q in questions:
        hits = index.search(q["question"], k=k)
        lines.append({"time": when, "retriever": "bm25", "k": k, "question_id": q["id"], "question": q["question"],
                      "supported": q["supported"], "top_score": hits[0]["score"] if hits else 0.0, "hits": hits})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = safe_files.temporary_for(path)
    temporary.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines), encoding="utf-8")
    safe_files.replace(temporary, path)
    return lines


def show(hits):
    for h in hits:
        print(f"  {h['rank']:2d}. {h['score']:7.3f}  {h['id']}  (revision {h['revision']})\n               {h['title']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="The lexical (BM25) baseline retriever over the evidence corpus.")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="the best records for one question")
    ask.add_argument("question")
    ask.add_argument("--k", type=int, default=TOP_K)
    ask.add_argument("--kind", choices=["clip", "run", "sanity"])
    every = sub.add_parser("run", help="every question in questions.json, saved to history/rag/baseline_results.jsonl")
    every.add_argument("--k", type=int, default=TOP_K)
    args = parser.parse_args(argv)

    index = BM25(evidence.load())
    if args.command == "ask":
        hits = index.search(args.question, k=args.k, kind=args.kind)
        print(f"{args.question}\n" + ("  nothing in the evidence shares a word with this question" if not hits else ""))
        show(hits)
    else:
        for line in run_all(index, load_questions(), k=args.k):
            print(f"\n{line['question_id']} [{'supported' if line['supported'] is True else line['supported'] or 'unsupported'}]"
                  f" {line['question']}")
            show(line["hits"][:5])
        print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()

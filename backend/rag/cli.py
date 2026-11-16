"""cli.py | The evidence system's commands, as `python main_backend.py rag ...`.

    python main_backend.py rag corpus                 # build the evidence corpus (history/rag/corpus.jsonl)
    python main_backend.py rag label                  # label which records answer each question (backend/rag/labels.json)
    python main_backend.py rag label --show q07       # one question's labelled records
    python main_backend.py rag refresh                # after new ratings: rebuild, report changes, flag stale labels
    python main_backend.py rag refresh --check        # only say whether the corpus is out of date
    python main_backend.py rag ask "a question"       # the baseline's best records for one question
    python main_backend.py rag run                    # every question in backend/rag/questions.json
    python main_backend.py rag ask "a question" --retriever hybrid   # the variant: BM25 + MiniLM (step 4)
    python main_backend.py rag compare [--show 5]     # bm25, dense and hybrid scored with hit@k and MRR (step 5)
    python main_backend.py rag answer "a question"    # an answer citing its records, or "no evidence" (safe absence)
    python main_backend.py rag answer "..." --as USERNAME   # only the evidence that account may see
    python main_backend.py rag answer --check         # the 10 questions and the paraphrases, checked
"""
import sys

from backend.rag import answer, baseline, compare, corpus, labels, refresh

COMMANDS = {"corpus": corpus.main, "label": labels.main, "refresh": refresh.main, "compare": compare.main, "answer": answer.main}


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    if argv[0] in COMMANDS:
        COMMANDS[argv[0]](argv[1:])
    else:
        baseline.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])

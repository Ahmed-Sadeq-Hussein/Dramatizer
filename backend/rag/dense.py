"""dense.py | Step 4 of the evidence milestone: the variant, a dense (MiniLM) search and a hybrid of it with BM25.

    python main_backend.py rag ask "Which genomes produce artifacts?" --retriever hybrid
    python main_backend.py rag ask "..." --retriever dense

Dense: each record (title + text) and the question are turned into a vector by MiniLM
(sentence-transformers/all-MiniLM-L6-v2, from LLM_LIBRARY_DIR: mean of the token vectors,
length 1), and records are ranked by cosine similarity. It matches meaning, not spelling:
"artifacts" finds "artifact", "bad" finds "worse than the plain line". It is weak where BM25
is strong: exact names and numbers ("genome 5 of CREMA-D/1014").

Hybrid: BM25 and dense each rank the records, and the two rankings are joined by reciprocal
rank fusion: a record scores 1 / (60 + rank) in each list it appears in, summed. 60 is the
usual constant from the paper that introduced it (Cormack et al., 2009). It was fixed before
the comparison and not tuned on the answer key, so the comparison in compare.py stays fair.

Every setting is fixed in advance and nothing is learned from the labels. The record vectors
are kept in history/rag/embeddings/, named by the model and a fingerprint of the record texts,
so they are made again only when the corpus changes. MiniLM reads at most 256 word pieces of
a record: the long run records are cut, which is one of the things the comparison shows.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from backend.rag import baseline
from backend.rag import corpus as evidence

MODEL_NAME = "all-MiniLM-L6-v2"
MAX_TOKENS = 256
RRF_K = 60                      # reciprocal rank fusion constant, fixed in advance
POOL = 100                      # each retriever's top records that go into the fusion
EMBEDDINGS_DIR = evidence.RAG_DIR / "embeddings"


def model_dir():
    from backend import model
    return Path(model.setting("LLM_LIBRARY_DIR")) / MODEL_NAME


class MiniLM:
    """Texts -> unit vectors, with MiniLM on the GPU if there is one."""

    def __init__(self, path=None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        path = Path(path or model_dir())
        if not (path / "model.safetensors").is_file():
            raise SystemExit(f"MiniLM is not at {path}: download sentence-transformers/{MODEL_NAME} there.")
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path).to(self.device).eval()

    def __call__(self, texts, batch=64):
        out = []
        with self.torch.inference_mode():
            for i in range(0, len(texts), batch):
                t = self.tokenizer(texts[i:i + batch], padding=True, truncation=True, max_length=MAX_TOKENS,
                                   return_tensors="pt").to(self.device)
                tokens = self.model(**t).last_hidden_state
                mask = t["attention_mask"].unsqueeze(-1).to(tokens.dtype)
                v = (tokens * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                out.append(self.torch.nn.functional.normalize(v, dim=1).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 384), dtype=np.float32)


def record_text(r):
    return r["title"] + ". " + r["text"]


def fingerprint(records, name=MODEL_NAME):
    h = hashlib.sha256(name.encode())
    for r in records:
        h.update(json.dumps([r["id"], record_text(r)], ensure_ascii=False).encode())
    return h.hexdigest()[:16]


def hit(records, i, rank, score):
    r = records[i]
    return {"rank": rank, "id": r["id"], "kind": r["kind"], "revision": r["revision"], "score": round(float(score), 4),
            "title": r["title"]}


class Dense:
    """Records ranked by cosine similarity of MiniLM vectors. `encode` can be replaced (the tests use a fake)."""

    name = "dense"

    def __init__(self, records, encode=None, cache_dir=EMBEDDINGS_DIR):
        self.records = records
        self.encode = encode or MiniLM()
        cache = Path(cache_dir) / f"{MODEL_NAME}_{fingerprint(records)}.npy" if cache_dir else None
        if cache and cache.is_file():
            self.vectors = np.load(cache)
        else:
            self.vectors = self.encode([record_text(r) for r in records])
            if cache:
                cache.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache, self.vectors)

    def ranked(self, question, kind=None):
        """[(score, record index)] for every record of the kind, best first, ties by ID."""
        q = self.encode([question])[0]
        scores = self.vectors @ q
        out = [(float(scores[i]), i) for i, r in enumerate(self.records) if kind in (None, r["kind"])]
        out.sort(key=lambda x: (-x[0], self.records[x[1]]["id"]))
        return out

    def search(self, question, k=baseline.TOP_K, kind=None):
        return [hit(self.records, i, rank, s) for rank, (s, i) in enumerate(self.ranked(question, kind)[:k], start=1)]


class Hybrid:
    """BM25 and dense joined by reciprocal rank fusion."""

    name = "hybrid"

    def __init__(self, records, bm25=None, dense=None, rrf_k=RRF_K, pool=POOL):
        self.records = records
        self.bm25 = bm25 or baseline.BM25(records)
        self.dense = dense or Dense(records)
        self.rrf_k, self.pool = rrf_k, pool
        self.position = {r["id"]: i for i, r in enumerate(records)}

    def search(self, question, k=baseline.TOP_K, kind=None):
        fused = {}
        lexical = [self.position[h["id"]] for h in self.bm25.search(question, k=self.pool, kind=kind)]
        semantic = [i for _, i in self.dense.ranked(question, kind)[:self.pool]]
        for ranking in (lexical, semantic):
            for rank, i in enumerate(ranking, start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + rank)
        order = sorted(fused, key=lambda i: (-fused[i], self.records[i]["id"]))
        return [hit(self.records, i, rank, fused[i]) for rank, i in enumerate(order[:k], start=1)]


def make(name, records, encode=None):
    """A retriever by name: bm25 (the baseline), dense or hybrid."""
    if name == "bm25":
        return baseline.BM25(records)
    if name == "dense":
        return Dense(records, encode=encode)
    if name == "hybrid":
        bm25 = baseline.BM25(records)
        return Hybrid(records, bm25=bm25, dense=Dense(records, encode=encode))
    raise ValueError(f"No retriever {name!r}: bm25, dense or hybrid.")

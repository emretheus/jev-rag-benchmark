from __future__ import annotations

import math
from collections import Counter, defaultdict

from .text import tokenize


class BM25:
    """Okapi BM25 with an inverted index; only documents matching query terms are scored."""

    def __init__(
        self,
        doc_ids: list[str],
        texts: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if len(doc_ids) != len(texts):
            raise ValueError("doc_ids and texts must have the same length")
        self.doc_ids = list(doc_ids)
        self.k1 = k1
        self.b = b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len: list[int] = []
        for index, text in enumerate(texts):
            tokens = tokenize(text)
            self.doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings[term].append((index, tf))
        self.n_docs = len(self.doc_ids)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.idf: dict[str, float] = {}
        for term, postings in self.postings.items():
            df = len(postings)
            self.idf[term] = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_index, tf in self.postings[term]:
                length_norm = 1 - self.b + self.b * self.doc_len[doc_index] / (self.avgdl or 1.0)
                scores[doc_index] += idf * tf * (self.k1 + 1) / (tf + self.k1 * length_norm)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], self.doc_ids[item[0]]))
        return [(self.doc_ids[index], score) for index, score in ranked[:top_k]]

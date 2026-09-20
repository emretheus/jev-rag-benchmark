from __future__ import annotations

import re
import string
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")
CITATION_RE = re.compile(r"\[([^\[\]]{1,64})\]")
SPACE_RE = re.compile(r"\s+")

ARTICLES = {"a", "an", "the"}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def strip_citations(text: str) -> str:
    return CITATION_RE.sub(" ", text)


def cited_ids(text: str) -> list[str]:
    return CITATION_RE.findall(text)


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    cleaned = "".join(ch if ch not in string.punctuation else " " for ch in lowered)
    tokens = [tok for tok in SPACE_RE.split(cleaned) if tok and tok not in ARTICLES]
    return " ".join(tokens)


def exact_match(prediction: str, golds: list[str]) -> float:
    pred = normalize_answer(strip_citations(prediction))
    if not golds:
        return 0.0
    return float(any(pred == normalize_answer(gold) for gold in golds))


def token_f1(prediction: str, golds: list[str]) -> float:
    pred_tokens = normalize_answer(strip_citations(prediction)).split()
    if not pred_tokens or not golds:
        return 0.0
    best = 0.0
    for gold in golds:
        gold_tokens = normalize_answer(gold).split()
        if not gold_tokens:
            continue
        overlap = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part for part in parts if part.strip()]

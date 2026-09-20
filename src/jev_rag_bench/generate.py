from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .clients.nvidia import ChatResult, OpenAIChat
from .text import cited_ids, exact_match, normalize_answer, strip_citations, token_f1

ABSTAIN_MARKER = "INSUFFICIENT_CONTEXT"

PROMPT_TEMPLATE = (
    "You answer questions using only the numbered context passages.\n"
    "Rules:\n"
    "- Use only information from the passages.\n"
    "- Cite the passage identifiers you used in square brackets, e.g. [doc-123].\n"
    f"- If the passages do not contain the answer, reply exactly: {ABSTAIN_MARKER}\n\n"
    "Question: {question}\n\n"
    "Context passages:\n{context}\n\n"
    "Answer:"
)


def build_answer_messages(
    question: str,
    contexts: list[tuple[str, str]],
    max_context_chars: int = 6000,
) -> list[dict]:
    lines: list[str] = []
    used = 0
    for doc_id, text in contexts:
        block = f"[{doc_id}] {text}"
        if lines and used + len(block) > max_context_chars:
            break
        lines.append(block)
        used += len(block)
    context = "\n\n".join(lines) if lines else "(none)"
    return [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question, context=context)}]


def is_abstention(text: str) -> bool:
    normalized = normalize_answer(strip_citations(text))
    if not normalized:
        return True
    patterns = [
        normalize_answer(ABSTAIN_MARKER),
        normalize_answer("I don't know"),
        normalize_answer("I do not know"),
        normalize_answer("cannot answer"),
    ]
    return any(pattern in normalized for pattern in patterns)


class ChatGenerator:
    def __init__(self, chat: OpenAIChat, max_context_chars: int = 6000) -> None:
        self.chat = chat
        self.max_context_chars = max_context_chars

    def answer(self, question: str, contexts: list[tuple[str, str]]) -> ChatResult:
        messages = build_answer_messages(question, contexts, self.max_context_chars)
        return self.chat.generate(messages)


def load_rows(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def replay_generator(
    results_path: str | Path,
    output_path: str | Path,
    branch: str,
    generator,
    corpus_texts: dict[str, str],
    run_kind: str = "real",
    top_k: int = 5,
    limit: int | None = None,
    concurrency: int = 1,
) -> dict:
    rows = load_rows(results_path)
    if limit is not None:
        rows = rows[:limit]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if output_path.exists():
        for row in load_rows(output_path):
            done.add(str(row.get("query_id")))

    pending = [row for row in rows if str(row["query_id"]) not in done]
    stats = {
        "total": len(rows),
        "already_done": len(rows) - len(pending),
        "written": 0,
        "failed": 0,
    }
    lock = threading.Lock()

    def work(row: dict) -> dict:
        branch_row = row["branches"][branch]
        order = list(branch_row["order"])[:top_k]
        contexts = [(doc_id, corpus_texts.get(doc_id, "")) for doc_id in order]
        result = generator.answer(row["question"], contexts)
        golds = row.get("answers") or []
        citations = cited_ids(result.text)
        valid_ids = {doc_id for doc_id, _ in contexts}
        return {
            "query_id": row["query_id"],
            "dataset": row.get("dataset"),
            "run_kind": run_kind,
            "branch": branch,
            "model": result.resolved_model,
            "text": result.text,
            "citations": citations,
            "invalid_citations": [c for c in citations if c not in valid_ids],
            "abstained": is_abstention(result.text),
            "em": exact_match(result.text, golds) if golds else None,
            "f1": token_f1(result.text, golds) if golds else None,
            "latency_ms": round(result.latency_ms, 3),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "finish_reason": result.finish_reason,
        }

    def emit(record: dict) -> None:
        with lock:
            with open(output_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["written"] += 1

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(work, row): row for row in pending}
            for future in as_completed(futures):
                row = futures[future]
                try:
                    emit(future.result())
                except Exception as error:  # noqa: BLE001
                    stats["failed"] += 1
                    print(f"  generation failed for {row['query_id']}: {error}", flush=True)
    else:
        for row in pending:
            try:
                emit(work(row))
            except Exception as error:  # noqa: BLE001
                stats["failed"] += 1
                print(f"  generation failed for {row['query_id']}: {error}", flush=True)

    return stats

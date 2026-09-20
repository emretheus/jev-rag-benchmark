from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import httpx

XQUAD_EN_URL = "https://raw.githubusercontent.com/deepmind/xquad/master/xquad.en.json"
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"

ALL_DATASETS = ["xquad-en", "scifact"]

DATASET_SOURCES: dict[str, dict] = {
    "xquad-en": {
        "urls": [XQUAD_EN_URL],
        "license": "CC BY-SA 4.0 (XQuAD, Artetxe et al. 2020)",
    },
    "scifact": {
        "urls": [SCIFACT_URL],
        "license": "Abstracts ODC-By 1.0; annotations CC BY 4.0 (SciFact / BEIR)",
    },
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def download(url: str) -> bytes:
    with httpx.Client(timeout=180.0, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.content


def normalize_xquad(raw: dict) -> tuple[list[dict], list[dict]]:
    docs: dict[str, dict] = {}
    queries: list[dict] = []
    for article in raw.get("data", []):
        title = str(article.get("title", "")).strip()
        for paragraph in article.get("paragraphs", []):
            context = str(paragraph.get("context", "")).strip()
            if not context:
                continue
            doc_id = "xq-" + hashlib.sha1(context.encode("utf-8")).hexdigest()[:12]
            docs.setdefault(doc_id, {"doc_id": doc_id, "title": title, "text": context})
            for qa in paragraph.get("qas", []):
                answers = [
                    str(answer["text"]).strip()
                    for answer in qa.get("answers", [])
                    if str(answer.get("text", "")).strip()
                ]
                question = str(qa.get("question", "")).strip()
                if not answers or not question:
                    continue
                queries.append(
                    {
                        "query_id": str(qa.get("id")),
                        "question": question,
                        "gold_doc_ids": [doc_id],
                        "answers": answers,
                    }
                )
    return list(docs.values()), queries


def normalize_scifact(
    corpus_text: str,
    queries_text: str,
    qrels_text: str,
) -> tuple[list[dict], list[dict]]:
    docs = []
    for line in corpus_text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        docs.append(
            {
                "doc_id": str(row["_id"]),
                "title": str(row.get("title") or "").strip(),
                "text": str(row.get("text") or "").strip(),
            }
        )

    queries: list[dict] = []
    for line in queries_text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        queries.append(
            {
                "query_id": str(row["_id"]),
                "question": str(row.get("text", "")).strip(),
                "gold_doc_ids": [],
                "answers": [],
            }
        )

    by_id = {query["query_id"]: query for query in queries}
    for line in qrels_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("query"):
            continue
        parts = stripped.split("\t")
        if len(parts) < 3:
            continue
        query_id, doc_id, score = parts[0], parts[1], float(parts[2])
        if score > 0 and query_id in by_id:
            by_id[query_id]["gold_doc_ids"].append(doc_id)

    return docs, [query for query in queries if query["gold_doc_ids"]]


def write_processed(
    out_dir: Path,
    docs: list[dict],
    queries: list[dict],
    meta: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "corpus.jsonl", "w", encoding="utf-8") as handle:
        for row in docs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out_dir / "queries.jsonl", "w", encoding="utf-8") as handle:
        for row in queries:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out_dir / "meta.json", "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)


def prepare_dataset(dataset: str, data_root: Path) -> dict:
    if dataset not in DATASET_SOURCES:
        raise ValueError(f"unknown dataset: {dataset}")
    raw_dir = data_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = data_root / "processed" / dataset

    if dataset == "xquad-en":
        payload = download(XQUAD_EN_URL)
        (raw_dir / "xquad.en.json").write_bytes(payload)
        docs, queries = normalize_xquad(json.loads(payload))
        raw_hashes = {"xquad.en.json": sha256_bytes(payload)}
    else:
        payload = download(SCIFACT_URL)
        (raw_dir / "scifact.zip").write_bytes(payload)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            names = archive.namelist()
            corpus_name = next(name for name in names if name.endswith("corpus.jsonl"))
            queries_name = next(name for name in names if name.endswith("queries.jsonl"))
            qrels_name = next(name for name in names if name.endswith("qrels/test.tsv"))
            docs, queries = normalize_scifact(
                archive.read(corpus_name).decode("utf-8"),
                archive.read(queries_name).decode("utf-8"),
                archive.read(qrels_name).decode("utf-8"),
            )
        raw_hashes = {"scifact.zip": sha256_bytes(payload)}

    meta = {
        "dataset": dataset,
        "sources": DATASET_SOURCES[dataset]["urls"],
        "license": DATASET_SOURCES[dataset]["license"],
        "counts": {"documents": len(docs), "queries": len(queries)},
        "raw_sha256": raw_hashes,
    }
    write_processed(out_dir, docs, queries, meta)
    return meta


def load_processed(dataset: str, data_root: Path) -> tuple[list[dict], list[dict]]:
    out_dir = data_root / "processed" / dataset
    corpus_path = out_dir / "corpus.jsonl"
    queries_path = out_dir / "queries.jsonl"
    if not corpus_path.exists() or not queries_path.exists():
        raise FileNotFoundError(
            f"processed dataset '{dataset}' not found; "
            f"run: jev-rag data prepare --dataset {dataset}"
        )
    docs = [
        json.loads(line)
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    queries = [
        json.loads(line)
        for line in queries_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return docs, queries

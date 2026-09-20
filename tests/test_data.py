from __future__ import annotations

from jev_rag_bench.data import normalize_scifact, normalize_xquad


def test_normalize_xquad_extracts_documents_queries_and_answers():
    raw = {
        "data": [
            {
                "title": "France",
                "paragraphs": [
                    {
                        "context": "Paris is the capital and most populous city of France.",
                        "qas": [
                            {
                                "id": "q1",
                                "question": "What is the capital of France?",
                                "answers": [{"text": "Paris", "answer_start": 0}],
                            },
                            {
                                "id": "q2",
                                "question": "Unanswerable question",
                                "answers": [],
                            },
                        ],
                    },
                    {
                        "context": "Mars is the fourth planet from the Sun.",
                        "qas": [
                            {
                                "id": "q3",
                                "question": "Which planet is fourth from the Sun?",
                                "answers": [{"text": "Mars", "answer_start": 0}],
                            }
                        ],
                    },
                ],
            }
        ]
    }
    docs, queries = normalize_xquad(raw)
    assert len(docs) == 2
    assert len(queries) == 2
    by_id = {query["query_id"]: query for query in queries}
    assert by_id["q1"]["gold_doc_ids"] == [docs[0]["doc_id"]]
    assert by_id["q1"]["answers"] == ["Paris"]
    assert set(by_id) == {"q1", "q3"}


def test_normalize_scifact_joins_qrels_and_keeps_supported_queries():
    corpus = "\n".join(
        [
            '{"_id": "1", "title": "T1", "text": "Alpha text."}',
            '{"_id": "2", "title": "T2", "text": "Beta text."}',
        ]
    )
    queries = "\n".join(
        [
            '{"_id": "q1", "text": "alpha?"}',
            '{"_id": "q2", "text": "unsupported?"}',
        ]
    )
    qrels = "\n".join(["query-id\tcorpus-id\tscore", "q1\t1\t1", "q2\t2\t0"])
    docs, parsed_queries = normalize_scifact(corpus, queries, qrels)
    assert [doc["doc_id"] for doc in docs] == ["1", "2"]
    assert len(parsed_queries) == 1
    assert parsed_queries[0]["gold_doc_ids"] == ["1"]

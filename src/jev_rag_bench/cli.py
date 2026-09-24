from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import data as data_module
from .answerability import run_answerability, summarize_answerability
from .audit import load_audits, retrieval_audit, write_audit
from .bm25 import BM25
from .charts import render_answerability_charts, render_charts, render_toolbench_charts
from .clients.systemone import build_state
from .config import api_key, load_config, load_dotenv
from .generate import ChatGenerator, load_rows, replay_generator
from .publish import (
    find_summaries,
    inject_into_readme,
    load_summaries,
    push_to_huggingface,
    render_readme_block,
    stage_published,
    validate_summaries,
    write_space,
)
from .report import generate_report
from .retrieval import corpus_text
from .run import add_branch, build_clients, merge_branch_scores, run_benchmark
from .toolbench import FAMILIES, run_toolbench

CODIV_FREE_QUOTA_TOKENS = 100_000_000

MODEL_SECTIONS = ("embedding", "reranker_nvidia", "systemone", "generator")


def _cmd_doctor(cfg: dict, _args: argparse.Namespace) -> int:
    print(f"python: {sys.version.split()[0]}")
    print("configured models:")
    for section_name in MODEL_SECTIONS:
        section = cfg["models"][section_name]
        model = section.get("name") or section.get("model")
        env_name = section.get("api_key_env")
        present = "set" if (env_name and api_key(section)) else "not set"
        print(f"  {section_name}: {model} [{env_name}: {present}]")
    data_root = Path(cfg["paths"]["data_dir"])
    print("datasets:")
    for dataset in data_module.ALL_DATASETS:
        processed = (data_root / "processed" / dataset / "queries.jsonl").exists()
        print(f"  {dataset}: {'ready' if processed else 'not prepared'}")
    print("fixture mode: available (no keys, no network)")
    return 0


def _cmd_data(cfg: dict, args: argparse.Namespace) -> int:
    data_root = Path(cfg["paths"]["data_dir"])
    targets = data_module.ALL_DATASETS if args.dataset == "all" else [args.dataset]
    for dataset in targets:
        print(f"preparing {dataset} ...")
        meta = data_module.prepare_dataset(dataset, data_root)
        counts = meta["counts"]
        print(f"  documents: {counts['documents']}, queries: {counts['queries']}")
    return 0


def _cmd_estimate(cfg: dict, args: argparse.Namespace) -> int:
    data_root = Path(cfg["paths"]["data_dir"])
    corpus, queries = data_module.load_processed(args.dataset, data_root)
    retrieval = cfg["retrieval"]
    bm25 = BM25(
        [row["doc_id"] for row in corpus],
        [corpus_text(row) for row in corpus],
        k1=float(retrieval["bm25"]["k1"]),
        b=float(retrieval["bm25"]["b"]),
    )
    text_map = {row["doc_id"]: corpus_text(row) for row in corpus}
    systemone = cfg["models"]["systemone"]
    max_passage_chars = int(systemone.get("max_passage_chars", 4000))
    max_state_chars = int(systemone.get("max_state_chars", 200000))

    total_chars = 0
    for query in queries:
        hits = bm25.search(query["question"], int(retrieval["candidate_depth"]))
        passages = [text_map[doc_id] for doc_id, _ in hits]
        total_chars += len(
            build_state(query["question"], passages, max_passage_chars, max_state_chars)
        )
    estimated_tokens = total_chars // 4
    requests = len(queries) if "J" in args.branches else 0

    print(f"dataset: {args.dataset}")
    print(f"queries: {len(queries)}")
    print(f"candidate depth: {retrieval['candidate_depth']}")
    print(f"estimated System One requests: {requests} (1 per query, batch noul)")
    share = estimated_tokens / CODIV_FREE_QUOTA_TOKENS
    print(
        "estimated System One input tokens: "
        f"~{estimated_tokens:,} ({share:.2%} of the 100M free quota)"
    )
    if estimated_tokens > int(cfg["safety"]["max_systemone_input_tokens"]):
        print("WARNING: estimate exceeds safety.max_systemone_input_tokens in the config")
    print("this estimate makes zero API calls")
    return 0


def _cmd_run(cfg: dict, args: argparse.Namespace) -> int:
    run_kind = "fixture" if args.fixture else "real"
    branches = [part.strip().upper() for part in args.branches.split(",") if part.strip()]
    try:
        run_benchmark(
            cfg,
            args.dataset,
            branches=branches,
            run_kind=run_kind,
            limit=args.limit,
            resume=not args.no_resume,
            output_path=args.output,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _cmd_generate(cfg: dict, args: argparse.Namespace) -> int:
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"error: results file not found: {results_path}", file=sys.stderr)
        return 1
    rows = load_rows(results_path)
    if not rows:
        print("error: results file is empty", file=sys.stderr)
        return 1
    dataset = rows[0].get("dataset", "xquad-en")
    data_root = Path(cfg["paths"]["data_dir"])
    corpus, _ = data_module.load_processed(dataset, data_root)
    corpus_texts = {row["doc_id"]: corpus_text(row) for row in corpus}

    run_kind = "fixture" if args.fixture else "real"
    branch = args.branch.upper()
    output = (
        Path(args.output)
        if args.output
        else results_path.with_name(f"{results_path.stem}-gen-{branch.lower()}-{run_kind}.jsonl")
    )
    try:
        clients = build_clients(cfg, run_kind)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    generator = clients["generator"]
    if run_kind == "real":
        generator = ChatGenerator(
            generator,
            max_context_chars=int(cfg["generation"]["max_context_chars"]),
        )
    stats = replay_generator(
        results_path,
        output,
        branch,
        generator,
        corpus_texts,
        run_kind=run_kind,
        top_k=int(cfg["generation"]["top_k"]),
        limit=args.limit,
        concurrency=args.concurrency,
    )
    print(json.dumps(stats, indent=2))
    return 0 if stats["failed"] == 0 else 1


def _cmd_report(cfg: dict, args: argparse.Namespace) -> int:
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"error: results file not found: {results_path}", file=sys.stderr)
        return 1
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(cfg["paths"]["reports_dir"]) / "generated" / results_path.stem
    )
    report_path = generate_report(
        results_path,
        output_dir,
        generation_path=args.generation or None,
        pricing=cfg.get("pricing"),
    )
    print(f"report: {report_path}")
    print(f"summary: {output_dir / 'summary.json'}")
    return 0


def _cmd_publish(cfg: dict, args: argparse.Namespace) -> int:
    summary_paths = [Path(path) for path in args.summary]
    if not summary_paths:
        summary_paths = find_summaries(cfg["paths"]["reports_dir"])
    if not summary_paths:
        print(
            "error: no summaries found; run the benchmark and report first, "
            "or pass --summary explicitly",
            file=sys.stderr,
        )
        return 1
    try:
        summaries = load_summaries(summary_paths)
        validate_summaries(summaries, allow_fixture=args.allow_fixture)
    except (ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    block = render_readme_block(summaries)
    try:
        inject_into_readme(args.readme, block)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"README updated: {args.readme}")

    published_dir = Path(cfg["paths"]["reports_dir"]) / "published"
    stage_published(published_dir, summaries, summary_paths, repo_url=args.repo_url)
    print(f"published artifacts: {published_dir}")

    if args.space_dir:
        index_path = write_space(args.space_dir, summaries, repo_url=args.repo_url)
        print(f"space page: {index_path}")

    if args.push:
        if not args.repo_id:
            print("error: --push requires --repo-id", file=sys.stderr)
            return 1
        folder = args.space_dir if args.repo_type == "space" and args.space_dir else published_dir
        try:
            url = push_to_huggingface(folder, args.repo_id, repo_type=args.repo_type)
        except RuntimeError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        print(f"pushed: {url}")
    return 0


def _cmd_add_branch(cfg: dict, args: argparse.Namespace) -> int:
    run_kind = "fixture" if args.fixture else "real"
    try:
        add_branch(
            cfg,
            args.results,
            args.branch,
            run_kind=run_kind,
            limit=args.limit,
            concurrency=args.concurrency,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _cmd_audit(cfg: dict, args: argparse.Namespace) -> int:
    try:
        audit = retrieval_audit(cfg, args.dataset, limit=args.limit)
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    audits_dir = Path(cfg["paths"]["reports_dir"]) / "audits"
    json_path, md_path = write_audit(audit, audits_dir)
    print(f"queries: {audit['n_queries']} | corpus: {audit['corpus_size']}")
    print("| depth | BM25 | hybrid |")
    print("|---|---|---|")
    for bm25_row, hybrid_row in zip(audit["bm25"], audit["hybrid"], strict=False):
        label = f"top-{bm25_row['depth']}" if bm25_row["depth"] != "all" else "all"
        print(f"| {label} | {bm25_row['recall'] * 100:.3f}% | {hybrid_row['recall'] * 100:.3f}% |")
    print(f"audit: {json_path}")
    print(f"summary: {md_path}")
    return 0


def _cmd_charts(cfg: dict, args: argparse.Namespace) -> int:
    reports_dir = (
        Path(args.reports_dir)
        if args.reports_dir
        else (Path(cfg["paths"]["reports_dir"]) / "generated")
    )
    summary_paths = sorted(reports_dir.glob("*/summary.json"))
    if not summary_paths:
        print(f"error: no summary.json files under {reports_dir}", file=sys.stderr)
        return 1
    from .publish import load_summaries

    summaries = load_summaries(summary_paths)
    audits_dir = (
        Path(args.audits_dir) if args.audits_dir else (Path(cfg["paths"]["reports_dir"]) / "audits")
    )
    written = render_charts(
        summaries,
        args.output_dir,
        audits=load_audits(audits_dir),
        png=not args.no_png,
        social_dir=args.social_dir,
    )
    if args.toolbench_summary and Path(args.toolbench_summary).exists():
        toolbench_summary = json.loads(Path(args.toolbench_summary).read_text(encoding="utf-8"))
        written.extend(
            render_toolbench_charts(toolbench_summary, args.output_dir, png=not args.no_png)
        )
    if args.answerability_summary and Path(args.answerability_summary).exists():
        answerability_summary = json.loads(
            Path(args.answerability_summary).read_text(encoding="utf-8")
        )
        written.extend(
            render_answerability_charts(answerability_summary, args.output_dir, png=not args.no_png)
        )
    for path in written:
        print(f"chart: {path}")
    return 0


def _cmd_stability(cfg: dict, args: argparse.Namespace) -> int:
    from .stability import stability_audit, write_stability

    run_kind = "fixture" if args.fixture else "real"
    try:
        audit = stability_audit(
            cfg,
            args.results,
            branch=args.branch.upper(),
            limit=args.limit,
            permutations=args.permutations,
            concurrency=args.concurrency,
            run_kind=run_kind,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    output_dir = Path(cfg["paths"]["reports_dir"]) / "audits"
    json_path, md_path = write_stability(audit, output_dir)
    print(
        f"mean Spearman {audit['mean_spearman']:.3f} | "
        f"mean top-5 Jaccard {audit['mean_top5_jaccard']:.3f} | "
        f"gold membership changes {audit['queries_with_gold_membership_change']}"
        f"/{audit['queries']}"
    )
    print(f"audit: {json_path}")
    print(f"summary: {md_path}")
    return 0


def _cmd_verify(cfg: dict, args: argparse.Namespace) -> int:
    from .verify import summarize_verification, verify_generations

    run_kind = "fixture" if args.fixture else "real"
    try:
        path = verify_generations(
            cfg,
            args.results,
            args.generation,
            branch=args.branch.upper(),
            limit=args.limit,
            concurrency=args.concurrency,
            run_kind=run_kind,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    summary = summarize_verification(path)
    output_dir = Path(cfg["paths"]["reports_dir"]) / "audits"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{Path(args.results).stem}-verification-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary:
        print(
            f"n={summary['n']} | mean p(correct)={summary['mean_probability_correct']:.3f} | "
            f"mean p(wrong)={summary['mean_probability_wrong']:.3f} | AUC={summary['auc']:.3f}"
        )
        for row in summary["sweep"]:
            print(
                f"  t={row['threshold']:.1f}: coverage {row['coverage'] * 100:.1f}% | "
                f"precision {row['precision_correct'] * 100:.1f}% | "
                f"wrong answers flagged {row['wrong_answers_flagged']}"
                f"/{row['wrong_answers_total']}"
            )
    print(f"summary: {summary_path}")
    return 0


def _cmd_merge_scores(cfg: dict, args: argparse.Namespace) -> int:
    try:
        merge_branch_scores(cfg, args.results, args.scores, branch=args.branch.upper())
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


def _cmd_toolbench(cfg: dict, args: argparse.Namespace) -> int:
    families = FAMILIES if args.families == "all" else [f.strip() for f in args.families.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(cfg["paths"]["reports_dir"]) / "toolbench"
    )
    try:
        result = run_toolbench(
            cfg,
            families=families,
            per_family=args.per_family,
            models=models,
            output_dir=output_dir,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    summary = result["summary"]
    for model, model_summary in summary.get("models", {}).items():
        print(f"== {model}")
        for family, entry in model_summary.get("families", {}).items():
            print(f"  {family:16s} n={entry['n']:4d}  accuracy={entry['accuracy'] * 100:.1f}%")
        if "cardinality_scaling" in model_summary:
            print(
                "  cardinality scaling:",
                {k: f"{v * 100:.1f}%" for k, v in model_summary["cardinality_scaling"].items()},
            )
    print(f"summary: {result['summary_path']}")
    return 0


def _cmd_answerability(cfg: dict, args: argparse.Namespace) -> int:
    run_kind = "fixture" if args.fixture else "real"
    try:
        path = run_answerability(
            cfg,
            args.results,
            negatives_per_query=args.negatives_per_query,
            limit=args.limit,
            concurrency=args.concurrency,
            run_kind=run_kind,
        )
    except (RuntimeError, FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    summary = summarize_answerability(path, args.generation)
    output_dir = Path(cfg["paths"]["reports_dir"]) / "audits"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{Path(args.results).stem}-answerability-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if summary:
        print(
            f"n={summary['n']} accuracy={summary['accuracy'] * 100:.1f}% "
            f"brier={summary['brier']:.3f} ece={summary['ece_10bin']:.3f}"
        )
        print(
            f"mean p matched={summary['mean_probability_matched']:.3f} "
            f"mismatched={summary['mean_probability_mismatched']:.3f}"
        )
        for row in summary.get("generation_gating", []):
            print(
                f"  t={row['threshold']:.1f}: coverage {row['coverage'] * 100:.1f}% "
                f"F1 {row['mean_f1'] * 100:.2f}% success {row['success_rate'] * 100:.2f}% "
                f"({row['success_per_1000_queries']:.0f} per 1k)"
            )
    print(f"summary: {summary_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jev-rag",
        description=(
            "Free English RAG benchmark for TypeSafe Jev 1.13 (no-reranker and NVIDIA baselines)."
        ),
    )
    parser.add_argument(
        "--config",
        default=None,
        help="path to a YAML config (default: configs/default.yaml if present)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("doctor", help="show configuration and key availability")

    data_parser = subparsers.add_parser("data", help="download and normalize datasets")
    data_parser.add_argument("action", choices=["prepare"])
    data_parser.add_argument("--dataset", default="all", choices=[*data_module.ALL_DATASETS, "all"])

    estimate_parser = subparsers.add_parser(
        "estimate", help="project System One token usage with zero API calls"
    )
    estimate_parser.add_argument("--dataset", required=True, choices=data_module.ALL_DATASETS)
    estimate_parser.add_argument("--branches", default="J")

    run_parser = subparsers.add_parser("run", help="run retrieval and reranking branches")
    run_parser.add_argument("--dataset", required=True, choices=data_module.ALL_DATASETS)
    run_parser.add_argument("--branches", default="A,J,N", help="comma-separated: A,J,N")
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--fixture", action="store_true", help="offline deterministic clients")
    run_parser.add_argument("--no-resume", action="store_true")
    run_parser.add_argument("--output", default=None)

    generate_parser = subparsers.add_parser(
        "generate", help="replay an answer generator over frozen branch contexts"
    )
    generate_parser.add_argument("--results", required=True)
    generate_parser.add_argument("--branch", default="J")
    generate_parser.add_argument("--limit", type=int, default=None)
    generate_parser.add_argument("--concurrency", type=int, default=4)
    generate_parser.add_argument("--fixture", action="store_true")
    generate_parser.add_argument("--output", default=None)

    add_branch_parser = subparsers.add_parser(
        "add-branch", help="add one branch (e.g. T: real Jev 1.13) to existing results"
    )
    add_branch_parser.add_argument("--results", required=True)
    add_branch_parser.add_argument("--branch", required=True, help="A, J, N or T")
    add_branch_parser.add_argument("--limit", type=int, default=None)
    add_branch_parser.add_argument("--concurrency", type=int, default=4)
    add_branch_parser.add_argument("--fixture", action="store_true")

    report_parser = subparsers.add_parser("report", help="build a markdown report and summary")
    report_parser.add_argument("--results", required=True)
    report_parser.add_argument(
        "--generation",
        action="append",
        default=[],
        help="generation JSONL to merge; repeat for multiple branches",
    )
    report_parser.add_argument("--output-dir", default=None)

    audit_parser = subparsers.add_parser(
        "audit", help="candidate-depth audit: BM25 vs hybrid gold-passage recall"
    )
    audit_parser.add_argument("--dataset", required=True, choices=data_module.ALL_DATASETS)
    audit_parser.add_argument("--limit", type=int, default=None)

    charts_parser = subparsers.add_parser(
        "charts", help="render benchmark charts (SVG, plus PNG via headless Chrome)"
    )
    charts_parser.add_argument("--reports-dir", default=None)
    charts_parser.add_argument("--audits-dir", default=None)
    charts_parser.add_argument("--output-dir", default="assets/benchmark")
    charts_parser.add_argument("--social-dir", default="assets/social")
    charts_parser.add_argument(
        "--toolbench-summary",
        default=None,
        help="path to reports/toolbench/toolbench-summary.json",
    )
    charts_parser.add_argument(
        "--answerability-summary",
        default=None,
        help="path to reports/audits/*-answerability-summary.json",
    )
    charts_parser.add_argument("--no-png", action="store_true")

    stability_parser = subparsers.add_parser(
        "stability", help="candidate-order stability audit for the Jev branch"
    )
    stability_parser.add_argument("--results", required=True)
    stability_parser.add_argument("--branch", default="T")
    stability_parser.add_argument("--limit", type=int, default=100)
    stability_parser.add_argument("--permutations", type=int, default=3)
    stability_parser.add_argument("--concurrency", type=int, default=4)
    stability_parser.add_argument("--fixture", action="store_true")

    verify_parser = subparsers.add_parser(
        "verify", help="verify generated answers against their cited passages with Jev"
    )
    verify_parser.add_argument("--results", required=True)
    verify_parser.add_argument("--generation", required=True)
    verify_parser.add_argument("--branch", default="T")
    verify_parser.add_argument("--limit", type=int, default=None)
    verify_parser.add_argument("--concurrency", type=int, default=4)
    verify_parser.add_argument("--fixture", action="store_true")

    answerability_parser = subparsers.add_parser(
        "answerability", help="answerability/abstention cases and generation-gating simulation"
    )
    answerability_parser.add_argument("--results", required=True)
    answerability_parser.add_argument("--generation", default=None)
    answerability_parser.add_argument("--negatives-per-query", type=int, default=1)
    answerability_parser.add_argument("--limit", type=int, default=None)
    answerability_parser.add_argument("--concurrency", type=int, default=4)
    answerability_parser.add_argument("--fixture", action="store_true")

    toolbench_parser = subparsers.add_parser(
        "toolbench", help="tool-calling governance benchmark (4 families)"
    )
    toolbench_parser.add_argument("--families", default="all")
    toolbench_parser.add_argument("--per-family", type=int, default=40)
    toolbench_parser.add_argument("--models", default="jev,laya")
    toolbench_parser.add_argument("--output-dir", default=None)

    merge_parser = subparsers.add_parser(
        "merge-scores", help="merge externally computed pointwise scores as a branch"
    )
    merge_parser.add_argument("--results", required=True)
    merge_parser.add_argument("--scores", required=True)
    merge_parser.add_argument("--branch", default="L")

    publish_parser = subparsers.add_parser(
        "publish", help="update the README results table and stage shareable artifacts"
    )
    publish_parser.add_argument("--summary", action="append", default=[])
    publish_parser.add_argument("--readme", default="README.md")
    publish_parser.add_argument("--space-dir", default=None)
    publish_parser.add_argument("--repo-url", default=None)
    publish_parser.add_argument("--allow-fixture", action="store_true")
    publish_parser.add_argument("--push", action="store_true")
    publish_parser.add_argument("--repo-id", default=None)
    publish_parser.add_argument(
        "--repo-type", default="dataset", choices=["dataset", "space", "model"]
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    load_dotenv()
    try:
        cfg = load_config(args.config)
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    handlers = {
        "doctor": _cmd_doctor,
        "data": _cmd_data,
        "estimate": _cmd_estimate,
        "run": _cmd_run,
        "generate": _cmd_generate,
        "add-branch": _cmd_add_branch,
        "report": _cmd_report,
        "audit": _cmd_audit,
        "charts": _cmd_charts,
        "stability": _cmd_stability,
        "verify": _cmd_verify,
        "merge-scores": _cmd_merge_scores,
        "toolbench": _cmd_toolbench,
        "answerability": _cmd_answerability,
        "publish": _cmd_publish,
    }
    return handlers[args.command](cfg, args)


if __name__ == "__main__":
    sys.exit(main())

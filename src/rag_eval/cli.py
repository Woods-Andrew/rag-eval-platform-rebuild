"""Command line entry point: index a PDF, search it, or run the benchmark."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .chunking import Chunker, FixedSizeChunker, StructureAwareChunker
from .evaluation import EvaluationReport, compare
from .evaluation.benchmark import BenchmarkError, load_benchmark
from .experiments import SPEC_HELP, format_sweep, parse_variant, run_sweep, write_sweep
from .generation import DEFAULT_GENERATION_MODEL, GenerationError
from .pipeline import build_corpus
from .retrieval import BM25Retriever, Corpus, DenseRetriever, HybridRetriever, Retriever

__all__ = ["main"]

CHUNKERS = ("fixed", "structure")
RETRIEVERS = ("bm25", "dense", "hybrid", "reranked")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code rather than calling ``sys.exit``."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        return args.handler(args)
    except (FileNotFoundError, ValueError, BenchmarkError, GenerationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-eval", description="Hybrid retrieval and evaluation for technical documents."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index = subparsers.add_parser("index", help="chunk a PDF and report corpus statistics")
    _add_corpus_arguments(index)
    index.set_defaults(handler=_command_index)

    search = subparsers.add_parser("search", help="run one query against a PDF")
    _add_corpus_arguments(search)
    search.add_argument("query", help="the search query")
    search.add_argument("-k", "--top-k", type=int, default=5, help="results to show (default 5)")
    search.add_argument(
        "-r", "--retriever", choices=RETRIEVERS, default="bm25", help="strategy (default bm25)"
    )
    search.set_defaults(handler=_command_search)

    evaluate = subparsers.add_parser("evaluate", help="run a labelled benchmark")
    _add_corpus_arguments(evaluate)
    evaluate.add_argument("benchmark", type=Path, help="path to the benchmark JSON file")
    evaluate.add_argument("-k", "--top-k", type=int, default=5, help="cutoff K (default 5)")
    evaluate.add_argument(
        "-r",
        "--retriever",
        choices=RETRIEVERS,
        action="append",
        help="strategy to evaluate; repeatable (default: all three)",
    )
    evaluate.set_defaults(handler=_command_evaluate)

    sweep = subparsers.add_parser(
        "sweep", help="compare chunking configurations on the same document"
    )
    sweep.add_argument("pdf", type=Path, help="path to the source PDF")
    sweep.add_argument(
        "-v", "--variant", action="append", required=True, help=SPEC_HELP
    )
    sweep.add_argument("-k", "--top-k", type=int, default=5, help="cutoff K (default 5)")
    sweep.add_argument(
        "-r",
        "--retriever",
        choices=RETRIEVERS,
        action="append",
        help="strategy to evaluate; repeatable (default: all four)",
    )
    sweep.add_argument("-o", "--out", type=Path, help="write the full results as JSON")
    sweep.set_defaults(handler=_command_sweep)

    ask = subparsers.add_parser("ask", help="answer a question with cited evidence")
    _add_corpus_arguments(ask)
    ask.add_argument("query", help="the question to answer")
    ask.add_argument(
        "-k", "--top-k", type=int, default=5, help="passages to ground on (default 5)"
    )
    ask.add_argument(
        "-r", "--retriever", choices=RETRIEVERS, default="hybrid", help="strategy (default hybrid)"
    )
    ask.add_argument("--model", default=DEFAULT_GENERATION_MODEL, help="generation model")
    ask.add_argument(
        "--show-evidence", action="store_true", help="print the passages the answer was given"
    )
    ask.set_defaults(handler=_command_ask)

    return parser


def _add_corpus_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("pdf", type=Path, help="path to the source PDF")
    parser.add_argument(
        "-c", "--chunker", choices=CHUNKERS, default="fixed", help="strategy (default fixed)"
    )


def _make_chunker(name: str) -> Chunker:
    return FixedSizeChunker() if name == "fixed" else StructureAwareChunker()


def _make_retriever(name: str, corpus: Corpus) -> Retriever:
    """Build a retriever, importing the ML models only when one is asked for."""
    if name == "bm25":
        return BM25Retriever(corpus)

    from .retrieval import SentenceTransformerEncoder

    if name == "dense":
        return DenseRetriever(corpus, SentenceTransformerEncoder())

    hybrid = HybridRetriever(
        [BM25Retriever(corpus), DenseRetriever(corpus, SentenceTransformerEncoder())]
    )
    if name == "hybrid":
        return hybrid

    from .reranking import CrossEncoderReranker, RerankingRetriever

    return RerankingRetriever(hybrid, CrossEncoderReranker())


def _command_index(args: argparse.Namespace) -> int:
    corpus = build_corpus(args.pdf, _make_chunker(args.chunker))
    words = [len(chunk.text.split()) for chunk in corpus]
    pages = {chunk.page_number for chunk in corpus}

    print(f"{args.pdf.name}: {len(corpus)} chunks across {len(pages)} page(s)")
    print(f"chunker: {args.chunker}")
    print(f"words per chunk: min {min(words)}, mean {sum(words) / len(words):.1f}, max {max(words)}")
    print()
    print("first chunk:")
    first = corpus.chunks[0]
    print(f"  id      {first.chunk_id}")
    print(f"  cite    {first.citation}")
    print(f"  text    {_preview(first.text)}")
    return 0


def _command_search(args: argparse.Namespace) -> int:
    corpus = build_corpus(args.pdf, _make_chunker(args.chunker))
    results = _make_retriever(args.retriever, corpus).retrieve(args.query, top_k=args.top_k)

    if not results:
        print(f"no results for {args.query!r}")
        return 0

    print(f"{len(results)} result(s) for {args.query!r} via {args.retriever}:\n")
    for result in results:
        print(f"  {result.rank}. [{result.score:.4f}] {result.citation}")
        print(f"     {result.chunk_id}")
        print(f"     {_preview(result.chunk.text)}\n")
    return 0


def _command_evaluate(args: argparse.Namespace) -> int:
    corpus = build_corpus(args.pdf, _make_chunker(args.chunker))
    benchmark = load_benchmark(args.benchmark, corpus)

    names = args.retriever or list(RETRIEVERS)
    retrievers = {name: _make_retriever(name, corpus) for name in names}
    reports = compare(retrievers, list(benchmark.questions), k=args.top_k)

    print(f"{benchmark.document or args.pdf.name}: {len(benchmark)} questions, "
          f"{benchmark.label_count} labels, chunker={args.chunker}, K={args.top_k}\n")
    _print_report_table(reports)

    print("\nper category (recall / nDCG):")
    for report in reports:
        summary = ", ".join(
            f"{category} {recall:.3f}/{ndcg:.3f}"
            for category, (recall, ndcg) in report.by_category().items()
        )
        print(f"  {report.retriever_name:<10} {summary}")

    for report in reports:
        if report.questions_with_no_hit:
            missed = ", ".join(report.questions_with_no_hit)
            print(f"\n{report.retriever_name}: nothing relevant in the top {args.top_k} for {missed}")
    return 0


def _command_sweep(args: argparse.Namespace) -> int:
    variants = [parse_variant(spec) for spec in args.variant]
    names = args.retriever or list(RETRIEVERS)

    def retriever_factory(corpus: Corpus) -> dict[str, Retriever]:
        return {name: _make_retriever(name, corpus) for name in names}

    result = run_sweep(args.pdf, variants, retriever_factory=retriever_factory, k=args.top_k)
    print(format_sweep(result))

    if args.out:
        print(f"\nwrote {write_sweep(result, args.out)}")
    return 0


def _command_ask(args: argparse.Namespace) -> int:
    """Retrieve, then answer from what was retrieved — never the other way round."""
    from .generation import AnswerGenerator, ClaudeLanguageModel

    corpus = build_corpus(args.pdf, _make_chunker(args.chunker))
    results = _make_retriever(args.retriever, corpus).retrieve(args.query, top_k=args.top_k)

    generator = AnswerGenerator(ClaudeLanguageModel(args.model), max_evidence=args.top_k)
    answer = generator.answer(args.query, results)

    if args.show_evidence:
        print(f"evidence ({len(answer.evidence)} passage(s) via {args.retriever}):\n")
        for item in answer.evidence:
            print(f"  [{item.marker}] {item.citation}")
            print(f"      {_preview(item.text)}\n")

    print(answer.format())

    if answer.unresolved_citations:
        markers = ", ".join(str(marker) for marker in answer.unresolved_citations)
        print(f"\nwarning: the answer cited [{markers}], which was never supplied as evidence")
        return 1
    return 0


def _print_report_table(reports: Sequence[EvaluationReport]) -> None:
    header = f"{'retriever':<12}{'Recall@K':>10}{'nDCG@K':>10}{'misses':>9}"
    print(header)
    print("-" * len(header))
    for report in reports:
        print(
            f"{report.retriever_name:<12}"
            f"{report.mean_recall:>10.4f}"
            f"{report.mean_ndcg:>10.4f}"
            f"{len(report.questions_with_no_hit):>9}"
        )


def _preview(text: str, limit: int = 100) -> str:
    flattened = " ".join(text.split())
    return flattened if len(flattened) <= limit else f"{flattened[:limit]}…"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

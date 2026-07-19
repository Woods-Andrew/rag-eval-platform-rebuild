# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

A hybrid retrieval and evaluation platform for technical documents: BM25 + dense embeddings
fused with reciprocal rank fusion, cross-encoder reranking, and Recall@K / nDCG@K benchmarking
against hand-labeled relevance judgments. It is a portfolio project whose point is to
demonstrate the *machinery* of RAG — retrieval, fusion, reranking, evaluation — not to ship
another PDF chatbot.

This means: the algorithms are the deliverable. Favor code that reads well in a technical
interview over code that is clever.

## Environment

- **Python 3.11 only.** Pinned in `.python-version` and `requires-python = ">=3.11,<3.12"`.
- Virtualenv lives at `.venv/` in the project root, created from `/opt/homebrew/bin/python3.11`.
- Run everything through the venv: `.venv/bin/python`, `.venv/bin/python -m pytest`.
- Do not create a second environment or use the system/conda Python (3.10 and 3.13 are both
  present on this machine — neither is correct here).

```bash
source .venv/bin/activate
python -m pytest -v
```

## Layout

All Python lives under `src/rag_eval/` (src-layout). Do not add top-level Python packages.

| Path | Contents |
| --- | --- |
| `src/rag_eval/ingestion/` | PDF → pages with metadata |
| `src/rag_eval/chunking/` | pages → chunks (fixed, structure-aware) |
| `src/rag_eval/retrieval/` | BM25, dense, hybrid RRF |
| `src/rag_eval/reranking/` | cross-encoder |
| `src/rag_eval/evaluation/` | metrics, evaluator |
| `src/rag_eval/generation/` | grounded answers, citations |
| `tests/` | offline, deterministic unit tests |
| `data/documents/` | source PDFs — **gitignored** |
| `data/evaluation/` | benchmark questions + labels — **version controlled** |
| `experiments/results/` | generated output — gitignored |

Create directories and modules when a milestone actually needs them. Empty scaffolding is
noise; do not pre-create the full tree.

## Architectural rules

These are non-negotiable and exist to keep the project honest:

1. **Retrieval and evaluation must not import Streamlit.** The UI is a consumer, never a
   dependency.
2. **Evaluation must not import generation.** Retrieval quality is measured on its own terms.
3. **Generation consumes retrieved evidence.** It never drives or re-runs retrieval.
4. **Provenance survives every stage** — source, page number, chunk ID, metadata. If a
   transformation drops provenance, it is wrong.
5. **Never add BM25 scores to cosine similarities.** They are not on a comparable scale.
   Combination happens through rank fusion (RRF), always.
6. **Encode the corpus once.** Never re-embed documents per query; only the query is encoded
   at retrieval time.
7. **Implement RRF, Recall@K, and nDCG@K explicitly.** These are the parts a reader should be
   able to verify by eye. Do not delegate them to a library.
8. **ML models are dependency-injected.** Encoders and rerankers are passed in so tests can
   substitute fakes.

## Dependencies

Allowed, added *when the milestone needs them*: `pymupdf`, `pytest`, `rank-bm25`, `numpy`,
`sentence-transformers`, `torch`, `streamlit`, and `scikit-learn` only if genuinely necessary.

**Do not introduce** LangChain, LlamaIndex, Pinecone, Chroma, FAISS, any external vector
database, or orchestration frameworks. The project exists to show the algorithms, not to hide
them behind a framework. If one of these seems necessary, ask first.

Do not front-load future dependencies into the environment. Install at the milestone that
uses them, and update `requirements.txt` with pinned versions in the same commit.

## Code style

- Strong type hints throughout; `from __future__ import annotations` at the top of modules.
- `@dataclass` (usually frozen) for data models. Protocols for interfaces, but only where they
  earn their place.
- Concise docstrings on public APIs — what it does and what the assumptions are, not a restatement
  of the signature.
- Small functions, small modules, no hidden global state, no factories or inheritance
  hierarchies without a concrete reason.
- Validate inputs at public boundaries and raise clear errors (empty query, `top_k <= 0`,
  missing file, unreadable PDF).

## Testing

- `python -m pytest -v` must pass before any commit. Every milestone lands green.
- Tests must be **deterministic** and run **fully offline**. No network, no Hugging Face
  downloads, no API calls in unit tests. Use fake encoders with fixed vectors.
- Test behavior, not line count: validation, edge cases, deterministic chunk IDs, provenance,
  ranking order, deduplication, metric mathematics, evaluator aggregation.
- Integration tests that need real models are kept separate and clearly marked.

## Git

- Branch: `main`. Conventional Commit messages (`feat:`, `test:`, `docs:`, `chore:`, `data:`,
  `perf:`).
- One logical unit of work per commit. Run the tests before committing.
- **Never commit without explicit approval.** Show the diff and the proposed message first.
- **Never push, and never create a GitHub repository, without explicit approval.**
- Work each milestone on its own branch, commit to it as the work progresses, and land it on
  `main` via a pull request merged with `--no-ff`.
- Keep `.venv/`, caches, model files, PDFs, and secrets out of git.

## Benchmark integrity

This is the rule that matters most.

- **Never invent benchmark numbers.** Not in the README, not in docs, not as placeholders, not
  as illustrative examples. If a number has not been measured, the section says so.
- **Never fabricate relevance labels.** Every label must correspond to a chunk actually
  produced by the pipeline and be verifiable against the document.
- **Report what was measured**, including when a simpler method beats a more sophisticated one.
  Do not tune the setup to make hybrid or reranking "win."
- The resume claim attached to this project will cite real measured results, so a fabricated
  number is a real-world liability, not a cosmetic one.

## Related directory — do not touch

`/Users/Andrew/Projects/rag-eval-platform` is the older version of this project, kept as a
read-only reference. Do not modify, delete, reset, or copy source code from it.

## Current status

Ingestion, both chunking strategies, all three retrievers (BM25, dense, hybrid RRF), the
Recall@K / nDCG@K evaluation runner, and the `index` / `search` / `evaluate` CLI are
implemented and tested offline.

**The benchmark dataset does not exist yet.** `data/documents/` is empty, so no labels have
been written and no results have been measured. `data/evaluation/SCHEMA.md` documents the file
format and the labelling procedure. Do not create a benchmark file with invented labels to
"unblock" anything — `load_benchmark` validates every label against the corpus and will reject
them, which is the intended behaviour.

Next milestone is cross-encoder reranking. See the roadmap in `README.md`.

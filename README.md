# RAG Evaluation Platform

A hybrid retrieval and evaluation platform for technical documents.

Most "chat with your PDF" projects stop at *does it produce an answer?* This one asks the
harder question: **does the retriever actually find the right passage, and can you prove it?**
Retrieval quality is the ceiling on answer quality — a generator cannot ground an answer in
evidence it was never given. So the core of this project is a measured comparison of retrieval
strategies over a real technical paper, using Recall@K and nDCG@K against manually validated
relevance labels.

> **Status: early build.** This README documents what exists today and what is planned.
> Sections marked _planned_ are not implemented yet, and no benchmark numbers appear
> anywhere in this repository until they have actually been measured.

---

## Table of contents

- [Problem](#problem)
- [Architecture](#architecture)
- [Chunking strategies](#chunking-strategies)
- [Retrieval methods](#retrieval-methods)
- [Evaluation methodology](#evaluation-methodology)
- [Benchmark dataset](#benchmark-dataset)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Tests](#tests)
- [Design principles](#design-principles)
- [Project layout](#project-layout)
- [Roadmap](#roadmap)
- [Limitations](#limitations)
- [License](#license)

---

## Problem

Technical documents — research papers, specifications, manuals — are hostile to naive
retrieval. They mix dense prose with tables and numeric results, they use acronyms that only
appear expanded once, and the terminology a reader searches with is often not the terminology
the document uses.

That creates a concrete tension:

- **Lexical search (BM25)** nails exact terminology, rare tokens, acronyms, and identifiers,
  but fails on paraphrase. Ask "how did they handle missing data?" about a paper that says
  "imputation strategy" and BM25 returns nothing useful.
- **Dense embedding search** handles paraphrase and conceptual similarity, but blurs precise
  tokens. It will happily rank a passage about a *different* metric near the top because the
  surrounding language is similar.

Neither is universally better. The interesting engineering question is how to combine them,
and — more importantly — how to *measure* whether the combination actually helped. This
platform is built to answer that with numbers rather than vibes.

## Architecture

Two pipelines, deliberately decoupled. Evaluation never imports generation, and neither
depends on the UI.

**Retrieval / generation pipeline**

```
                 PDF
                  │
          page extraction            (source, page number, metadata preserved)
                  │
             chunking
          ┌───────┴────────┐
     fixed-size      structure-aware
          └───────┬────────┘
                  │
        parallel retrieval
          ┌───────┴────────┐
       BM25              dense
     (lexical)        (embeddings)
          └───────┬────────┘
                  │
       reciprocal rank fusion        (fuse ranks, never raw scores)
                  │
       cross-encoder reranking       (query × passage, jointly scored)
                  │
         best evidence chunks
                  │
      grounded answer generation
                  │
        page-level citations
```

**Evaluation pipeline**

```
   benchmark questions
            │
   known relevant chunk IDs
            │
        run retriever
            │
      ranked results
            │
    Recall@K  +  nDCG@K
            │
 compare retrieval configurations
```

The two pipelines share the retriever interface and nothing else. That separation is what
makes it possible to swap a chunking strategy or a reranker and get a clean, comparable
measurement out the other side.

## Chunking strategies

Chunking decides what the retriever is even allowed to return, so it is treated as a variable
to be measured rather than a detail to be settled by taste. Both strategies implement the same
`Chunker` protocol, which is what makes them swappable in the benchmark.

| Strategy | Boundary | Tradeoff |
| --- | --- | --- |
| Fixed-size | Every N words, with overlap | Predictable sizes; cuts wherever the count runs out, routinely separating a definition from its term |
| Structure-aware | Heading → paragraph → sentence → word window | Respects the author's own boundaries; produces uneven sizes, and depends on headings being detectable |

Sizes are counted in **words**, not characters — word counts track token counts far more
closely, BM25 tokenizes on word boundaries anyway, and a boundary that lands on a word can
never bisect an acronym or identifier.

Neither strategy lets a chunk span a page boundary, so every chunk cites exactly one page.

Structure-aware chunking degrades one step at a time rather than failing: a heading always
starts a new chunk, paragraphs accumulate until the next would overflow, an oversized paragraph
is split between sentences, and only a single sentence longer than the limit falls back to a
blind word window. Heading detection is intentionally conservative — known section names,
numbered headings, and all-caps lines, but *not* generic title case, which fires on figure
captions and author lines. A missed heading degrades to paragraph chunking; a false one splits
a section in half.

Which strategy actually retrieves better is an open question here, and is measured rather than
assumed — see [Results](#results).

## Retrieval methods

Each is implemented as an independent, individually benchmarkable component.

| Method | Mechanism | Strength |
| --- | --- | --- |
| BM25 | Sparse lexical scoring over a tokenized corpus | Exact terms, acronyms, rare tokens |
| Dense | Bi-encoder embeddings, cosine similarity via normalized dot product | Paraphrase, conceptual similarity |
| Hybrid RRF | Reciprocal rank fusion over both ranked lists | Recovers documents either method alone misses |
| Hybrid + reranker | Cross-encoder rescoring of the fused top-K | Precision at the very top of the ranking |

**Why rank fusion and not score blending.** BM25 scores are unbounded, corpus-dependent, and
not comparable across queries; cosine similarities live in `[-1, 1]`. Adding them (or
min-max normalizing them per query) makes the weighting an artifact of score distributions
rather than of retrieval quality. Reciprocal rank fusion sidesteps this entirely by using only
*rank position*:

```
RRF(d) = Σ  1 / (k + rank_i(d))          k ≈ 60, ranks 1-indexed
         i
```

A document ranked highly by either retriever gets a strong contribution; a document ranked
highly by both wins. The constant `k` damps the influence of the top rank so that a single
retriever's #1 result cannot unilaterally dominate the fused ordering.

**Why a cross-encoder on top.** The bi-encoder embeds query and passage *separately*, so it can
never model term-level interaction between them. A cross-encoder reads the pair jointly and is
substantially more accurate — and far too slow to run over a whole corpus. Hence the standard
shape: cheap retrieval to get a candidate set, expensive reranking to order it.

RRF, Recall@K, and nDCG@K are implemented explicitly in this repository rather than imported.
They are the parts a reader should be able to inspect and verify.

## Evaluation methodology

**Recall@K** — of the chunks known to be relevant to a query, what fraction appear in the top K?
This measures whether the evidence reached the generator at all. If a passage is not in the
retrieved set, no amount of prompt engineering recovers it.

**nDCG@K** — normalized discounted cumulative gain. Recall treats every position in the top K
as equivalent; nDCG does not. It discounts each hit logarithmically by rank and normalizes
against the ideal ordering, producing a score in `[0, 1]`:

```
DCG@K  = Σ  rel_i / log2(i + 1)
        i=1..K

nDCG@K = DCG@K / IDCG@K
```

Both metrics matter and they answer different questions. Recall@K asks *did we find it*;
nDCG@K asks *did we rank it well*. A reranker that improves nDCG while leaving Recall flat is
doing exactly what a reranker is supposed to do — reordering an already-adequate candidate set.

Relevance is binary in the initial implementation, which is the honest choice for a
hand-labeled benchmark of this size; graded relevance is left as future work.

## Benchmark dataset

_Planned._ The benchmark document will be a technical research paper,
*"Agentic AI for Disease-Aware Adaptive Multi-Omics Embedding"* (~7 pages).

The benchmark will contain roughly 10–15 hand-written questions with manually verified
relevant chunk IDs, spanning the query types that stress retrieval differently:

- exact / lexical terminology
- semantic paraphrases
- acronyms
- methodology questions
- numerical results
- scientific findings

Labels are validated against actual chunks produced by the ingestion pipeline. No relevance
label in this repository is generated or guessed.

Source PDFs are gitignored; the benchmark questions and labels are version controlled, since
they are the reproducible part.

## Results

_Not yet measured._

This section will hold real measured comparisons once the benchmark exists — retriever
comparisons, chunking-strategy comparisons, and the latency tradeoff. Nothing is reported here
until it has actually been run, and results are reported as measured whether or not they favor
the more sophisticated method.

## Installation

Requires **Python 3.11** (pinned; the project does not target 3.12+).

```bash
git clone <repository-url>
cd rag-eval-platform-rebuild

python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python --version                   # expect Python 3.11.x

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` if you need to set optional cache paths or, later, a generation
API key. Nothing in the current milestone requires it.

## Usage

_Planned._ The benchmark CLI and the Streamlit interface arrive at their respective
milestones. This section will document the exact commands once they exist.

## Tests

```bash
python -m pytest -v
```

Unit tests are deterministic and run fully offline. No test downloads a model, hits the
network, or calls an external API — embedding and reranking models are injected as
dependencies so they can be replaced with fakes. Any test that genuinely needs a downloaded
model will be marked as an integration test and kept separate.

## Design principles

1. Components are modular and independently testable.
2. Retrieval and evaluation never depend on Streamlit.
3. Evaluation never depends on answer generation.
4. Generation consumes retrieved evidence; it does not control retrieval.
5. Provenance — source, page, chunk ID, metadata — survives every stage of the pipeline.
6. Corpus embeddings are computed once, not per query.
7. BM25 scores are never added directly to cosine similarities; fusion happens over ranks.
8. Mathematically important algorithms are written out explicitly and kept readable.
9. Strong type hints, concise docstrings on public APIs, no premature abstraction.
10. Every milestone leaves the full test suite passing.

## Project layout

```
src/rag_eval/
├── ingestion/     PDF → pages with metadata
├── chunking/      pages → chunks (fixed-size and structure-aware)
├── retrieval/     BM25, dense, hybrid RRF
├── reranking/     cross-encoder
├── evaluation/    Recall@K, nDCG@K, benchmark runner
├── generation/    grounded answers, citations
└── config.py

tests/          unit tests (offline, deterministic)
data/           documents (gitignored) + benchmark labels (version controlled)
experiments/    benchmark scripts and generated results
scripts/        CLI entry points
app/            Streamlit interface
docs/           architecture and methodology notes
```

Directories appear as the milestones that need them land, rather than as empty scaffolding.

## Roadmap

- [x] Repository, packaging, and Python 3.11 environment
- [x] PDF ingestion with page metadata
- [x] Fixed-size chunking with configurable overlap
- [x] Structure-aware chunking along headings and sentences
- [ ] BM25 lexical retrieval
- [ ] Dense embedding retrieval
- [ ] Reciprocal rank fusion + hybrid retriever
- [ ] Recall@K and nDCG@K + evaluation runner
- [ ] Hand-labeled benchmark on a real technical paper
- [ ] Benchmark CLI and baseline results
- [ ] Cross-encoder reranking
- [ ] Chunking strategy experiments
- [ ] Grounded generation with page-level citations
- [ ] Streamlit interface and evaluation dashboard

## Limitations

Known and accepted, to be revisited as the project matures:

- The benchmark is small and single-document, so results characterize *this* paper rather than
  technical documents in general.
- Relevance labels are binary and hand-authored by one person.
- Text-only ingestion; tables and figures are not specially handled.
- Chunks never span a page boundary, which keeps every citation unambiguous but splits any
  passage that straddles a page break.
- No approximate-nearest-neighbor index — exact search is correct and fast enough at this
  corpus size, and adding a vector database would obscure the mechanics this project exists to
  demonstrate.

## License

MIT.

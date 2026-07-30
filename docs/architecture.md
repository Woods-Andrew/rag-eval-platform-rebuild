# Architecture

How the pieces fit, and which dependencies are allowed to exist. The short version: data
flows one way, provenance never drops, and the interesting algorithms are written out rather
than imported.

- [Module map](#module-map)
- [The path a query takes](#the-path-a-query-takes)
- [Boundaries](#boundaries)
- [Provenance](#provenance)
- [Dependency injection](#dependency-injection)
- [Where cost lives](#where-cost-lives)

## Module map

```
src/rag_eval/
├── ingestion/     PDF → pages, text cleaned, metadata preserved
├── chunking/      pages → chunks (fixed-size, structure-aware)
├── retrieval/     corpus, BM25, dense, RRF, hybrid, embedding cache
├── reranking/     cross-encoder rescoring of a candidate set
├── evaluation/    Recall@K, nDCG@K, benchmark loading, runner
├── experiments/   chunking sweeps, corpus statistics, reporting
├── generation/    grounded answers, citations, refusal
├── ui/            Streamlit app + a service layer that imports no UI framework
├── factory.py     builds the four strategies over one corpus, each at most once
├── pipeline.py    the one PDF → corpus path everything shares
└── cli.py         index / search / evaluate / sweep / ask
```

`pipeline.build_corpus` exists so the CLI, the benchmark, the sweep harness, and the UI all
index a document *identically*. If they did not, a chunk ID would mean something different
depending on who asked, and a relevance label would stop being portable between them.

## The path a query takes

```
PDF
 │  ingestion       page text + source + page number + document metadata
 ▼
pages
 │  chunking        fixed-size windows, or heading/paragraph/sentence boundaries
 ▼
chunks             each with a deterministic ID: provenance + SHA-256 of its own text
 │
 ▼
Corpus             unique IDs enforced at construction
 │
 ├──────────────┬─────────────────┐
 │ BM25         │ dense           │   ← corpus embedded once, cached across runs
 │ (lexical)    │ (bi-encoder)    │
 └──────┬───────┴────────┬────────┘
        │                │
        ▼                ▼
        reciprocal rank fusion       fuses ranks, never raw scores
                 │
                 ▼
        cross-encoder reranking      query × passage, scored jointly
                 │
                 ▼
        ranked RetrievalResults      chunk + score + 1-indexed rank
                 │
        ┌────────┴─────────┐
        ▼                  ▼
   evaluation          generation
   Recall@K            numbered evidence → cited answer
   nDCG@K              or a refusal
```

Evaluation and generation are siblings, not a sequence. They consume the same ranked results
and never see each other.

## Boundaries

Four rules, each enforced by a test that launches a subprocess and inspects `sys.modules`
rather than by convention:

| Rule | Why |
| --- | --- |
| Retrieval and evaluation never import Streamlit | The UI is a consumer. If retrieval depended on it, the pipeline could not run headless — and a benchmark you cannot run in a script is not a benchmark. |
| Evaluation never imports generation | Retrieval quality is measured on its own terms. If evaluation could reach generation, a benchmark could quietly start scoring answers instead of rankings. |
| Generation consumes retrieval, never drives it | If the generator could widen its own evidence, Recall@K would stop describing what the answer was actually built from. |
| Importing a package loads no ML libraries | `torch` costs seconds to import. Model libraries are imported lazily inside the adapters that need them, so the offline test suite never pays for — or accidentally reaches — a model. |

The last one has a second effect worth naming: `python -m rag_eval search paper.pdf "q"` with
BM25 never constructs an encoder at all.

## Provenance

Provenance is the property everything else rests on, so it is carried explicitly rather than
reconstructed:

```
PDFPage(source, page_number, metadata)
   → TextChunk(chunk_id, source, page_number, chunk_index, section, metadata)
     → RetrievalResult(chunk, score, rank)
       → Evidence(marker, result)
         → Citation(marker, chunk_id, source, page_number, section)
```

Nothing in that chain drops a field. A citation in a generated answer can be traced to a chunk
ID, and the chunk ID names its source, its page, and — through its digest — the exact text it
was written against.

Chunks never span a page boundary, which is what makes "p.4" unambiguous. The cost is that a
passage straddling a page break gets split; that tradeoff is taken deliberately.

## Dependency injection

Encoders, rerankers, and language models are all passed in rather than constructed internally.
This is not abstraction for its own sake — it is what makes the test suite offline and
deterministic. `tests/fakes.py` holds a fixed-vector encoder, a lookup-table reranker, and a
scripted language model, and the entire suite runs without a network.

The same injection makes the real code cheaper: one encoder is shared across dense, hybrid, and
reranked rather than each loading its own copy.

## Where cost lives

Roughly ordered, most expensive first:

1. **Embedding the corpus.** Cached in-process by `RetrieverFactory` and across runs by
   `EmbeddingCache`. See [Performance](../README.md#performance).
2. **Cross-encoder reranking.** Quadratic in corpus size if applied to everything, which is
   exactly why it sits behind a cheap retriever and only sees `top_k × multiplier` candidates.
3. **Chunking and BM25 indexing.** Linear and fast; the corpus is tokenized once at
   construction.
4. **Query encoding.** One short string per query. Never avoided, never worth avoiding.

Search is exact — a full matrix multiply, no approximate-nearest-neighbour index. At this
corpus size that is both correct and fast, and an ANN index would obscure the mechanics this
project exists to demonstrate.

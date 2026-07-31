# RAG Evaluation Platform

A hybrid retrieval and evaluation platform for technical documents.

Most "chat with your PDF" projects stop at *does it produce an answer?* This one asks the
harder question: **does the retriever actually find the right passage, and can you prove it?**
Retrieval quality is the ceiling on answer quality — a generator cannot ground an answer in
evidence it was never given. So the core of this project is a measured comparison of retrieval
strategies over a real technical paper, using Recall@K and nDCG@K against manually validated
relevance labels.

> **Status: complete pipeline, unmeasured benchmark.** Every stage described below is
> implemented and tested. What does not exist is the labelled dataset — no source document
> has been added, so [Results](#results) is empty and stays empty. No benchmark number
> appears anywhere in this repository until it has actually been measured.

---

## Table of contents

- [Problem](#problem)
- [Architecture](#architecture)
- [Chunking strategies](#chunking-strategies)
- [Retrieval methods](#retrieval-methods)
- [Evaluation methodology](#evaluation-methodology)
- [Grounded generation](#grounded-generation)
- [Interface](#interface)
- [Benchmark dataset](#benchmark-dataset)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Performance](#performance)
- [Tests](#tests)
- [Design principles](#design-principles)
- [Project layout](#project-layout)
- [Roadmap](#roadmap)
- [Limitations](#limitations)
- [Further reading](#further-reading)
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

Reranking **cannot raise Recall@K beyond what the first stage already surfaced** within its
candidate window — a chunk the retriever never returned cannot be promoted. What it improves is
ordering, which is what nDCG measures. A reranker that lifts nDCG while leaving recall flat is
working exactly as intended.

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

## Grounded generation

Generation is the last stage and the least trusted one. It consumes passages the retriever
already selected and never triggers or widens a search of its own — if it could, Recall@K
would stop describing what the answer was actually built from.

Passages are handed to the model numbered, each carrying its own provenance, and the model
is required to cite the marker supporting each claim. Retrieval *scores* are deliberately
withheld: a score is a within-retriever artifact that means nothing to a language model, and
showing it invites treating rank as truth. The ranking's job was to select the evidence, not
to weigh it.

Two mechanisms do the real work:

**Refusal.** When retrieval returns nothing, the model is never called at all — there is
nothing to ground an answer in, and asking anyway only buys an unsourced answer. When the
passages are present but do not answer the question, the model replies with a sentinel token
rather than a phrase, because "I don't have enough information" has a hundred paraphrases and
detecting refusal by fuzzy matching would eventually misread a real answer containing a hedge.

**Unresolved citations.** A marker the model invented is reported, not dropped. Silently
discarding `[9]` when only three passages were supplied would turn a hallucinated citation
into an answer that merely looks lightly sourced — so `ask` prints a warning and exits
non-zero. An answer is only `is_grounded` when it cites at least one real passage and invents
none.

The Anthropic client is a single `urllib` POST rather than an SDK, for the same reason there
is no vector database here: it is one endpoint, and writing it out keeps the dependency list
honest and the mechanics visible. No streaming, no retries, no pooling — none of which matter
behind a retriever that costs more than the request does.

## Interface

```bash
streamlit run streamlit_app.py
```

A single page: pick a document and a chunking strategy, choose among the four retrieval
strategies, set K, and optionally generate a cited answer. Retrieved passages are shown with
their rank, score, chunk ID, and full text, so what the ranking did is inspectable rather
than implied.

Streamlit reruns the entire script on every interaction, which makes the naive version of
this app re-chunk and re-embed the document on each keystroke. Both the corpus and every
retriever are therefore cached for the life of a session: switching from dense to hybrid to
reranked constructs one encoder, total. The cache key is `(document, chunker)` because
changing the chunker genuinely produces a different index — different chunks, different IDs —
rather than the same index viewed differently.

Generation is opt-in and degrades rather than failing: with no `ANTHROPIC_API_KEY` set, the
app says so and shows retrieval only. Retrieval never needs a key.

The dependency runs one way. `src/rag_eval/ui/service.py` holds everything the interface
does and imports no UI framework; `src/rag_eval/ui/app.py` is the only module in the project
that imports Streamlit, and nothing imports it back. Four tests enforce this by launching a
subprocess and asserting `streamlit` is absent from `sys.modules` after importing retrieval,
evaluation, the CLI, and the service layer.

## Benchmark dataset

The loading, validation, and evaluation machinery is implemented. **The labelled dataset
itself does not exist yet** — no source document has been added to `data/documents/`, and
a relevance label can only be written by a person reading the paper. The file format is
documented in [`data/evaluation/SCHEMA.md`](data/evaluation/SCHEMA.md), along with the
labelling procedure.

The intended benchmark document is a technical research paper,
*"Agentic AI for Disease-Aware Adaptive Multi-Omics Embedding"* (~7 pages), with roughly
10–15 hand-written questions spanning the query types that stress retrieval differently:

- exact / lexical terminology
- semantic paraphrases
- acronyms
- methodology questions
- numerical results
- scientific findings

Labels are validated against actual chunks produced by the ingestion pipeline — `evaluate`
refuses to run if any label fails to resolve, because a benchmark scored against the wrong
chunks is worse than no benchmark. No relevance label in this repository is generated or
guessed.

A chunk ID carries a digest of its own text, so a benchmark is valid **only** for the
chunking configuration it was labelled under. Comparing chunking strategies means
re-labelling, not reusing.

Source PDFs are gitignored; the benchmark questions and labels are version controlled, since
they are the reproducible part.

## Results

_Not yet measured._ No source document has been labelled, so there is nothing to report.

The runner that will produce these numbers is implemented and tested; what is missing is the
hand-labelled data, which cannot be generated. Once it exists this section will hold real
measured comparisons — retriever comparisons, chunking-strategy comparisons, and the latency
tradeoff. Nothing is reported here until it has actually been run, and results are reported as
measured whether or not they favor the more sophisticated method.

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

`requirements.txt` is the fully pinned environment. To install less, the package declares
extras that match how the code is layered:

```bash
pip install -e .              # ingestion, chunking, BM25, evaluation, generation
pip install -e ".[models]"    # + dense retrieval and cross-encoder reranking (pulls torch)
pip install -e ".[ui]"        # + the Streamlit interface
pip install -e ".[dev]"       # + pytest
```

The split is real, not cosmetic: **the test suite never needs `torch`**. Encoders, rerankers,
and language models are all dependency-injected and every test substitutes a fake, so CI
installs neither `torch` nor `sentence-transformers` — and then asserts they are absent, so a
future test cannot quietly start requiring one.

Installing the package also puts a `rag-eval` command on the path, equivalent to
`python -m rag_eval`.

Copy `.env.example` to `.env` if you need to set optional cache paths or, later, a generation
API key. Nothing in the current milestone requires it.

## Usage

Three commands, all reading the same ingestion → chunking → corpus path, so a chunk ID
means the same thing everywhere.

```bash
# Chunk a PDF and report corpus statistics
python -m rag_eval index data/documents/paper.pdf
python -m rag_eval index data/documents/paper.pdf --chunker structure

# Run one query
python -m rag_eval search data/documents/paper.pdf "how are missing modalities handled?" -k 5
python -m rag_eval search data/documents/paper.pdf "imputation prior" -r hybrid

# Run a labelled benchmark
python -m rag_eval evaluate data/documents/paper.pdf data/evaluation/benchmark.json -k 5
```

```bash
# Compare chunking configurations on the same document
python -m rag_eval sweep data/documents/paper.pdf -v fixed -v structure
python -m rag_eval sweep data/documents/paper.pdf \
    -v fixed:200/40=data/evaluation/fixed-200.json \
    -v fixed:400/80=data/evaluation/fixed-400.json \
    -r bm25 -r hybrid -o experiments/results/chunk-size.json
```

`--retriever` accepts `bm25`, `dense`, `hybrid`, or `reranked`; `evaluate` and `sweep` take
it repeatedly and default to all four. `bm25` never loads an embedding model — the dense
import happens only when a strategy that needs it is requested.

`sweep` takes each variant as `strategy[:A/B][=benchmark.json]`. With no benchmark attached
it still runs, reporting chunk-count and chunk-size distribution — the half of a chunking
experiment that needs no relevance labels. Attach a benchmark and it scores retrieval too.

```bash
# Answer a question from retrieved evidence, with page-level citations
export ANTHROPIC_API_KEY=...
python -m rag_eval ask data/documents/paper.pdf "how are missing modalities handled?"
python -m rag_eval ask data/documents/paper.pdf "what is the ablation result?" \
    -r reranked -k 8 --show-evidence
```

`ask` exits non-zero if the answer cites a passage that was never supplied. It defaults to
hybrid retrieval; every other strategy is available through the same `--retriever` flag.

```bash
# The interface
streamlit run streamlit_app.py
```

## Performance

Embedding the corpus is by far the slowest thing here, and the most wasteful to repeat: the
vectors depend only on the chunk text and the model, neither of which changes between runs.
Two layers of reuse follow from that.

**Within a run**, one factory builds all four strategies, so the encoder is loaded once and
the corpus embedded once no matter how many strategies are evaluated — hybrid and reranked
are assembled from the same components as dense rather than rebuilding them.

**Across runs**, `EmbeddingCache` writes the corpus matrix to `.cache/embeddings/`, keyed by
a digest of the model name and every chunk ID. Because chunk IDs already contain a digest of
their own text, that key covers the document *and* the chunking configuration: editing the
PDF, re-chunking, or switching models all miss cleanly.

The cache is safe to leave on because it cannot be wrong, only unhelpful. Every entry stores
the chunk IDs it was built from and verifies them on load, so a stale or colliding key is a
miss rather than silently mismatched vectors; corrupt files and failed writes are misses too.
The worst case is recomputation. Disable it with `--no-cache`, or point it elsewhere with
`--cache-dir`.

```bash
python -m rag_eval search data/documents/paper.pdf "imputation prior" -r hybrid
python -m rag_eval search data/documents/paper.pdf "imputation prior" -r hybrid  # warm
```

No wall-clock numbers are quoted here. Nothing has been measured on a real corpus yet, and an
unmeasured speedup is exactly the kind of number this project refuses to invent.

## Tests

```bash
python -m pytest -v
```

End-to-end tests run the assembled path — PDF → pages → chunks → corpus → retrieval → fusion
→ reranking → evaluation → generation — with only the ML models faked, because a pipeline can
be correct at every seam and still be wrong once put together.

Unit tests are deterministic and run fully offline. No test downloads a model, hits the
network, or calls an external API — embedding and reranking models are injected as
dependencies so they can be replaced with fakes. Any test that genuinely needs a downloaded
model will be marked as an integration test and kept separate.

## Design principles

1. Components are modular and independently testable.
2. Retrieval and evaluation never depend on Streamlit; the boundary is tested, not trusted.
3. Evaluation never depends on answer generation.
4. Generation consumes retrieved evidence; it does not control retrieval.
5. Provenance — source, page, chunk ID, metadata — survives every stage of the pipeline.
6. Corpus embeddings are computed once, not per query, and reused across runs.
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
├── experiments/   chunking sweeps and result reporting
├── generation/    grounded answers, citations, refusal
├── ui/            Streamlit app + the service layer beneath it
├── factory.py     builds the four strategies over one corpus, each once
└── cli.py         index / search / evaluate / sweep / ask

streamlit_app.py   Streamlit entry point (a one-line script)
tests/             unit tests (offline, deterministic)
data/              documents (gitignored) + benchmark labels (version controlled)
experiments/       generated results (gitignored)
```

Directories appear as the milestones that need them land, rather than as empty scaffolding.

## Roadmap

- [x] Repository, packaging, and Python 3.11 environment
- [x] PDF ingestion with page metadata
- [x] Fixed-size chunking with configurable overlap
- [x] Structure-aware chunking along headings and sentences
- [x] BM25 lexical retrieval
- [x] Dense embedding retrieval
- [x] Reciprocal rank fusion + hybrid retriever
- [x] Recall@K and nDCG@K + evaluation runner
- [ ] Hand-labeled benchmark on a real technical paper
- [x] Benchmark CLI (`index`, `search`, `evaluate`)
- [ ] Baseline results — needs a labelled document
- [x] Cross-encoder reranking
- [x] Chunking strategy experiment harness
- [ ] Chunking strategy results — needs labelled documents
- [x] Grounded generation with page-level citations
- [x] Streamlit interface
- [x] End-to-end tests and cross-run embedding cache
- [x] Architecture, methodology, and decision documentation
- [x] Packaging, CI, and public release

## Further reading

Longer-form notes live in [`docs/`](docs/):

| Document | Contents |
| --- | --- |
| [Architecture](docs/architecture.md) | Module map, the path a query takes, the four import boundaries and how they are enforced, where cost lives |
| [Methodology](docs/methodology.md) | Recall@K, nDCG@K, and RRF worked by hand; the Okapi IDF degeneracy; why relevance is binary; what the numbers cannot tell you |
| [Design decisions](docs/decisions.md) | Thirteen choices that could have gone the other way, each with what it costs |
| [Benchmark format](data/evaluation/SCHEMA.md) | The label file schema and the labelling procedure |

Every worked figure in the methodology notes is recomputed by `tests/test_docs.py`, so a
change in the code that contradicts the documentation fails the suite rather than drifting
quietly.

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

MIT — see [LICENSE](LICENSE).

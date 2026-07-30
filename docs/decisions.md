# Design decisions

Choices that could reasonably have gone the other way, and why they went this way. Each names
what it costs, because a decision recorded without its downside is advertising.

- [No vector database](#no-vector-database)
- [No RAG framework](#no-rag-framework)
- [Chunk IDs contain a text digest](#chunk-ids-contain-a-text-digest)
- [Sizes counted in words](#sizes-counted-in-words)
- [Chunks never span pages](#chunks-never-span-pages)
- [Conservative heading detection](#conservative-heading-detection)
- [Fusion over ranks, not scores](#fusion-over-ranks-not-scores)
- [Zero-scoring BM25 chunks are dropped](#zero-scoring-bm25-chunks-are-dropped)
- [Ties break on chunk ID](#ties-break-on-chunk-id)
- [No SDK for the generation API](#no-sdk-for-the-generation-api)
- [A refusal sentinel, not a phrase](#a-refusal-sentinel-not-a-phrase)
- [Unresolved citations are reported](#unresolved-citations-are-reported)
- [Comparison output is never sorted by score](#comparison-output-is-never-sorted-by-score)

## No vector database

Exact search over a few hundred chunks is one matrix multiply against an L2-normalized matrix.
Adding FAISS, Chroma, or a hosted index would introduce an approximate index, a persistence
format, and a service boundary in exchange for a speedup that is unmeasurable at this size.

**Cost:** this does not scale past a corpus that fits comfortably in memory. That limit is
real, and the right fix at that point is an ANN index — not a different architecture.

## No RAG framework

LangChain or LlamaIndex would supply chunking, retrievers, fusion, and an evaluation harness
out of the box. The point of this project is to demonstrate those mechanics, so importing them
would delete the deliverable.

**Cost:** more code to maintain, and none of the integrations a framework brings.

## Chunk IDs contain a text digest

An ID looks like `omics-p004-c02-1f9ab3c7`: readable provenance plus a SHA-256 prefix of the
chunk's own text. This makes a relevance label self-validating — change the document or the
chunking settings and the label stops resolving, loudly, instead of silently pointing at text
it was never written against.

**Cost:** re-chunking invalidates every label. Comparing two chunking strategies therefore
requires labelling twice. That is expensive and it is the honest price; the alternative is a
benchmark that quietly measures the wrong thing.

## Sizes counted in words

Word counts track token counts far more closely than character counts do, BM25 tokenizes on
word boundaries anyway, and a boundary that lands on a word can never bisect an acronym or an
identifier.

**Cost:** not exact tokens. A chunk sized at 200 words is not 200 tokens for any particular
tokenizer, so a hard model context limit needs headroom rather than arithmetic.

## Chunks never span pages

Every chunk cites exactly one page, so "p.4" in an answer is unambiguous and checkable.

**Cost:** a passage straddling a page break is split, and neither half may retrieve well on
its own. Accepted because an unverifiable citation defeats the purpose of citing.

## Conservative heading detection

Headings are detected from known section names, numbered headings (`3.2 Imputation Strategy`),
and all-caps lines — but deliberately **not** generic title case, which fires on figure
captions and author lines.

The asymmetry is the reason: a missed heading degrades to paragraph-boundary chunking, which is
still reasonable. A false heading splits a section in half, producing a chunk that begins
mid-argument. Under-detection is the cheaper failure.

**Cost:** documents whose headings are only distinguished by typography get chunked as plain
prose.

## Fusion over ranks, not scores

BM25 scores are unbounded and corpus-dependent; cosine similarities live in `[-1, 1]`. Adding
them makes the weighting an artifact of score distributions. Per-query min-max normalization
looks principled but just relocates the arbitrariness. RRF uses rank position only. See
[Methodology](methodology.md#reciprocal-rank-fusion).

**Cost:** the magnitude of a retriever's confidence is discarded. A chunk BM25 is certain about
and one it barely matched contribute identically if they rank the same.

## Zero-scoring BM25 chunks are dropped

A chunk sharing no terms with the query scores zero and is excluded rather than returned at the
bottom of the list.

**Cost:** `search` can return fewer than K results, which surprises people. The alternative
surprises them worse: padding the list implies relevance that was never measured, and it
inflates fused rankings with documents that matched nothing.

## Ties break on chunk ID

Equal scores are ordered by chunk ID, not by corpus order.

Without this, two chunks with identical scores swap places depending on how the corpus happened
to be built, and a benchmark whose numbers move when the input order changes is not measuring
retrieval quality. An end-to-end test asserts a fully-tied ranking is identical over a reversed
corpus.

**Cost:** the tie-break is arbitrary — it just has to be *stable*.

## No SDK for the generation API

Grounded generation is one HTTPS POST to one endpoint, written with `urllib`. No new
dependency, and the mechanics stay visible.

**Cost:** no streaming, no automatic retries, no connection pooling. None of these matter for
answering one question at a time behind a retriever that costs more than the request does. If
this grew into a service, an SDK would earn its place.

## A refusal sentinel, not a phrase

When the evidence does not answer the question, the model replies with an exact token rather
than prose, and the token is matched exactly — not searched for within the reply.

"I don't have enough information" has a hundred paraphrases; detecting refusal by fuzzy string
matching would eventually misread a real answer that happens to contain a hedge. A test asserts
that an answer merely *containing* the sentinel is still an answer.

**Cost:** depends on the model following an instruction exactly. When it does not, the reply
falls through as a normal answer — which is the safe direction, since it will then be visibly
uncited.

## Unresolved citations are reported

A marker the model invented (`[9]` when three passages were supplied) is recorded in
`unresolved_citations`, surfaced in the UI, and exits `ask` non-zero.

Dropping it silently would turn a hallucinated citation into an answer that merely *looks*
lightly sourced. This is the failure mode the whole project exists to make visible.

**Cost:** a noisier interface. Worth it.

## Comparison output is never sorted by score

`compare()` returns reports in the order the retrievers were given, never ranked by
performance.

Ordering by score invites reading row one as "the winner", and what a comparison means is the
caller's judgement. A simpler method beating a more sophisticated one is a result to report,
not to bury — which is also why nothing in this repository tunes a setup until hybrid or
reranking wins.

**Cost:** the reader has to look at the numbers.

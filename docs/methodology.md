# Methodology

The maths this project implements explicitly, worked by hand so a reader can verify the code
against the definition rather than trusting either.

- [Recall@K](#recallk)
- [nDCG@K](#ndcgk)
- [Reciprocal rank fusion](#reciprocal-rank-fusion)
- [BM25, and where it degenerates](#bm25-and-where-it-degenerates)
- [Why binary relevance](#why-binary-relevance)
- [What these numbers cannot tell you](#what-these-numbers-cannot-tell-you)

## Recall@K

Of the chunks known to be relevant, what fraction appear in the top K?

```
Recall@K = |relevant ∩ retrieved[:K]| / |relevant|
```

The denominator is the label count, **not** K. A question with three relevant chunks that
surfaces two of them scores 0.667, regardless of whether K was 5 or 50.

Worked: labels `{a, c}`, retrieved `[b, a, d, e, f]`, K=5 → the intersection is `{a}`, so
`1/2 = 0.5`.

This is the metric that answers *did the evidence reach the generator at all*. A passage
outside the retrieved set cannot be recovered by any amount of prompt engineering.

## nDCG@K

Recall treats every position in the top K as equivalent. nDCG does not — it discounts each hit
logarithmically by rank and normalizes against the best possible ordering.

```
DCG@K  = Σ  rel_i / log2(i + 1)          i = 1..K, ranks 1-indexed
IDCG@K = Σ  1 / log2(i + 1)              i = 1..min(|relevant|, K)
nDCG@K = DCG@K / IDCG@K
```

Worked, labels `{a, c}`, retrieved `[b, a, d, c, e]`, K=5:

| rank i | chunk | rel | 1/log2(i+1) | contribution |
| --- | --- | --- | --- | --- |
| 1 | b | 0 | 1.000 | 0 |
| 2 | a | 1 | 0.631 | 0.631 |
| 3 | d | 0 | 0.500 | 0 |
| 4 | c | 1 | 0.431 | 0.431 |
| 5 | e | 0 | 0.387 | 0 |

`DCG@5 = 1.062`. The ideal ordering puts both relevant chunks first:
`IDCG@5 = 1.000 + 0.631 = 1.631`. So `nDCG@5 = 1.062 / 1.631 = 0.651`.

**IDCG is capped at K.** With five relevant chunks and K=3, the ideal is three hits, not five —
otherwise a perfect ranking would score below 1.0 and the metric would stop being normalized.

## Reciprocal rank fusion

BM25 scores are unbounded and corpus-dependent; cosine similarities live in `[-1, 1]`. Adding
them makes the weighting an artifact of score distributions rather than of retrieval quality,
and per-query min-max normalization just hides that behind arithmetic. RRF uses only rank
position:

```
RRF(d) = Σ  1 / (k + rank_i(d))          k = 60, ranks 1-indexed
         i
```

A document ranked highly by either retriever gets a strong contribution; one ranked highly by
both wins. Documents missing from a list simply contribute nothing from it.

Worked with k=60, BM25 `[a, b, z]` and dense `[z, a, b]`:

| chunk | BM25 | dense | RRF |
| --- | --- | --- | --- |
| a | 1 → 1/61 | 2 → 1/62 | 0.03252 |
| z | 3 → 1/63 | 1 → 1/61 | 0.03227 |
| b | 2 → 1/62 | 3 → 1/63 | 0.03200 |

`a` wins: consistently near the top of both beats first-place-in-one, third-in-the-other. That
is the intended behaviour, and `k = 60` is what tunes it — a large `k` flattens the difference
between ranks, a small one lets a single retriever's #1 dominate.

## BM25, and where it degenerates

Okapi IDF is:

```
IDF(q) = log(N − n(q) + 0.5) − log(n(q) + 0.5)
```

At `N = 2, n = 1` this is `log(1.5) − log(1.5) = 0` **exactly**. On a two-document corpus where
a term appears in one document, BM25 scores it zero and returns nothing.

This is not a bug, it is the formula behaving correctly on a corpus far too small for it — a
term in half the collection carries no discriminative information. It is documented here
because it is genuinely surprising the first time a small test fixture returns an empty result
set. Test fixtures in this repository use seven pages for that reason.

Chunks scoring exactly zero are dropped rather than returned, so an empty result means "nothing
matched" rather than "here are some documents that share no terms with your query".

## Why binary relevance

Labels are binary: a chunk either answers the question or it does not. Graded relevance (a
0–3 scale, say) is standard in IR benchmarks and would make nDCG more expressive.

It is not used here because the labels are hand-written by one person on a small,
single-document benchmark. A graded scale needs either multiple annotators or a rubric strict
enough to be reproducible by one; without that, the grades encode the labeller's mood on the
day, and nDCG then reports that mood with four decimal places of false precision. Binary is
the claim this labelling process can actually support.

## What these numbers cannot tell you

Stated plainly, because a benchmark that oversells itself is worse than none:

- **Single document.** Results characterize *this paper*, not technical documents in general.
- **One labeller.** No inter-annotator agreement can be computed, so label noise is unknown
  rather than small.
- **Small question set.** A handful of questions means wide confidence intervals; a gap of a
  few points between two retrievers is not evidence that one is better.
- **Cross-chunking comparisons are confounded.** Comparing fixed-size against structure-aware
  requires labelling twice, since chunk IDs do not survive re-chunking. Any gap therefore mixes
  chunking quality with labelling variance. Retriever comparisons *within* one chunking
  configuration share a single label set and carry no such confound — those are the clean
  comparisons.
- **Retrieval ≠ answer quality.** These metrics measure whether the right passage was found and
  ranked well. They say nothing about whether the generated answer used it correctly.

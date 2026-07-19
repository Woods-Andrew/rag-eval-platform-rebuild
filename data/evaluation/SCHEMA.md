# Benchmark file format

A benchmark is a single JSON file of hand-written questions with manually verified
relevant chunk IDs. It lives here, under version control, because it is the reproducible
part of the experiment — the source PDF is gitignored, the labels are not.

**No benchmark file is committed yet.** Labels must be written against a real document
by a human reading it, so this directory holds the format and nothing else until that
work is done. See [Labelling procedure](#labelling-procedure) below.

## Shape

```json
{
  "document": "agentic-multi-omics.pdf",
  "chunking": "fixed(chunk_size=200, overlap=40)",
  "notes": "Labelled by reading each candidate chunk against the paper.",
  "questions": [
    {
      "id": "q01",
      "query": "How are missing modalities handled?",
      "category": "paraphrase",
      "relevant_chunk_ids": ["agentic-multi-omics-p004-c02-1f9ab3c7"],
      "notes": "Paper says 'imputation prior'; the query deliberately avoids that term."
    }
  ]
}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `document` | no | Source PDF the labels were written against |
| `chunking` | no | Chunker and settings used when labelling — labels are only valid for these |
| `notes` | no | Free text about how labelling was done |
| `questions[].id` | **yes** | Unique within the file |
| `questions[].query` | **yes** | The question as a user would ask it |
| `questions[].relevant_chunk_ids` | **yes** | At least one; every ID must exist in the corpus |
| `questions[].category` | no | Defaults to `uncategorized` |
| `questions[].notes` | no | Why this chunk is relevant, for the next reader |

## Categories

Used to report *where* a retriever wins, not just that it does. The intended spread:

`lexical`, `paraphrase`, `acronym`, `methodology`, `numeric`, `finding`.

## Why chunk IDs, and why they are strict

A chunk ID looks like `agentic-multi-omics-p004-c02-1f9ab3c7`: readable provenance plus
a SHA-256 prefix of the chunk text. The digest is what makes labels trustworthy — change
the document or the chunking settings and every ID changes, so a stale label fails to
resolve instead of silently pointing at text it was never written against.

`load_benchmark` therefore **rejects** any file whose labels do not all resolve against
the corpus:

```
error: benchmark.json: 1 question(s) reference chunk IDs that are not in the corpus,
so the document or the chunking settings changed since labelling (q01 -> ...)
```

This is deliberate. A benchmark that scores against the wrong chunks is worse than no
benchmark, because it produces numbers that look real.

**A benchmark is valid only for the chunking configuration it was labelled under.**
Comparing chunking strategies means re-labelling, not reusing.

## Labelling procedure

1. Put the PDF in `data/documents/` (gitignored).
2. `python -m rag_eval index data/documents/<paper>.pdf` to produce the chunks.
3. Write each question, then find its supporting chunks by reading — `search` is a way
   to locate candidates, never a way to decide the label. Labelling from retriever
   output would score the retriever against its own opinion.
4. Record the chunk ID and a note saying why that chunk answers the question.
5. `python -m rag_eval evaluate data/documents/<paper>.pdf data/evaluation/<file>.json`
   fails loudly if any label does not resolve.

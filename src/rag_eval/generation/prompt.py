"""The grounding prompt: evidence in, cited answer out."""

from __future__ import annotations

from collections.abc import Sequence

from .models import Evidence

__all__ = ["INSUFFICIENT_EVIDENCE", "NO_EVIDENCE_TEXT", "SYSTEM_PROMPT", "build_prompt"]

# The model emits this exact token when the passages do not answer the question. A
# sentinel rather than a phrase to match on: "I don't have enough information" has a
# hundred paraphrases, and detecting refusal by fuzzy string matching would eventually
# misread a real answer that happens to contain a hedge.
INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"

NO_EVIDENCE_TEXT = (
    "The retrieved passages do not contain enough information to answer this question."
)

SYSTEM_PROMPT = f"""\
You answer questions about a technical document using only the numbered passages you \
are given.

Rules:
- Use only the passages. Do not add facts from outside them, and do not fill gaps with \
what you expect a paper like this to say.
- Cite every claim with the marker of the passage supporting it, like [2]. A sentence \
resting on two passages gets both, like [1][3].
- Never cite a marker that is not in the passage list.
- If the passages do not answer the question, reply with exactly \
{INSUFFICIENT_EVIDENCE} and nothing else. A partial answer built on a passage that \
does not really address the question is worse than no answer.
- Be concise and technical. Do not restate the question or describe what the passages \
are.\
"""


def build_prompt(query: str, evidence: Sequence[Evidence]) -> str:
    """Render the numbered evidence and the question into one user message.

    Passages carry their markers and their provenance, but *not* their retrieval
    scores. A score is a within-retriever artifact that means nothing to a language
    model, and showing it invites the model to treat rank as truth — the ranking's job
    was to select the evidence, not to weigh it.
    """
    if not evidence:
        raise ValueError("cannot build a grounding prompt with no evidence")

    passages = "\n\n".join(
        f"[{item.marker}] ({item.citation})\n{item.text.strip()}" for item in evidence
    )
    return f"Passages:\n\n{passages}\n\nQuestion: {query.strip()}"

"""Generation: grounded answers with page-level citations.

Strictly downstream of retrieval. Nothing here re-runs or widens a search, and nothing
in ``evaluation`` imports this package — retrieval quality is measured on its own terms,
not through the answers it happens to enable.
"""

from __future__ import annotations

from .generator import DEFAULT_MAX_EVIDENCE, AnswerGenerator
from .llm import (
    DEFAULT_GENERATION_MODEL,
    ClaudeLanguageModel,
    GenerationError,
    LanguageModel,
)
from .models import Citation, Evidence, GroundedAnswer, evidence_from_results
from .prompt import INSUFFICIENT_EVIDENCE, NO_EVIDENCE_TEXT, SYSTEM_PROMPT, build_prompt

__all__ = [
    "DEFAULT_GENERATION_MODEL",
    "DEFAULT_MAX_EVIDENCE",
    "INSUFFICIENT_EVIDENCE",
    "NO_EVIDENCE_TEXT",
    "SYSTEM_PROMPT",
    "AnswerGenerator",
    "Citation",
    "ClaudeLanguageModel",
    "Evidence",
    "GenerationError",
    "GroundedAnswer",
    "LanguageModel",
    "build_prompt",
    "evidence_from_results",
]

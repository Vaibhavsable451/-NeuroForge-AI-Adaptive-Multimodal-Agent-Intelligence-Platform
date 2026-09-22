"""
AI Governance & Hallucination Firewall
----------------------------------------
Implements:
  - PII detection (regex-based; swap for a proper NER/PII model in prod)
  - Prompt-injection heuristic detection
  - Hallucination firewall: claim extraction -> evidence matching ->
    accept/regenerate decision, mirroring the pipeline in the design doc.
  - Token/cost/latency accounting.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from app.models import (
    GovernanceReport,
    HallucinationReport,
    RAGResult,
    RiskLevel,
)

# --------------------------------------------------------------------------
# PII detection
# --------------------------------------------------------------------------
_PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"\b(\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}


def detect_pii(text: str) -> List[str]:
    found = []
    for label, pattern in _PII_PATTERNS.items():
        if pattern.search(text):
            found.append(label)
    return found


# --------------------------------------------------------------------------
# Prompt injection heuristics
# --------------------------------------------------------------------------
_INJECTION_MARKERS = re.compile(
    r"(ignore (all|previous|the above) instructions|"
    r"disregard (all|previous) (instructions|rules)|"
    r"you are now|new system prompt|reveal your (system prompt|instructions)|"
    r"act as (dan|jailbreak))",
    re.IGNORECASE,
)


def detect_prompt_injection(text: str) -> bool:
    return bool(_INJECTION_MARKERS.search(text))


# --------------------------------------------------------------------------
# Hallucination firewall
# --------------------------------------------------------------------------
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _extract_claims(answer: str) -> List[str]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer) if s.strip()]
    # A "claim" is any declarative sentence with enough content words.
    return [s for s in sentences if len(s.split()) >= 4]


def _claim_supported(claim: str, context: str) -> bool:
    if not context:
        return False
    claim_terms = {w.lower().strip(".,!?") for w in claim.split() if len(w) > 3}
    context_terms = {w.lower().strip(".,!?") for w in context.split() if len(w) > 3}
    if not claim_terms:
        return True
    overlap = len(claim_terms & context_terms) / len(claim_terms)
    return overlap >= 0.25


def run_hallucination_firewall(
    answer: str, rag: Optional[RAGResult]
) -> HallucinationReport:
    context = rag.context if rag else ""
    claims = _extract_claims(answer)

    # If no retrieval was performed at all (domain didn't require RAG, e.g.
    # coding/reasoning), there is no evidence set to check claims against.
    # Flagging every such answer as "hallucinated" would be a false
    # positive by construction, not a real detection — so we only run the
    # claim/evidence match when we actually have evidence to match against.
    if not claims or not context:
        return HallucinationReport(
            hallucination_score=0.0,
            grounding_score=1.0,
            citations_verified=bool(rag and rag.chunks),
            risk_level=RiskLevel.LOW,
            unsupported_claims=[],
        )

    unsupported = [c for c in claims if not _claim_supported(c, context)]
    hallucination_score = round(len(unsupported) / len(claims), 4)
    grounding_score = round(1.0 - hallucination_score, 4)

    if hallucination_score <= 0.15:
        risk = RiskLevel.LOW
    elif hallucination_score <= 0.4:
        risk = RiskLevel.MEDIUM
    else:
        risk = RiskLevel.HIGH

    return HallucinationReport(
        hallucination_score=hallucination_score,
        grounding_score=grounding_score,
        citations_verified=bool(rag and rag.chunks) and hallucination_score < 0.5,
        risk_level=risk,
        unsupported_claims=unsupported,
    )


# --------------------------------------------------------------------------
# Token / cost accounting (simple estimator; swap for provider usage stats)
# --------------------------------------------------------------------------
_COST_PER_1K_TOKENS_USD = 0.003


def estimate_tokens(text: str) -> int:
    # Rough heuristic: ~1.3 tokens per word, no external tokenizer needed.
    return max(1, int(len(text.split()) * 1.3))


def estimate_cost(tokens: int) -> float:
    return round((tokens / 1000) * _COST_PER_1K_TOKENS_USD, 6)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def evaluate(
    user_message: str,
    answer: str,
    rag: Optional[RAGResult],
    latency_ms: float,
) -> Tuple[GovernanceReport, HallucinationReport]:
    pii = detect_pii(user_message) + detect_pii(answer)
    pii = sorted(set(pii))
    injection = detect_prompt_injection(user_message)

    hallucination = run_hallucination_firewall(answer, rag)

    tokens = estimate_tokens(user_message) + estimate_tokens(answer)
    cost = estimate_cost(tokens)

    blocked = False
    reason = None
    if injection:
        blocked = True
        reason = "prompt_injection_detected"
    elif hallucination.risk_level == RiskLevel.HIGH:
        blocked = True
        reason = "hallucination_risk_high"

    report = GovernanceReport(
        pii_detected=pii,
        prompt_injection_detected=injection,
        hallucination=hallucination,
        tokens_used=tokens,
        estimated_cost_usd=cost,
        latency_ms=round(latency_ms, 2),
        blocked=blocked,
        block_reason=reason,
    )
    return report, hallucination

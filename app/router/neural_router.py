"""
Adaptive Neural Router
----------------------

Routes incoming requests by deciding:

- domain
- complexity
- required tools
- expert
- adapter

Architecture:

1. NeuroForge-specific intent rules
2. Lightweight TF-IDF + Logistic Regression classifier
3. Complexity estimation
4. Tool selection
5. Expert / adapter selection

The NeuroForge-specific rules run before the ML classifier so that
platform knowledge / architecture questions reliably reach RAG instead
of being misclassified as analytics or another domain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from app.models import Complexity, Domain, RoutingDecision


# ==========================================================================
# BOOTSTRAP TRAINING DATA
# ==========================================================================

_bootstrap_dataset: List[Tuple[str, str]] = [

    # ----------------------------------------------------------------------
    # CODING
    # ----------------------------------------------------------------------
    ("convert this java method to python", "coding"),
    ("fix the bug in this react component", "coding"),
    ("write a sql query to join two tables", "coding"),
    ("refactor this function to be async", "coding"),
    ("explain this stack trace and how to fix it", "coding"),
    ("write a fastapi endpoint", "coding"),
    ("debug this python code", "coding"),
    ("implement this java class", "coding"),
    ("fix this api error", "coding"),

    # ----------------------------------------------------------------------
    # RESEARCH
    # ----------------------------------------------------------------------
    ("analyze this company's financial report and explain the risk", "research"),
    ("summarize the latest research papers on transformers", "research"),
    ("what are the key findings in this document", "research"),
    ("compare these two market strategies", "research"),
    ("find supporting evidence for this claim", "research"),
    ("explain this technical document", "research"),
    ("summarize this knowledge base", "research"),
    ("what does this document say", "research"),
    ("explain the information in these documents", "research"),

    # ----------------------------------------------------------------------
    # REASONING
    # ----------------------------------------------------------------------
    ("why did the model make this decision, explain step by step", "reasoning"),
    ("solve this logic puzzle", "reasoning"),
    ("plan a multi-step approach to this problem", "reasoning"),
    ("what is the best strategy given these constraints", "reasoning"),
    ("reason through this problem", "reasoning"),
    ("evaluate these alternatives", "reasoning"),

    # ----------------------------------------------------------------------
    # SECURITY
    # ----------------------------------------------------------------------
    ("is this request a prompt injection attempt", "security"),
    ("scan this code for vulnerabilities", "security"),
    ("check if this input contains sensitive pii", "security"),
    ("review this access control policy", "security"),
    ("audit this authentication flow for weaknesses", "security"),
    ("detect prompt injection", "security"),
    ("check for sensitive information", "security"),
    ("analyze this security vulnerability", "security"),

    # ----------------------------------------------------------------------
    # ANALYTICS
    # ----------------------------------------------------------------------
    ("compute the average latency across requests", "analytics"),
    ("show me a breakdown of token usage by model", "analytics"),
    ("what is the trend in hallucination rate this month", "analytics"),
    ("aggregate cost by department", "analytics"),
    ("plot request volume over time", "analytics"),
    ("show dashboard metrics", "analytics"),
    ("calculate average request latency", "analytics"),
    ("show token usage statistics", "analytics"),
]


# ==========================================================================
# TOOL HINTS
# ==========================================================================

_TOOL_HINTS: Dict[str, List[str]] = {

    "coding": [
        "code_interpreter",
        "repo_access",
    ],

    "research": [
        "rag_retrieval",
        "calculator",
    ],

    "reasoning": [
        "rag_retrieval",
    ],

    "security": [
        "pii_scanner",
        "prompt_injection_scanner",
    ],

    "analytics": [
        "metrics_store",
        "calculator",
    ],
}


# ==========================================================================
# ADAPTER MAP
# ==========================================================================

_ADAPTER_MAP: Dict[str, str] = {

    "coding": "coding-lora-v2",

    "research": "research-lora-v1",

    "reasoning": "general-base",

    "security": "enterprise-lora-v1",

    "analytics": "general-base",
}


# ==========================================================================
# COMPLEXITY SIGNALS
# ==========================================================================

_HIGH_COMPLEXITY_MARKERS = re.compile(
    r"\b("
    r"analyz|"
    r"explain.*risk|"
    r"multi[- ]?step|"
    r"strategy|"
    r"architecture|"
    r"compare|"
    r"evaluate|"
    r"comprehensive|"
    r"report|"
    r"pipeline|"
    r"system design|"
    r"how does .* work"
    r")\w*",
    re.IGNORECASE,
)


_LOW_COMPLEXITY_MARKERS = re.compile(
    r"\b("
    r"convert|"
    r"translate|"
    r"rename|"
    r"format|"
    r"what is|"
    r"define|"
    r"who is"
    r")\b",
    re.IGNORECASE,
)


# ==========================================================================
# NEUROFORGE KNOWLEDGE INTENT
# ==========================================================================

_NEUROFORGE_TERMS = (
    "neuroforge",
    "neuroforge ai",
    "adaptive neural router",
    "neural router",
    "mixture of experts",
    "mixture-of-experts",
    "mixtureofexperts",
    "moe",
    "lora",
    "lora lab",
    "quantization",
    "quantization engine",
    "agentic rag",
    "hybrid rag",
    "rag pipeline",
    "rag system",
    "mcp tool fabric",
    "mcp",
    "speech intelligence",
    "hallucination firewall",
    "ai governance",
    "governance layer",
    "multimodal agent",
    "agent intelligence",
)


_NEUROFORGE_KNOWLEDGE_PATTERNS = (
    re.compile(r"\bwhat\s+is\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+does\b", re.IGNORECASE),
    re.compile(r"\bhow\s+does\b", re.IGNORECASE),
    re.compile(r"\bhow\s+do\b", re.IGNORECASE),
    re.compile(r"\bexplain\b", re.IGNORECASE),
    re.compile(r"\bdescribe\b", re.IGNORECASE),
    re.compile(r"\barchitecture\b", re.IGNORECASE),
    re.compile(r"\bwork\b", re.IGNORECASE),
)


def _is_neuroforge_knowledge_query(text: str) -> bool:
    """
    Detect questions asking about NeuroForge itself or its architecture.

    Examples:

        What is NeuroForge AI?
        Explain NeuroForge architecture
        How does the adaptive neural router work?
        What is the MoE system?
        Explain the LoRA pipeline
        What is hybrid agentic RAG?
    """

    normalized = " ".join(
        text.lower()
        .replace("-", " ")
        .replace("_", " ")
        .split()
    )

    has_neuroforge_term = any(
        term in normalized
        for term in _NEUROFORGE_TERMS
    )

    if not has_neuroforge_term:
        return False

    # Direct knowledge questions.
    if any(
        pattern.search(normalized)
        for pattern in _NEUROFORGE_KNOWLEDGE_PATTERNS
    ):
        return True

    # A short query containing a NeuroForge-specific component is also
    # treated as a knowledge query.
    words = normalized.split()

    if len(words) <= 12:
        return True

    return False


# ==========================================================================
# MODEL
# ==========================================================================

@dataclass
class _Model:
    vectorizer: TfidfVectorizer
    classifier: LogisticRegression


def _train() -> _Model:
    """
    Train the lightweight bootstrap classifier.
    """

    texts = [text for text, _ in _bootstrap_dataset]

    labels = [domain for _, domain in _bootstrap_dataset]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
    )

    X = vectorizer.fit_transform(texts)

    classifier = LogisticRegression(
        max_iter=1000,
    )

    classifier.fit(X, labels)

    return _Model(
        vectorizer=vectorizer,
        classifier=classifier,
    )


# ==========================================================================
# ROUTER
# ==========================================================================

class AdaptiveNeuralRouter:
    """
    Adaptive Neural Router.

    Routing priority:

        NeuroForge knowledge intent
                    ↓
        TF-IDF classifier
                    ↓
        complexity
                    ↓
        tools
                    ↓
        expert + adapter
    """

    def __init__(self) -> None:
        self._model = _train()

    # ----------------------------------------------------------------------
    # DOMAIN CLASSIFICATION
    # ----------------------------------------------------------------------

    def _classify_domain(
        self,
        text: str,
    ) -> Tuple[Domain, float]:

        # --------------------------------------------------------------
        # NeuroForge knowledge queries must use RAG.
        # --------------------------------------------------------------

        if _is_neuroforge_knowledge_query(text):
            return Domain.RESEARCH, 0.99

        # --------------------------------------------------------------
        # Normal ML classification.
        # --------------------------------------------------------------

        X = self._model.vectorizer.transform([text])

        probs = self._model.classifier.predict_proba(X)[0]

        classes = self._model.classifier.classes_

        best_idx = probs.argmax()

        domain = Domain(classes[best_idx])

        confidence = float(probs[best_idx])

        return domain, confidence

    # ----------------------------------------------------------------------
    # COMPLEXITY
    # ----------------------------------------------------------------------

    def _estimate_complexity(
        self,
        text: str,
    ) -> Complexity:

        word_count = len(text.split())

        high_hits = len(
            _HIGH_COMPLEXITY_MARKERS.findall(text)
        )

        low_hits = len(
            _LOW_COMPLEXITY_MARKERS.findall(text)
        )

        # NeuroForge architecture / system questions should be treated
        # as substantial knowledge requests.
        if _is_neuroforge_knowledge_query(text):

            if any(
                marker in text.lower()
                for marker in (
                    "architecture",
                    "pipeline",
                    "how does",
                    "how do",
                    "system",
                    "work",
                )
            ):
                return Complexity.HIGH

            return Complexity.MEDIUM

        if high_hits > 0 or word_count > 40:
            return Complexity.HIGH

        if low_hits > 0 and word_count <= 15:
            return Complexity.LOW

        return Complexity.MEDIUM

    # ----------------------------------------------------------------------
    # PUBLIC ROUTING API
    # ----------------------------------------------------------------------

    def route(
        self,
        text: str,
    ) -> RoutingDecision:

        domain, confidence = self._classify_domain(text)

        complexity = self._estimate_complexity(text)

        tools = list(
            _TOOL_HINTS.get(
                domain.value,
                [],
            )
        )

        # --------------------------------------------------------------
        # NeuroForge knowledge requests always require RAG.
        # --------------------------------------------------------------

        if _is_neuroforge_knowledge_query(text):

            if "rag_retrieval" not in tools:
                tools.append("rag_retrieval")

        # --------------------------------------------------------------
        # High-complexity requests also get RAG grounding.
        # --------------------------------------------------------------

        elif (
            complexity == Complexity.HIGH
            and "rag_retrieval" not in tools
        ):

            tools.append("rag_retrieval")

        adapter = _ADAPTER_MAP.get(
            domain.value,
            "general-base",
        )

        return RoutingDecision(
            domain=domain,
            complexity=complexity,
            required_tools=tools,
            expert=domain,
            confidence=round(confidence, 4),
            adapter=adapter,
        )


# ==========================================================================
# SINGLETON
# ==========================================================================

router_singleton = AdaptiveNeuralRouter()
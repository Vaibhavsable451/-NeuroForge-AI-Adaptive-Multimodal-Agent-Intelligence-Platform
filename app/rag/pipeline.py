"""
NeuroForge AI — Agentic RAG Pipeline
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from app.rag.schemas import RAGResult, RetrievedChunk


# ============================================================
# Document
# ============================================================

@dataclass
class Document:
    doc_id: str
    text: str
    source: str
    metadata: dict


# ============================================================
# NeuroForge AI knowledge corpus
# ============================================================

_SEED_CORPUS: List[Document] = [
    Document(
        doc_id="neuroforge-architecture",
        text=(
            "NeuroForge AI is an Adaptive Multimodal Agent Intelligence Platform. "
            "Its architecture combines an Adaptive Neural Router, Mixture-of-Experts "
            "(MoE), LoRA Model Lab, Quantization Engine, Hybrid Agentic RAG, MCP "
            "Tool Fabric, Speech Intelligence, and AI Governance with a Hallucination "
            "Firewall. The platform uses FastAPI for backend APIs and Streamlit for "
            "the interactive dashboard. Redis provides supporting runtime infrastructure."
        ),
        source="neuroforge-architecture.md",
        metadata={
            "domain": "neuroforge",
            "component": "architecture",
        },
    ),

    Document(
        doc_id="neuroforge-router",
        text=(
            "The Adaptive Neural Router analyzes incoming requests and determines "
            "the appropriate processing path. It considers request domain and "
            "complexity and can route requests toward specialized expert workflows. "
            "The router is the entry point of the adaptive agent intelligence pipeline."
        ),
        source="adaptive-neural-router.md",
        metadata={
            "domain": "neuroforge",
            "component": "router",
        },
    ),

    Document(
        doc_id="neuroforge-moe",
        text=(
            "NeuroForge AI uses a Mixture-of-Experts architecture to support "
            "specialized expert routing. Instead of requiring every request to use "
            "the same expert path, the routing layer can select an expert according "
            "to task characteristics. This supports adaptive inference across "
            "different classes of AI workloads."
        ),
        source="mixture-of-experts.md",
        metadata={
            "domain": "neuroforge",
            "component": "moe",
        },
    ),

    Document(
        doc_id="neuroforge-lora",
        text=(
            "The LoRA Model Lab provides parameter-efficient model adaptation. "
            "LoRA, or Low-Rank Adaptation, freezes pretrained model weights and "
            "introduces trainable low-rank matrices. This reduces the number of "
            "parameters that must be trained while enabling task-specific adaptation."
        ),
        source="lora-model-lab.md",
        metadata={
            "domain": "neuroforge",
            "component": "lora",
        },
    ),

    Document(
        doc_id="neuroforge-quantization",
        text=(
            "The Quantization Engine supports reduced-precision model inference "
            "such as INT8 and INT4. Quantization can reduce model memory usage and "
            "inference cost and can improve deployment efficiency, with potential "
            "accuracy trade-offs depending on the model and workload."
        ),
        source="quantization-engine.md",
        metadata={
            "domain": "neuroforge",
            "component": "quantization",
        },
    ),

    Document(
        doc_id="neuroforge-rag",
        text=(
            "NeuroForge AI includes Hybrid Agentic RAG, or Retrieval-Augmented "
            "Generation. The RAG layer retrieves relevant knowledge before generation "
            "and can participate in an agentic workflow. Pinecone is used as the "
            "vector backend when configured, while the application also supports "
            "an in-memory TF-IDF fallback. Retrieved context is passed into the "
            "governance and response pipeline."
        ),
        source="hybrid-agentic-rag.md",
        metadata={
            "domain": "neuroforge",
            "component": "rag",
        },
    ),

    Document(
        doc_id="neuroforge-mcp",
        text=(
            "The MCP Tool Fabric provides a tool-integration layer for NeuroForge AI. "
            "It allows agent workflows to interact with external tools and services "
            "through structured tool interfaces. The routing layer can identify "
            "when tools are required for a request."
        ),
        source="mcp-tool-fabric.md",
        metadata={
            "domain": "neuroforge",
            "component": "mcp",
        },
    ),

    Document(
        doc_id="neuroforge-speech",
        text=(
            "Speech Intelligence provides speech-oriented capabilities in the "
            "NeuroForge multimodal pipeline. The platform exposes speech processing "
            "interfaces for transcription and text-to-speech workflows. These "
            "capabilities allow voice input and generated audio to participate "
            "in the broader agent pipeline."
        ),
        source="speech-intelligence.md",
        metadata={
            "domain": "neuroforge",
            "component": "speech",
        },
    ),

    Document(
        doc_id="neuroforge-governance",
        text=(
            "The NeuroForge AI Governance layer evaluates generated responses for "
            "safety and reliability. It includes PII detection, prompt-injection "
            "detection, hallucination evaluation, grounding evaluation, risk "
            "assessment, and response blocking. When hallucination risk is high, "
            "the governance layer can block the generated response instead of "
            "returning unsupported information."
        ),
        source="governance-hallucination-firewall.md",
        metadata={
            "domain": "neuroforge",
            "component": "governance",
        },
    ),

    Document(
        doc_id="neuroforge-pipeline",
        text=(
            "The primary NeuroForge AI processing flow is: user input enters the "
            "Adaptive Neural Router, the router selects an appropriate expert or "
            "processing path, retrieval and tools are invoked when required, the "
            "model generates a response, and the Governance and Hallucination "
            "Firewall evaluates the result before it is returned to the user. "
            "The system is designed as an adaptive multimodal agent intelligence "
            "pipeline rather than a single-model chatbot."
        ),
        source="neuroforge-pipeline.md",
        metadata={
            "domain": "neuroforge",
            "component": "pipeline",
        },
    ),
]


# ============================================================
# TF-IDF fallback vector store
# ============================================================

class VectorStore:
    def __init__(self) -> None:
        self.documents: List[Document] = []
        self._vectors: List[dict[str, float]] = []
        self._idf: dict[str, float] = {}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(
            r"[a-zA-Z0-9_]+",
            text.lower(),
        )

    def _rebuild_index(self) -> None:
        tokenized = [
            self._tokenize(document.text)
            for document in self.documents
        ]

        document_frequency: dict[str, int] = {}

        for tokens in tokenized:
            for token in set(tokens):
                document_frequency[token] = (
                    document_frequency.get(token, 0) + 1
                )

        total_documents = len(self.documents)

        self._idf = {
            token: math.log(
                (1 + total_documents) / (1 + df)
            ) + 1.0
            for token, df in document_frequency.items()
        }

        self._vectors = []

        for tokens in tokenized:
            counts: dict[str, int] = {}

            for token in tokens:
                counts[token] = counts.get(token, 0) + 1

            total_terms = max(
                len(tokens),
                1,
            )

            vector = {
                token: (
                    count / total_terms
                ) * self._idf.get(token, 1.0)
                for token, count in counts.items()
            }

            self._vectors.append(vector)

    def upsert(
        self,
        documents: Sequence[Document],
    ) -> None:
        existing = {
            document.doc_id: index
            for index, document in enumerate(self.documents)
        }

        for document in documents:
            if document.doc_id in existing:
                self.documents[
                    existing[document.doc_id]
                ] = document
            else:
                self.documents.append(document)

        self._rebuild_index()

    def _vectorize_query(
        self,
        text: str,
    ) -> dict[str, float]:
        tokens = self._tokenize(text)

        counts: dict[str, int] = {}

        for token in tokens:
            counts[token] = counts.get(token, 0) + 1

        total_terms = max(
            len(tokens),
            1,
        )

        return {
            token: (
                count / total_terms
            ) * self._idf.get(token, 1.0)
            for token, count in counts.items()
        }

    @staticmethod
    def _cosine_similarity(
        left: dict[str, float],
        right: dict[str, float],
    ) -> float:
        if not left or not right:
            return 0.0

        dot = sum(
            value * right.get(key, 0.0)
            for key, value in left.items()
        )

        left_norm = math.sqrt(
            sum(
                value * value
                for value in left.values()
            )
        )

        right_norm = math.sqrt(
            sum(
                value * value
                for value in right.values()
            )
        )

        if (
            left_norm == 0.0
            or right_norm == 0.0
        ):
            return 0.0

        return dot / (
            left_norm * right_norm
        )

    def query(
        self,
        text: str,
        top_k: int = 4,
        metadata_filter: dict | None = None,
    ) -> List[dict]:

        if not self.documents:
            return []

        query_vector = self._vectorize_query(text)

        results = []

        for document, vector in zip(
            self.documents,
            self._vectors,
        ):
            if metadata_filter:
                matches = all(
                    document.metadata.get(key) == value
                    for key, value in metadata_filter.items()
                )

                if not matches:
                    continue

            score = self._cosine_similarity(
                query_vector,
                vector,
            )

            results.append(
                {
                    "doc_id": document.doc_id,
                    "text": document.text,
                    "source": document.source,
                    "metadata": document.metadata,
                    "score": float(score),
                }
            )

        results.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        return results[:max(1, top_k)]


# ============================================================
# Store builder
# ============================================================

def _build_store() -> tuple[object, str]:
    api_key = os.environ.get(
        "PINECONE_API_KEY"
    )

    if api_key:
        try:
            from app.rag.pinecone_store import (
                PineconeVectorStore,
            )

            store = PineconeVectorStore(
                api_key=api_key,
                index_name=os.environ.get(
                    "PINECONE_INDEX_NAME",
                    "neuroforge-ai-1024",
                ),
                cloud=os.environ.get(
                    "PINECONE_CLOUD",
                    "aws",
                ),
                region=os.environ.get(
                    "PINECONE_REGION",
                    "us-east-1",
                ),
                namespace=os.environ.get(
                    "PINECONE_NAMESPACE",
                    "default",
                ),
            )

            return store, "pinecone"

        except Exception as exc:
            print(
                "[rag] PINECONE_API_KEY set but "
                "Pinecone init failed "
                f"({exc}); falling back to "
                "in-memory TF-IDF store."
            )

    return VectorStore(), "tfidf-memory"


# ============================================================
# Agentic RAG
# ============================================================

class AgenticRAG:
    def __init__(self) -> None:
        self.store, self.backend = _build_store()

        # Insert NeuroForge knowledge at startup.
        self.store.upsert(
            _SEED_CORPUS
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        metadata_filter: dict | None = None,
    ) -> RAGResult:
        """
        Retrieve relevant knowledge.

        Returns the shared Pydantic RAGResult model so:
          - MoE can use rag.context
          - governance can inspect the result
          - FastAPI ChatResponse accepts it directly
        """

        raw_chunks = self.store.query(
            query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        chunks = [
            RetrievedChunk(
                text=chunk["text"],
                source=chunk["source"],
                score=float(
                    chunk["score"]
                ),
            )
            for chunk in raw_chunks
        ]

        context = "\n\n".join(
            (
                f"[Source: {chunk.source}]\n"
                f"{chunk.text}"
            )
            for chunk in chunks
        )

        return RAGResult(
            query=query,
            chunks=chunks,
            context=context,
        )

    def query(
        self,
        query: str,
        top_k: int = 4,
        metadata_filter: dict | None = None,
    ) -> RAGResult:
        """
        Alias for retrieve().
        """

        return self.retrieve(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )


# ============================================================
# Singleton used by app.main
# ============================================================

rag_singleton = AgenticRAG()
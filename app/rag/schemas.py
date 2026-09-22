"""
Shared RAG schemas for NeuroForge AI.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="ignore")

    text: str
    source: str
    score: float


class RAGResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str
    chunks: list[RetrievedChunk]
    context: str
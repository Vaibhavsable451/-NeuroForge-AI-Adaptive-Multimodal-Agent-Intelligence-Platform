"""
Pinecone-backed vector store for the Hybrid Agentic RAG pipeline.
--------------------------------------------------------------------
Activated automatically by `app/rag/pipeline.py` when PINECONE_API_KEY is
set in the environment; otherwise the pipeline falls back to the
in-memory TF-IDF `VectorStore`. Implements the same duck-typed interface
(`upsert(documents)`, `query(text, top_k, metadata_filter)`) so the two
are interchangeable.

Uses Pinecone's hosted inference API (`multilingual-e5-large`, 1024-dim)
to generate embeddings, so no local embedding model, torch install, or
GPU is required — only the `pinecone` package and an API key.

Configure via env vars (see .env.example):
  PINECONE_API_KEY       (required to activate this backend)
  PINECONE_INDEX_NAME     default: "neuroforge-ai"
  PINECONE_CLOUD          default: "aws"
  PINECONE_REGION         default: "us-east-1"
  PINECONE_NAMESPACE      default: "default"
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.models import RetrievedChunk

_EMBED_MODEL = "multilingual-e5-large"
_EMBED_DIMENSION = 1024


class PineconeVectorStore:
    """Real Pinecone-backed vector store: hosted embeddings + a serverless
    index for storage and cosine-similarity search."""

    def __init__(
        self,
        api_key: str,
        index_name: str = "neuroforge-ai",
        cloud: str = "aws",
        region: str = "us-east-1",
        namespace: str = "default",
    ) -> None:
        from pinecone import Pinecone, ServerlessSpec  # optional dependency

        self._pc = Pinecone(api_key=api_key)
        self._namespace = namespace
        self._index_name = index_name

        if not self._pc.has_index(index_name):
            self._pc.create_index(
                name=index_name,
                dimension=_EMBED_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        self._index = self._pc.Index(index_name)

    def _embed(self, texts: List[str], input_type: str) -> List[List[float]]:
        result = self._pc.inference.embed(
            model=_EMBED_MODEL,
            inputs=texts,
            parameters={"input_type": input_type, "truncate": "END"},
        )
        return [item["values"] for item in result]

    def upsert(self, documents: List[Any]) -> None:
        """`documents` are app.rag.pipeline.Document instances (duck-typed
        here as anything with .doc_id / .text / .source / .metadata) to
        avoid a circular import with pipeline.py."""
        if not documents:
            return
        vectors = self._embed([d.text for d in documents], input_type="passage")
        payload = [
            {
                "id": d.doc_id,
                "values": vec,
                "metadata": {"text": d.text, "source": d.source, **d.metadata},
            }
            for d, vec in zip(documents, vectors)
        ]
        self._index.upsert(vectors=payload, namespace=self._namespace)

    def query(
        self, text: str, top_k: int = 5, metadata_filter: Optional[Dict[str, str]] = None
    ) -> List[RetrievedChunk]:
        q_vec = self._embed([text], input_type="query")[0]
        response = self._index.query(
            vector=q_vec,
            top_k=top_k,
            namespace=self._namespace,
            filter=metadata_filter or None,
            include_metadata=True,
        )
        chunks: List[RetrievedChunk] = []
        for match in response.matches:
            meta = match.metadata or {}
            chunks.append(
                RetrievedChunk(
                    text=meta.get("text", ""),
                    source=meta.get("source", "pinecone"),
                    score=round(float(match.score), 4),
                )
            )
        return chunks

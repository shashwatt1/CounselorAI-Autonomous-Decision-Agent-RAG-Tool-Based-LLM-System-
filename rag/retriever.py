from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from openai import OpenAI


@dataclass(frozen=True)
class RetrievalQuery:
    rank: int
    category: str
    branch_preference: str
    state: str
    top_k: int = 10


class CollegeRetriever:
    """FAISS-based retriever for cutoff entries.

    This module handles retrieval only. It does not implement planning or
    recommendation logic.
    """

    def __init__(
        self,
        index_path: str | Path,
        metadata_path: str | Path,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.embedding_model = embedding_model
        self.index = self._load_index(self.index_path)
        self.documents = self._load_metadata(self.metadata_path)
        self.client = self._get_openai_client()

    def retrieve(
        self,
        *,
        rank: int,
        category: str,
        branch_preference: str,
        state: str,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        query = RetrievalQuery(
            rank=int(rank),
            category=str(category).upper().strip(),
            branch_preference=str(branch_preference).upper().strip(),
            state=str(state).upper().strip(),
            top_k=max(int(top_k), 1),
        )
        return self._retrieve_structured(query)

    def retrieve_from_dict(self, payload: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
        """Compatibility wrapper for dict-based callers."""
        return self.retrieve(
            rank=int(payload["rank"]),
            category=str(payload["category"]),
            branch_preference=str(payload.get("branch_preference", payload.get("preferred_branch", ""))),
            state=str(payload["state"]),
            top_k=top_k,
        )

    def _retrieve_structured(self, query: RetrievalQuery) -> Dict[str, Any]:
        vector = self._embed_query(self._build_query_text(query))
        distances, indices = self.index.search(vector, query.top_k)

        entries: List[Dict[str, Any]] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.documents):
                continue

            doc = self.documents[idx]
            metadata = doc.get("metadata", {})
            closing_rank = int(metadata.get("closing_rank", -1))
            is_eligible = closing_rank >= query.rank if closing_rank >= 0 else False

            entries.append(
                {
                    "doc_id": int(doc.get("doc_id", idx)),
                    "relevance_score": float(score),
                    "text": doc.get("text", ""),
                    "college_name": metadata.get("college_name"),
                    "branch": metadata.get("branch"),
                    "category": metadata.get("category"),
                    "closing_rank": closing_rank,
                    "state": metadata.get("state"),
                    "eligible_for_rank": is_eligible,
                }
            )

        return {
            "query": {
                "rank": query.rank,
                "category": query.category,
                "branch_preference": query.branch_preference,
                "state": query.state,
                "top_k": query.top_k,
            },
            "results_count": len(entries),
            "results": entries,
        }

    def _build_query_text(self, query: RetrievalQuery) -> str:
        return (
            f"Find engineering college cutoffs for branch {query.branch_preference}, "
            f"category {query.category}, state {query.state}, around rank {query.rank}."
        )

    def _embed_query(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(model=self.embedding_model, input=text)
        vector = np.asarray(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(vector)
        return vector

    @staticmethod
    def _load_index(index_path: Path) -> faiss.Index:
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        return faiss.read_index(str(index_path))

    @staticmethod
    def _load_metadata(metadata_path: Path) -> List[Dict[str, Any]]:
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Metadata JSON must be a list of documents.")
        return data

    @staticmethod
    def _get_openai_client() -> OpenAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set.")
        return OpenAI(api_key=api_key)

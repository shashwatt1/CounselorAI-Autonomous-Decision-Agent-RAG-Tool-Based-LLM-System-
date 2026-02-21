from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import faiss
import numpy as np
import pandas as pd
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CollegeDocument:
    doc_id: int
    text: str
    metadata: Dict[str, Any]


def load_cutoff_csv(csv_path: str | Path) -> pd.DataFrame:
    """Load and normalize cutoff data from CSV.

    Supports both:
    1. Long format: college_name, branch, category, closing_rank, state
    2. Wide format: college_name, branch, state, cutoff_<category>...
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV file is empty: {path}")

    df.columns = [_to_snake_case(c) for c in df.columns]

    if {"college_name", "branch", "category", "closing_rank", "state"}.issubset(df.columns):
        normalized = df.copy()
        normalized["category"] = normalized["category"].astype(str).str.upper().str.strip()
        normalized["closing_rank"] = pd.to_numeric(normalized["closing_rank"], errors="coerce")
        normalized = normalized.dropna(subset=["closing_rank"])
        normalized["closing_rank"] = normalized["closing_rank"].astype(int)
        return normalized

    wide_cutoff_cols = [c for c in df.columns if c.startswith("cutoff_")]
    required_wide_cols = {"college_name", "branch", "state"}
    if wide_cutoff_cols and required_wide_cols.issubset(df.columns):
        melted = df.melt(
            id_vars=["college_name", "branch", "state"],
            value_vars=wide_cutoff_cols,
            var_name="category",
            value_name="closing_rank",
        )
        melted["category"] = (
            melted["category"].str.replace("cutoff_", "", regex=False).str.upper().str.strip()
        )
        melted["closing_rank"] = pd.to_numeric(melted["closing_rank"], errors="coerce")
        melted = melted.dropna(subset=["closing_rank"])
        melted["closing_rank"] = melted["closing_rank"].astype(int)
        return melted

    raise ValueError(
        "Unsupported CSV schema. Expected long format "
        "({college_name, branch, category, closing_rank, state}) "
        "or wide format ({college_name, branch, state, cutoff_<category>...})."
    )


def build_documents(df: pd.DataFrame) -> List[CollegeDocument]:
    """Create text documents for retrieval and preserve structured metadata."""
    docs: List[CollegeDocument] = []

    for idx, row in df.reset_index(drop=True).iterrows():
        text = (
            f"College Name: {row['college_name']}\n"
            f"Branch: {row['branch']}\n"
            f"Category: {row['category']}\n"
            f"Closing Rank: {int(row['closing_rank'])}\n"
            f"State: {row['state']}"
        )
        docs.append(
            CollegeDocument(
                doc_id=idx,
                text=text,
                metadata={
                    "college_name": str(row["college_name"]),
                    "branch": str(row["branch"]),
                    "category": str(row["category"]),
                    "closing_rank": int(row["closing_rank"]),
                    "state": str(row["state"]),
                },
            )
        )
    return docs


def embed_documents(
    texts: Sequence[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 128,
) -> np.ndarray:
    """Generate OpenAI embeddings in batches using OPENAI_API_KEY."""
    if not texts:
        raise ValueError("No texts provided for embedding.")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    client = _get_openai_client()
    vectors: List[List[float]] = []

    for chunk in _chunked(texts, batch_size):
        response = client.embeddings.create(model=model, input=list(chunk))
        vectors.extend(item.embedding for item in response.data)

    embeddings = np.asarray(vectors, dtype=np.float32)
    if embeddings.ndim != 2:
        raise RuntimeError("Embedding API returned an unexpected shape.")

    return embeddings


def build_faiss_index(embeddings: np.ndarray, normalize_for_cosine: bool = True) -> faiss.Index:
    """Build a FAISS index from embedding vectors."""
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array.")
    if embeddings.shape[0] == 0:
        raise ValueError("embeddings array is empty.")

    vectors = embeddings.copy()
    if normalize_for_cosine:
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
    else:
        index = faiss.IndexFlatL2(vectors.shape[1])

    index.add(vectors)
    return index


def save_faiss_index(index: faiss.Index, output_path: str | Path) -> Path:
    """Persist a FAISS index to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out))
    return out


def save_metadata(documents: Sequence[CollegeDocument], output_path: str | Path) -> Path:
    """Save document metadata mapping for downstream retrieval display."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "doc_id": doc.doc_id,
            "text": doc.text,
            "metadata": doc.metadata,
        }
        for doc in documents
    ]
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def create_and_save_index(
    csv_path: str | Path,
    index_path: str | Path,
    metadata_path: str | Path | None = None,
    model: str = "text-embedding-3-small",
    batch_size: int = 128,
) -> Dict[str, Any]:
    """End-to-end utility: CSV -> docs -> embeddings -> FAISS -> disk."""
    df = load_cutoff_csv(csv_path)
    docs = build_documents(df)
    embeddings = embed_documents([doc.text for doc in docs], model=model, batch_size=batch_size)
    index = build_faiss_index(embeddings, normalize_for_cosine=True)
    saved_index = save_faiss_index(index, index_path)

    result: Dict[str, Any] = {
        "documents_indexed": len(docs),
        "embedding_dim": int(embeddings.shape[1]),
        "index_path": str(saved_index),
    }

    if metadata_path is not None:
        saved_metadata = save_metadata(docs, metadata_path)
        result["metadata_path"] = str(saved_metadata)

    return result


def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)


def _chunked(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _to_snake_case(name: str) -> str:
    return "_".join(name.strip().lower().replace("-", " ").split())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create and persist FAISS index from cutoff CSV.")
    parser.add_argument("--csv-path", required=True, help="Path to cutoff CSV file.")
    parser.add_argument("--index-path", required=True, help="Path to save FAISS index.")
    parser.add_argument(
        "--metadata-path",
        default=None,
        help="Optional path to save document metadata JSON.",
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="OpenAI embedding model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Embedding batch size.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()
    result = create_and_save_index(
        csv_path=args.csv_path,
        index_path=args.index_path,
        metadata_path=args.metadata_path,
        model=args.model,
        batch_size=args.batch_size,
    )
    logger.info("Index build completed: %s", result)


if __name__ == "__main__":
    main()

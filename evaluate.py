from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from agents.planner import PlannerAgent
from rag.retriever import CollegeRetriever

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_PROFILES: List[Dict[str, Any]] = [
    {"rank": 4500, "category": "GEN", "branch": "CSE", "state": "UP"},
    {"rank": 12000, "category": "OBC", "branch": "CSE", "state": "UP"},
    {"rank": 18000, "category": "OBC", "branch": "CSE", "state": "UP"},
    {"rank": 26000, "category": "SC", "branch": "CSE", "state": "UP"},
    {"rank": 34000, "category": "ST", "branch": "CSE", "state": "UP"},
]


def load_environment() -> None:
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
    except Exception:
        logger.debug("python-dotenv unavailable; using existing environment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CounselorAI retrieval and recommendation quality")
    parser.add_argument("--index-path", default="data/cutoff.index", help="Path to FAISS index")
    parser.add_argument("--metadata-path", default="data/cutoff_metadata.json", help="Path to metadata JSON")
    parser.add_argument("--output-csv", default="evaluation_results.csv", help="Path to save evaluation CSV")
    parser.add_argument("--retrieval-top-k", type=int, default=30, help="Retriever top-k candidates")
    parser.add_argument("--profiles-json", default=None, help="Optional path to JSON list of student profiles")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")


def load_profiles(profiles_json: str | None) -> List[Dict[str, Any]]:
    if not profiles_json:
        return DEFAULT_SAMPLE_PROFILES

    path = Path(profiles_json)
    if not path.exists():
        raise FileNotFoundError(f"Profiles JSON not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Profiles JSON must be a list of student profile objects.")

    return payload


def evaluate_profile(
    planner: PlannerAgent,
    retriever: CollegeRetriever,
    profile: Dict[str, Any],
    profile_id: int,
    retrieval_top_k: int,
) -> Dict[str, Any]:
    normalized = _normalize_profile(profile)

    retrieval = retriever.retrieve(
        rank=normalized["rank"],
        category=normalized["category"],
        branch_preference=normalized["branch"],
        state=normalized["state"],
        top_k=retrieval_top_k,
    )
    retrieval_accuracy = _compute_retrieval_accuracy(retrieval.get("results", []), normalized)

    started = time.perf_counter()
    planner_output = planner.run(student_input=normalized, user_id=f"eval_user_{profile_id}")
    latency_ms = (time.perf_counter() - started) * 1000.0

    top_colleges = planner_output.get("top_5_colleges", [])
    constraint_satisfaction = _compute_constraint_satisfaction(top_colleges, normalized)

    row = {
        "profile_id": profile_id,
        "rank": normalized["rank"],
        "category": normalized["category"],
        "branch": normalized["branch"],
        "state": normalized["state"],
        "retrieved_count": int(retrieval.get("results_count", 0)),
        "recommended_count": len(top_colleges),
        "retrieval_accuracy": round(retrieval_accuracy, 4),
        "constraint_satisfaction": round(constraint_satisfaction, 4),
        "response_latency_ms": round(latency_ms, 2),
        "overall_confidence": float(planner_output.get("overall_confidence", 0.0)),
        "status": "ok",
    }
    logger.info(
        "Profile %s | retrieval_accuracy=%.3f | constraint_satisfaction=%.3f | latency_ms=%.2f",
        profile_id,
        row["retrieval_accuracy"],
        row["constraint_satisfaction"],
        row["response_latency_ms"],
    )
    return row


def _normalize_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    branch = profile.get("branch") or profile.get("branch_preference") or profile.get("preferred_branch")
    normalized = {
        "rank": int(profile["rank"]),
        "category": str(profile["category"]).upper().strip(),
        "branch": str(branch or "CSE").upper().strip(),
        "state": str(profile["state"]).upper().strip(),
    }
    return normalized


def _compute_retrieval_accuracy(retrieved: List[Dict[str, Any]], profile: Dict[str, Any]) -> float:
    if not retrieved:
        return 0.0

    hits = 0
    for item in retrieved:
        if str(item.get("branch", "")).upper() != profile["branch"]:
            continue
        if str(item.get("state", "")).upper() != profile["state"]:
            continue
        if str(item.get("category", "")).upper() != profile["category"]:
            continue
        hits += 1

    return hits / len(retrieved)


def _compute_constraint_satisfaction(recommendations: List[Dict[str, Any]], profile: Dict[str, Any]) -> float:
    if not recommendations:
        return 0.0

    passed = 0
    for item in recommendations:
        if str(item.get("branch", "")).upper() != profile["branch"]:
            continue
        if str(item.get("state", "")).upper() != profile["state"]:
            continue
        if str(item.get("category", "")).upper() != profile["category"]:
            continue

        closing_rank = item.get("closing_rank")
        if closing_rank is None:
            continue
        if int(profile["rank"]) <= int(closing_rank):
            passed += 1

    return passed / len(recommendations)


def save_results(rows: List[Dict[str, Any]], output_csv: str | Path) -> Path:
    df = pd.DataFrame(rows)

    if not df.empty:
        summary = {
            "profile_id": "SUMMARY",
            "rank": "",
            "category": "",
            "branch": "",
            "state": "",
            "retrieved_count": int(df["retrieved_count"].mean()),
            "recommended_count": float(df["recommended_count"].mean()),
            "retrieval_accuracy": round(float(df["retrieval_accuracy"].mean()), 4),
            "constraint_satisfaction": round(float(df["constraint_satisfaction"].mean()), 4),
            "response_latency_ms": round(float(df["response_latency_ms"].mean()), 2),
            "overall_confidence": round(float(df["overall_confidence"].mean()), 4),
            "status": "summary",
        }
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    load_environment()

    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set.")

        profiles = load_profiles(args.profiles_json)

        retriever = CollegeRetriever(index_path=args.index_path, metadata_path=args.metadata_path)
        planner = PlannerAgent(
            retriever=retriever,
            retrieval_top_k=args.retrieval_top_k,
            recommendation_count=5,
        )

        rows: List[Dict[str, Any]] = []
        for idx, profile in enumerate(profiles, start=1):
            try:
                rows.append(
                    evaluate_profile(
                        planner=planner,
                        retriever=retriever,
                        profile=profile,
                        profile_id=idx,
                        retrieval_top_k=args.retrieval_top_k,
                    )
                )
            except Exception as exc:
                logger.exception("Evaluation failed for profile %s: %s", idx, exc)
                normalized = _normalize_profile(profile)
                rows.append(
                    {
                        "profile_id": idx,
                        "rank": normalized["rank"],
                        "category": normalized["category"],
                        "branch": normalized["branch"],
                        "state": normalized["state"],
                        "retrieved_count": 0,
                        "recommended_count": 0,
                        "retrieval_accuracy": 0.0,
                        "constraint_satisfaction": 0.0,
                        "response_latency_ms": 0.0,
                        "overall_confidence": 0.0,
                        "status": f"error: {type(exc).__name__}",
                    }
                )

        saved_to = save_results(rows, args.output_csv)
        logger.info("Evaluation complete. Results saved to %s", saved_to)
        return 0
    except Exception as exc:
        logger.exception("Evaluation run failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

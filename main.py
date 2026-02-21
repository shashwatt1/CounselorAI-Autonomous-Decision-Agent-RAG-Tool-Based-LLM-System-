from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

from agents.planner import PlannerAgent
from rag.retriever import CollegeRetriever

logger = logging.getLogger(__name__)


def load_environment() -> None:
    """Load environment variables from .env when available."""
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv()
        logger.debug("Loaded environment variables from .env")
    except Exception:
        # dotenv is optional; environment may already be provided externally.
        logger.debug("python-dotenv unavailable; relying on existing environment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CounselorAI CLI")

    parser.add_argument("--rank", type=int, help="Student rank (positive integer)")
    parser.add_argument("--category", type=str, help="Category, e.g. GEN/OBC/SC/ST")
    parser.add_argument("--branch", type=str, help="Preferred branch, e.g. CSE")
    parser.add_argument("--state", type=str, help="State, e.g. UP")
    parser.add_argument("--user-id", type=str, default="default_user", help="Memory user identifier")

    parser.add_argument(
        "--index-path",
        type=str,
        default="data/cutoff.index",
        help="Path to local FAISS index",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="data/cutoff_metadata.json",
        help="Path to metadata JSON used with FAISS",
    )
    parser.add_argument(
        "--retrieval-top-k",
        type=int,
        default=30,
        help="Retriever candidate count before filtering",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def collect_student_input(args: argparse.Namespace) -> Dict[str, Any]:
    """Collect required profile fields from CLI flags, prompting if missing."""

    def ask_if_missing(value: Any, prompt: str) -> str:
        if value is not None and str(value).strip():
            return str(value).strip()
        return input(prompt).strip()

    rank_str = ask_if_missing(args.rank, "Enter rank: ")
    category = ask_if_missing(args.category, "Enter category (GEN/OBC/SC/ST): ").upper()
    branch = ask_if_missing(args.branch, "Enter preferred branch: ").upper()
    state = ask_if_missing(args.state, "Enter state: ").upper()

    return {
        "rank": int(rank_str),
        "category": category,
        "branch": branch,
        "state": state,
    }


def build_planner(args: argparse.Namespace) -> PlannerAgent:
    retriever = CollegeRetriever(
        index_path=Path(args.index_path),
        metadata_path=Path(args.metadata_path),
    )
    return PlannerAgent(
        retriever=retriever,
        retrieval_top_k=args.retrieval_top_k,
        recommendation_count=5,
    )


def print_results(payload: Dict[str, Any]) -> None:
    colleges = payload.get("top_5_colleges", [])
    if not colleges:
        print("No eligible colleges found for the provided constraints.")
        return

    print("\nTop 5 Recommended Colleges\n")
    for idx, college in enumerate(colleges, start=1):
        print(f"{idx}. {college.get('college_name', 'N/A')} ({college.get('branch', 'N/A')}, {college.get('state', 'N/A')})")
        print(f"   Confidence: {college.get('confidence', 0.0)}")
        print(f"   Explanation: {college.get('explanation', '')}")
        print()

    print(f"Overall Confidence: {payload.get('overall_confidence', 0.0)}")


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    load_environment()

    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise EnvironmentError("OPENAI_API_KEY is not set.")

        student_input = collect_student_input(args)
        planner = build_planner(args)
        result = planner.run(student_input=student_input, user_id=args.user_id)
        print_results(result)
        return 0
    except ValueError as exc:
        logger.error("Invalid input: %s", exc)
        return 2
    except FileNotFoundError as exc:
        logger.error("Required file missing: %s", exc)
        logger.error("Ensure FAISS index and metadata exist. Build them via rag/embed.py first.")
        return 3
    except EnvironmentError as exc:
        logger.error("Environment configuration error: %s", exc)
        return 4
    except Exception as exc:
        logger.exception("Unexpected runtime failure: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())

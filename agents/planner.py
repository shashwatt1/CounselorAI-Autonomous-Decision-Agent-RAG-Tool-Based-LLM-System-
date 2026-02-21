from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List

from langchain_core.runnables import RunnableLambda

from agents.memory import get_preferences, save_context, save_preferences
from agents.tools import (
    explain_recommendation,
    filter_by_category,
    filter_by_rank,
    rank_colleges_by_proximity,
)
from rag.retriever import CollegeRetriever


@dataclass
class StudentProfile:
    rank: int
    category: str
    branch: str
    state: str


class PlannerAgent:
    """Planner orchestration layer for retrieval + tools + reasoning."""

    def __init__(
        self,
        retriever: CollegeRetriever,
        retrieval_top_k: int = 30,
        recommendation_count: int = 5,
    ) -> None:
        self.retriever = retriever
        self.retrieval_top_k = retrieval_top_k
        self.recommendation_count = recommendation_count
        self.steps = [
            "validate_input",
            "retrieve_candidates",
            "filter_by_category",
            "filter_by_rank",
            "rank_colleges_by_proximity",
            "explain_recommendation",
        ]

        self._pipeline = (
            RunnableLambda(self._retrieve_candidates)
            | RunnableLambda(self._apply_category_filter)
            | RunnableLambda(self._apply_rank_filter)
            | RunnableLambda(self._rank_candidates)
            | RunnableLambda(self._attach_explanations)
        )

    def run(self, student_input: Dict[str, Any], user_id: str = "default_user") -> Dict[str, Any]:
        profile = self._build_profile(student_input, user_id=user_id)
        save_preferences(
            user_id,
            {
                "category": profile.category,
                "branch": profile.branch,
                "state": profile.state,
            },
        )

        pipeline_input = {
            "profile": profile,
            "user_id": user_id,
            "retrieval_top_k": self.retrieval_top_k,
            "recommendation_count": self.recommendation_count,
        }
        pipeline_output = self._pipeline.invoke(pipeline_input)

        top_colleges = pipeline_output.get("top_colleges", [])
        overall_confidence = _average_confidence(top_colleges)

        final_output = {
            "input_used": {
                "rank": profile.rank,
                "category": profile.category,
                "branch": profile.branch,
                "state": profile.state,
            },
            "steps_executed": self.steps,
            "top_5_colleges": top_colleges[:5],
            "overall_confidence": overall_confidence,
        }

        save_context(
            user_id=user_id,
            user_input=json.dumps(student_input),
            agent_output=json.dumps(final_output),
        )
        return final_output

    def _build_profile(self, student_input: Dict[str, Any], user_id: str) -> StudentProfile:
        preferences = get_preferences(user_id)
        merged = {**preferences, **student_input}

        branch_value = (
            merged.get("branch")
            or merged.get("branch_preference")
            or merged.get("preferred_branch")
            or "CSE"
        )

        rank = self._validate_rank(merged.get("rank"))
        category = str(merged.get("category", "GEN")).upper().strip()
        branch = str(branch_value).upper().strip()
        state = str(merged.get("state", "UP")).upper().strip()

        return StudentProfile(rank=rank, category=category, branch=branch, state=state)

    def _retrieve_candidates(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        profile: StudentProfile = payload["profile"]
        retrieval = self.retriever.retrieve(
            rank=profile.rank,
            category=profile.category,
            branch_preference=profile.branch,
            state=profile.state,
            top_k=payload["retrieval_top_k"],
        )
        payload["retrieval"] = retrieval
        payload["candidates"] = retrieval.get("results", [])
        return payload

    def _apply_category_filter(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        profile: StudentProfile = payload["profile"]
        filtered = filter_by_category(payload.get("candidates", []), profile.category)
        payload["category_filter"] = filtered
        payload["candidates"] = filtered.get("colleges", [])
        return payload

    def _apply_rank_filter(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        profile: StudentProfile = payload["profile"]
        filtered = filter_by_rank(payload.get("candidates", []), profile.rank)
        payload["rank_filter"] = filtered
        payload["candidates"] = filtered.get("colleges", [])
        return payload

    def _rank_candidates(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        profile: StudentProfile = payload["profile"]
        ranked = rank_colleges_by_proximity(payload.get("candidates", []), profile.rank)
        payload["ranking"] = ranked
        payload["candidates"] = ranked.get("colleges", [])[: payload["recommendation_count"]]
        return payload

    def _attach_explanations(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        profile: StudentProfile = payload["profile"]
        student_profile = {
            "rank": profile.rank,
            "category": profile.category,
            "branch_preference": profile.branch,
            "state": profile.state,
        }
        explained: List[Dict[str, Any]] = []

        for college in payload.get("candidates", []):
            reasoning = explain_recommendation(college_data=college, student_profile=student_profile)
            explained.append(
                {
                    "college_name": college.get("college_name"),
                    "branch": college.get("branch"),
                    "category": college.get("category"),
                    "state": college.get("state"),
                    "closing_rank": college.get("closing_rank"),
                    "rank_margin": college.get("rank_margin"),
                    "relevance_score": college.get("relevance_score"),
                    "explanation": reasoning["output"]["explanation"],
                    "confidence": reasoning["output"]["confidence"],
                    "eligibility_reason": reasoning["output"]["eligibility_reason"],
                    "caution": reasoning["output"]["caution"],
                }
            )

        payload["top_colleges"] = explained
        return payload

    @staticmethod
    def _validate_rank(rank_value: Any) -> int:
        if rank_value is None:
            raise ValueError("Rank is required.")

        rank = int(rank_value)
        if rank <= 0:
            raise ValueError("Rank must be a positive integer.")
        return rank


def _average_confidence(colleges: List[Dict[str, Any]]) -> float:
    if not colleges:
        return 0.0
    confidences = [float(c.get("confidence", 0.0)) for c in colleges]
    return round(sum(confidences) / len(confidences), 3)

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence

from openai import OpenAI


def filter_by_rank(colleges: Sequence[Dict[str, Any]], student_rank: int) -> Dict[str, Any]:
    """Return colleges where student rank is within closing rank."""
    rank = int(student_rank)
    filtered: List[Dict[str, Any]] = []

    for college in colleges:
        closing_rank = _extract_closing_rank(college)
        if closing_rank is None:
            continue
        if rank <= closing_rank:
            item = dict(college)
            item["closing_rank"] = closing_rank
            item["rank_margin"] = closing_rank - rank
            filtered.append(item)

    return {
        "tool": "filter_by_rank",
        "input": {"student_rank": rank, "total_colleges": len(colleges)},
        "output_count": len(filtered),
        "colleges": filtered,
    }


def filter_by_category(colleges: Sequence[Dict[str, Any]], category: str) -> Dict[str, Any]:
    """Normalize category-specific cutoff onto each college record.

    Preferred inputs per college:
    - category + closing_rank (already long format), or
    - cutoff_<category> fields (wide format, e.g. cutoff_obc).
    """
    cat = str(category).strip().upper()
    normalized: List[Dict[str, Any]] = []

    for college in colleges:
        item = dict(college)
        existing_category = str(item.get("category", "")).upper()

        if existing_category and existing_category != cat:
            continue

        closing_rank = _closing_rank_for_category(item, cat)
        if closing_rank is None:
            continue

        item["category"] = cat
        item["closing_rank"] = closing_rank
        normalized.append(item)

    return {
        "tool": "filter_by_category",
        "input": {"category": cat, "total_colleges": len(colleges)},
        "output_count": len(normalized),
        "colleges": normalized,
    }


def rank_colleges_by_proximity(colleges: Sequence[Dict[str, Any]], student_rank: int) -> Dict[str, Any]:
    """Rank by minimum positive gap between closing rank and student rank."""
    rank = int(student_rank)
    ranked: List[Dict[str, Any]] = []

    for college in colleges:
        closing_rank = _extract_closing_rank(college)
        if closing_rank is None:
            continue

        item = dict(college)
        item["closing_rank"] = closing_rank
        item["rank_margin"] = closing_rank - rank
        item["proximity_score"] = abs(closing_rank - rank)
        item["eligible_for_rank"] = rank <= closing_rank
        ranked.append(item)

    ranked.sort(
        key=lambda c: (
            not c["eligible_for_rank"],
            c["proximity_score"],
            -int(c["closing_rank"]),
        )
    )

    return {
        "tool": "rank_colleges_by_proximity",
        "input": {"student_rank": rank, "total_colleges": len(colleges)},
        "output_count": len(ranked),
        "colleges": ranked,
    }


def explain_recommendation(college_data: Dict[str, Any], student_profile: Dict[str, Any]) -> Dict[str, Any]:
    """Use OpenAI ChatCompletion API to generate concise recommendation rationale."""
    required_profile_keys = ("rank", "category", "branch_preference", "state")
    missing = [key for key in required_profile_keys if key not in student_profile]
    if missing:
        raise ValueError(f"student_profile missing required keys: {missing}")

    client = _get_openai_client()
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    prompt_payload = {
        "student_profile": {
            "rank": int(student_profile["rank"]),
            "category": str(student_profile["category"]).upper(),
            "branch_preference": str(student_profile["branch_preference"]).upper(),
            "state": str(student_profile["state"]).upper(),
        },
        "college_data": college_data,
        "instruction": (
            "Explain eligibility and fit in 3-5 sentences. Mention category cutoff logic, "
            "rank margin, and one caution. End with a confidence (0.0-1.0). Return strict JSON "
            "with keys: explanation, confidence, eligibility_reason, caution."
        ),
    }

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a college counseling decision-support assistant.",
            },
            {
                "role": "user",
                "content": json.dumps(prompt_payload),
            },
        ],
    )

    content = response.choices[0].message.content or "{}"
    parsed = json.loads(content)

    return {
        "tool": "explain_recommendation",
        "input": {
            "college_name": college_data.get("college_name"),
            "student_profile": prompt_payload["student_profile"],
        },
        "output": {
            "explanation": str(parsed.get("explanation", "")),
            "confidence": float(parsed.get("confidence", 0.0)),
            "eligibility_reason": str(parsed.get("eligibility_reason", "")),
            "caution": str(parsed.get("caution", "")),
        },
        "model_used": model,
    }


def _closing_rank_for_category(college: Dict[str, Any], category: str) -> int | None:
    cat_key = category.lower()
    candidate_keys = [
        f"cutoff_{cat_key}",
        f"closing_rank_{cat_key}",
        "closing_rank",
        "cutoff_open",
    ]
    for key in candidate_keys:
        if key in college and college[key] is not None:
            try:
                return int(college[key])
            except (TypeError, ValueError):
                continue
    return None


def _extract_closing_rank(college: Dict[str, Any]) -> int | None:
    for key in ("closing_rank", "active_cutoff", "cutoff_rank"):
        if key in college and college[key] is not None:
            try:
                return int(college[key])
            except (TypeError, ValueError):
                continue
    for key, value in college.items():
        if key.startswith("cutoff_") and value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return None


def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key)

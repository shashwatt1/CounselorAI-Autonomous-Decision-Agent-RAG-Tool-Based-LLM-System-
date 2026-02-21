# CounselorAI - Autonomous College Counseling Agent

CounselorAI is an agentic AI system that autonomously retrieves, filters, and ranks engineering colleges based on student constraints (rank, category, branch, state), then explains recommendations with confidence.

## Highlights
- Multi-step planner-driven reasoning
- Retrieval + tool-based filtering/ranking
- Memory-aware preference reuse across sessions
- Explainable recommendations with confidence scores

## Architecture
User Input -> Planner Agent -> Retriever -> Tools (Filter, Rank, Explain) -> Final Recommendations

## Project Structure
```text
counselor-agent/
├── agents/
│   ├── planner.py
│   ├── tools.py
│   └── memory.py
├── rag/
│   ├── embed.py
│   └── retriever.py
├── data/
│   └── cutoff_sample.csv
├── main.py
├── requirements.txt
└── README.md
```

## How It Works
1. Planner validates and normalizes constraints.
2. Retriever fetches branch/state-relevant rows from cutoff data.
3. Tools apply category cutoff, eligibility checks, ranking, and explanation.
4. Memory stores user preferences for future queries.

## Quickstart
```bash
cd counselor-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

## Sample Input
```python
{
  "rank": 12000,
  "category": "OBC",
  "preferred_branch": "CSE",
  "state": "UP"
}
```

## Sample Output
- Top eligible colleges
- Justification based on cutoff margins
- Confidence score per recommendation

## Next Improvements
- Replace lightweight retriever with FAISS/Chroma semantic search
- Add FastAPI/Streamlit frontend
- Add evaluator agent for recommendation quality checks
- Integrate live cutoff refresh pipeline

# 🧠 CounselorAI – Autonomous Agentic RAG System

CounselorAI is a modular Agentic AI system designed to perform structured, constraint-driven decision making using Retrieval-Augmented Generation (RAG) and tool-based orchestration.

Unlike traditional chatbot implementations, this system demonstrates autonomous multi-step reasoning, dynamic tool invocation, and explainable recommendation generation for engineering college counseling scenarios.

---

## 🚀 Overview

Engineering counseling involves complex constraints:

- Rank-based eligibility
- Category-based cutoff rules
- Branch preferences
- State-level quotas
- Historical cutoff trends

CounselorAI acts as an autonomous decision agent that:

1. Parses structured student input
2. Retrieves relevant historical cutoff data using vector search (FAISS)
3. Applies eligibility constraints via tool functions
4. Ranks viable colleges based on proximity logic
5. Generates LLM-powered explanations
6. Returns structured output with confidence scoring

---

## 🏗️ System Architecture

User Input  
→ Planner Agent  
→ Retrieval Layer (FAISS Vector DB)  
→ Tool Invocation Layer  
→ LLM Reasoning Engine  
→ Structured Output + Confidence Score  

The architecture maintains strict modular separation between:

- Agent planning logic  
- Retrieval pipeline  
- Tool execution layer  
- Memory handling  
- Evaluation framework  

---

## 🧠 Core Capabilities

### 1️⃣ Planner-Driven Orchestration
Coordinates retrieval, filtering, ranking, and reasoning in a multi-step workflow.

### 2️⃣ Retrieval-Augmented Generation (RAG)
Uses OpenAI embeddings + FAISS for semantic retrieval of relevant cutoff entries.

### 3️⃣ Tool-Based Decision Layer
Implements structured functions:
- Rank-based filtering
- Category eligibility checks
- Proximity-based ranking
- Structured recommendation explanation

### 4️⃣ Conversational Memory
Maintains contextual user preferences across interactions.

### 5️⃣ Evaluation Pipeline
Tracks:
- Retrieval relevance
- Constraint adherence accuracy
- Ranking consistency
- Response latency

---

## 🛠️ Tech Stack

- Python  
- LangChain  
- OpenAI API  
- FAISS (Vector Database)  
- Pandas / NumPy  
- dotenv  

---

## 📂 Project Structure

```
counselor-agent/
│
├── agents/
│   ├── planner.py
│   ├── tools.py
│   ├── memory.py
│
├── rag/
│   ├── embed.py
│   ├── retriever.py
│
├── data/
│   ├── cutoff_sample.csv
│
├── main.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## 🎯 Why This Project

This project demonstrates practical Agentic AI system design beyond simple prompt chaining.

It highlights:

- Autonomous planning
- Tool invocation architecture
- Data-grounded reasoning
- Structured decision intelligence
- Engineering-focused LLM integration

Built as a portfolio demonstration of production-style Agentic AI design principles.


## 🏗 Agentic RAG Architecture

```mermaid
flowchart TD

    %% Input Layer
    A["User Query"] --> B["Planner Agent\n(LangChain Orchestration)"]

    %% Agent Layer
    subgraph Agent Layer
        B
        C["Tool Invocation Layer\n(Filter • Ranking • Memory)"]
    end

    %% Retrieval Layer
    subgraph Retrieval Layer
        D["FAISS Vector Store\n(Embeddings + Semantic Search)"]
    end

    %% Reasoning Layer
    subgraph Reasoning Engine
        E["LLM Reasoning Engine\n(OpenAI API)"]
    end

    %% Output Layer
    F["Structured Output\n+ Confidence Score"]

    %% Connections
    B --> C
    B --> D
    C --> E
    D --> E
    E --> F

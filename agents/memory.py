from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from langchain.memory import ConversationBufferMemory


class AgentMemoryStore:
    """Conversation + preference memory, isolated from planner/tool logic."""

    def __init__(self, preference_path: str = "memory_preferences.json") -> None:
        self.preference_path = Path(preference_path)
        self._preferences: Dict[str, Dict[str, Any]] = {}
        self._conversations: Dict[str, ConversationBufferMemory] = {}
        self._load_preferences()

    def get_memory(self, user_id: str = "default_user") -> ConversationBufferMemory:
        if user_id not in self._conversations:
            self._conversations[user_id] = ConversationBufferMemory(
                memory_key="chat_history",
                input_key="input",
                output_key="output",
                return_messages=True,
            )
        return self._conversations[user_id]

    def save_context(self, user_id: str, user_input: str, agent_output: str) -> None:
        memory = self.get_memory(user_id)
        memory.save_context({"input": user_input}, {"output": agent_output})

    def get_preferences(self, user_id: str = "default_user") -> Dict[str, Any]:
        return dict(self._preferences.get(user_id, {}))

    def save_preferences(self, user_id: str, payload: Dict[str, Any]) -> None:
        existing = self._preferences.get(user_id, {})
        existing.update(payload)
        self._preferences[user_id] = existing
        self._persist_preferences()

    def _load_preferences(self) -> None:
        if not self.preference_path.exists():
            return
        raw = json.loads(self.preference_path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            self._preferences = raw

    def _persist_preferences(self) -> None:
        self.preference_path.write_text(
            json.dumps(self._preferences, indent=2),
            encoding="utf-8",
        )


_MEMORY_STORE = AgentMemoryStore()


def get_memory(user_id: str = "default_user") -> ConversationBufferMemory:
    """Return a per-user LangChain ConversationBufferMemory instance."""
    return _MEMORY_STORE.get_memory(user_id)


def save_context(user_id: str, user_input: str, agent_output: str) -> None:
    """Persist one conversational turn into ConversationBufferMemory."""
    _MEMORY_STORE.save_context(user_id=user_id, user_input=user_input, agent_output=agent_output)


def get_preferences(user_id: str = "default_user") -> Dict[str, Any]:
    """Fetch stored user preferences as structured data."""
    return _MEMORY_STORE.get_preferences(user_id)


def save_preferences(user_id: str, payload: Dict[str, Any]) -> None:
    """Store/merge user preference fields for future sessions."""
    _MEMORY_STORE.save_preferences(user_id, payload)

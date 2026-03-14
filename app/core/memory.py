"""
Conversation memory management using in-memory store.
Optionally backed by Redis for production.
"""
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
import structlog

from app.models.schemas import ConversationMessage, ConversationHistory

logger = structlog.get_logger()


class ConversationMemoryStore:
    """In-memory conversation store with optional Redis backend."""

    def __init__(self):
        self._store: Dict[str, ConversationHistory] = {}

    def create_conversation(self) -> str:
        """Create a new conversation and return its ID."""
        conv_id = str(uuid.uuid4())
        self._store[conv_id] = ConversationHistory(conversation_id=conv_id)
        return conv_id

    def get_conversation(self, conversation_id: str) -> Optional[ConversationHistory]:
        return self._store.get(conversation_id)

    def add_message(
        self, conversation_id: str, role: str, content: str
    ) -> ConversationHistory:
        """Add a message to an existing conversation."""
        if conversation_id not in self._store:
            self._store[conversation_id] = ConversationHistory(
                conversation_id=conversation_id
            )
        conv = self._store[conversation_id]
        conv.messages.append(
            ConversationMessage(role=role, content=content)
        )
        # Keep last 20 messages to avoid context overflow
        if len(conv.messages) > 20:
            conv.messages = conv.messages[-20:]
        return conv

    def get_history_as_string(self, conversation_id: str, max_turns: int = 5) -> str:
        """Format recent history as string for prompt injection."""
        conv = self.get_conversation(conversation_id)
        if not conv or not conv.messages:
            return "No previous conversation."

        recent = conv.messages[-(max_turns * 2):]
        lines = []
        for msg in recent:
            role = "Patient" if msg.role == "user" else "Assistant"
            lines.append(f"{role}: {msg.content}")
        return "\n".join(lines)

    def get_history_as_list(
        self, conversation_id: str, max_turns: int = 5
    ) -> List[Dict[str, str]]:
        """Return history as list of {role, content} dicts."""
        conv = self.get_conversation(conversation_id)
        if not conv:
            return []
        recent = conv.messages[-(max_turns * 2):]
        return [{"role": m.role, "content": m.content} for m in recent]

    def delete_conversation(self, conversation_id: str) -> bool:
        if conversation_id in self._store:
            del self._store[conversation_id]
            return True
        return False

    def list_conversations(self) -> List[str]:
        return list(self._store.keys())


memory_store = ConversationMemoryStore()
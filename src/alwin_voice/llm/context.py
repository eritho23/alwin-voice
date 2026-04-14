from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


class ConversationContext:
    def __init__(self, max_turns: int = 12) -> None:
        self._messages: deque[ChatMessage] = deque(maxlen=max_turns * 2)

    def add_user(self, content: str) -> None:
        self._messages.append(ChatMessage(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        self._messages.append(ChatMessage(role="assistant", content=content))

    def as_ollama_messages(self, system_prompt: str) -> list[dict[str, str]]:
        out: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for msg in self._messages:
            out.append({"role": msg.role, "content": msg.content})
        return out

from typing import Any
from langchain_core.messages import AIMessage, HumanMessage


def dict_to_aimessage(d: dict[str, Any]) -> AIMessage:
    # Accepts either {"content": "...", "type":"assistant"} or {"content": "..."}
    content = d.get("content") if isinstance(d, dict) else str(d)
    return AIMessage(content=content)

def format_conversation(messages: list[Any]) -> str:
        conversation = "Conversation history:\n\n"
        for message in messages:
            if isinstance(message, HumanMessage):
                conversation += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                text = message.content or "[Tools use]"
                conversation += f"Assistant: {text}\n"
        return conversation

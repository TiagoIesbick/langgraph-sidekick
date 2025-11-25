from typing import Any
from langchain_core.messages import AIMessage, HumanMessage


CAPABILITIES_MANIFEST = {
    "researcher": {
        "description": "Finds information online, navigates the web, extracts structured content.",
        "tools": [
            "search", "wikipedia", "click", "navigate",
            "navigate_back", "extract_text", "extract_hyperlinks",
            "get_elements", "current_web_page"
        ]
    },
    "executor": {
        "description": "Runs code, handles files, performs external actions.",
        "tools": [
            "python_repl", "copy_file", "delete_file",
            "file_search", "move_file", "read_file", "write_file",
            "list_directory", "send_whatsapp"
        ]
    },
    "summarizer": {
        "description": "Summarizes text, rewrites content, produces narratives.",
        "tools": []
    },
    "evaluator": {
        "description": "Evaluates work against success criteria, detects issues.",
        "tools": []
    }
}

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

from typing import Any, Optional
from langchain_core.messages import AIMessage, HumanMessage
from enum import Enum
from schema import ToolInference


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


class ToolSafety(str, Enum):
    READ_ONLY = "read_only"
    IRREVERSIBLE = "irreversible"
    SANDBOXED_COMPUTE = "sandboxed_compute"


EXECUTOR_TOOL_SAFETY = {
    # Safe
    "read_file": ToolSafety.READ_ONLY,
    "file_search": ToolSafety.READ_ONLY,
    "list_directory": ToolSafety.READ_ONLY,

    # Sandboxed but potentially harmful
    "python_repl": ToolSafety.SANDBOXED_COMPUTE,

    # Irreversible
    "write_file": ToolSafety.IRREVERSIBLE,
    "delete_file": ToolSafety.IRREVERSIBLE,
    "move_file": ToolSafety.IRREVERSIBLE,
    "copy_file": ToolSafety.IRREVERSIBLE,
    "send_whatsapp": ToolSafety.IRREVERSIBLE,
}

def dict_to_aimessage(d: dict[str, Any]) -> AIMessage:
    # Accepts either {"content": "...", "type":"assistant"} or {"content": "..."}
    content = d.get("content") if isinstance(d, dict) else str(d)
    return AIMessage(content=content)

def format_conversation(messages: list[Any]) -> str:
        conversation = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                conversation += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                text = message.content or "[Tools use]"
                conversation += f"Assistant: {text}\n"
        return conversation

def infer_tool_name(message: AIMessage) -> Optional[ToolInference]:
    """
    Extracts the tool name from a structured AIMessage.
    Returns None if no tool call exists.
    Raises if the message is malformed.
    """

    if not isinstance(message, AIMessage):
        return None

    tool_calls = getattr(message, "tool_calls", None)

    if not tool_calls:
        return None

    if len(tool_calls) != 1:
        raise RuntimeError(
            f"Expected exactly one tool call, got {len(tool_calls)}"
        )

    tool_call = tool_calls[0]

    if "name" not in tool_call:
        raise RuntimeError("Malformed tool call: missing 'name'")

    return ToolInference(
        tool_name=tool_call["name"],
        tool_call_id=tool_call["id"],
    )

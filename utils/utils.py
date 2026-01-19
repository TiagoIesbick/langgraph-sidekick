from typing import Any, Optional
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from enum import Enum
from schema import ToolInference


CAPABILITIES_MANIFEST = {
    "researcher": {
        "description": "Finds information online, navigates the web, extracts structured content.",
        "tools": [
            "search", "wikipedia", "click_element", "navigate_browser",
            "previous_webpage", "extract_text", "extract_hyperlinks",
            "get_elements", "current_webpage"
        ]
    },
    "executor": {
        "description": "Runs code, handles files, performs external actions.",
        "tools": [
            "Python_REPL", "copy_file", "file_delete",
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
    "Python_REPL": ToolSafety.SANDBOXED_COMPUTE,

    # Irreversible
    "write_file": ToolSafety.IRREVERSIBLE,
    "file_delete": ToolSafety.IRREVERSIBLE,
    "move_file": ToolSafety.IRREVERSIBLE,
    "copy_file": ToolSafety.IRREVERSIBLE,
    "send_whatsapp": ToolSafety.IRREVERSIBLE,
}

def dict_to_aimessage(d: dict[str, Any]) -> AIMessage:
    # Accepts either {"content": "...", "type":"assistant"} or {"content": "..."}
    content = d.get("content") if isinstance(d, dict) else str(d)
    return AIMessage(content=content)

def truncate(text: str, max_len=500):
    return text if len(text) <= max_len else text[:max_len] + "…"

def format_conversation(messages: list[Any]) -> str:
        conversation = ""
        for message in messages:
            if isinstance(message, HumanMessage):
                conversation += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                text = message.content or "[Requested tool execution]"
                conversation += f"Assistant: {text}\n"
            elif isinstance(message, ToolMessage):
                tool_name = message.name or "unknown_tool"
                tool_output = truncate(message.content or "[No output]")
                conversation += f"Tool ({tool_name}) output: {tool_output}\n"
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

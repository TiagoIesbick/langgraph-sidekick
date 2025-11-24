from schema import State, PlannerOutput, PlannerStateDiff
from utils.utils import dict_to_aimessage, format_conversation
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.language_models import LanguageModelInput
from langchain_openai.chat_models.base import _DictOrPydantic
from datetime import datetime
from typing import Any
import json


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


def planner_agent(
    llm_with_output: Runnable[LanguageModelInput, _DictOrPydantic],
    state: State
) -> dict:
    system_msg = f"""
Role:
You are the PLANNER.

Task:
1. Read the full conversation.
2. Generate:
    - A global plan (string)
    - A list of subtasks (atomic, sequential)
    - Success criteria (string)

Capabilities Manifest (required for assigning subtasks):
{json.dumps(CAPABILITIES_MANIFEST, indent=2)}

Subtask Semantics:
- "task": a minimal actionable instruction.
- "assigned_to": which agent should execute it:
    - researcher
    - executor
    - summarizer
    - evaluator

Output Requirements:
- You MUST output ONLY JSON.
- Structure your output EXACTLY like this:

{{
    "state_diff": {{
        "plan": "...",
        "subtasks": [
            {{ "task": "...", "assigned_to": "researcher" }},
            {{ "task": "...", "assigned_to": "summarizer" }}
        ],
        "success_criteria": "...",
        "messages": [ {{ "type": "assistant", "content": "..." }} ]
    }}
}}

Rules:
- You MUST produce a plan.
- Subtasks must be sequential, actionable, and minimal.
- You MUST generate success_criteria.
- Do NOT modify any other State fields.
- Current date/time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    human_msg = f"""
Conversation so far:
{format_conversation(state.messages)}

Generate the plan, subtasks, and success criteria.
"""


    llm_response: PlannerOutput = llm_with_output.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=human_msg)
    ])


    diff: PlannerStateDiff = llm_response.state_diff


    updates: dict[str, Any] = {
        "plan": diff.plan,
        "subtasks": diff.subtasks,
        "success_criteria": diff.success_criteria
    }

    if diff.messages:
        updates["messages"] = [dict_to_aimessage(m) for m in diff.messages]

    return updates

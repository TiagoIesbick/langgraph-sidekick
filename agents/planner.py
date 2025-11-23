from schema import State, PlannerOutput
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.language_models import LanguageModelInput
from langchain_openai.chat_models.base import _DictOrPydantic
from datetime import datetime
from typing import Any, Callable


def planner_agent(
    llm_with_output: Runnable[LanguageModelInput, _DictOrPydantic],
    format_conversation: Callable[[list[Any]], str],
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

Output Requirements:
- You MUST output ONLY JSON.
- Structure your output EXACTLY like this:

{{
    "state_diff": {{
        "plan": "...",
        "subtasks": ["...", "..."],
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


    resp: PlannerOutput = llm.invoke([
    SystemMessage(content=system_msg),
    HumanMessage(content=human_msg)
    ])


    diff = resp.state_diff


    updates: Dict[str, Any] = {
    "plan": diff.plan,
    "subtasks": diff.subtasks,
    }


    if diff.success_criteria:
    updates["success_criteria"] = diff.success_criteria


    if diff.messages:
    updates["messages"] = [_to_ai(m) for m in diff.messages]


    return state.model_copy(update=updates)

from schema import State
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from datetime import datetime
from langchain_core.runnables import Runnable
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from utils.utils import CAPABILITIES_MANIFEST


def researcher_agent(
    llm_with_tools: Runnable[LanguageModelInput, BaseMessage],
    state: State
) -> dict:

    if not state.subtasks:
        raise RuntimeError("Researcher invoked with no subtasks")

    if state.next_subtask_index >= len(state.subtasks):
        raise RuntimeError("Researcher invoked with invalid task index")

    current = state.subtasks[state.next_subtask_index]

    if current.assigned_to != "researcher":
        raise RuntimeError(
            f"Researcher invoked for task assigned to {current.assigned_to}"
        )

    system_msg = f"""
Role:
You are the RESEARCHER agent in a LangGraph-based multi-agent system.

Tools:
{CAPABILITIES_MANIFEST.get("researcher").get("tools")}

CRITICAL TOOL USAGE RULE:
- You may call AT MOST ONE tool in a single response.
- If multiple pieces of information are needed, choose the single most
  appropriate tool and extract all required data with it.
- Do NOT call multiple tools in the same turn.

Task:
- Use your tools to fulfill the user's request.

Rules:
- Use tools as needed.
- When you are done, produce a concise summary.
- Do NOT call tools after your final summary.
- The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.
"""

    human_msg = f"""
Current task:
{current.task}
"""

    if state.subtask_results:
        human_msg += f"""
Results (from previous agents):
{chr(10).join(f"- {r}" for r in state.subtask_results)}
"""

    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=human_msg)
    ]

    for msg in state.messages:
        if isinstance(msg, (AIMessage, ToolMessage)):
            messages.append(msg)

    llm_response = llm_with_tools.invoke(messages)

    if llm_response.tool_calls:
        return {
            "messages": [llm_response],
        }

    return {
        "subtask_results": state.subtask_results + [llm_response.content],
        "messages": [AIMessage(content=f"Research completed for task: {current.task}")],
        "next_subtask_index": state.next_subtask_index + 1
    }

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

    if state.current_subtask_index >= len(state.subtasks):
        raise RuntimeError("Researcher invoked with invalid task index")

    current = state.subtasks[state.current_subtask_index]

    if current.assigned_to != "researcher":
        raise RuntimeError(
            f"Researcher invoked for task assigned to {current.assigned_to}"
        )

    system_msg = f"""
Role:
You are the Researcher Agent.

Tools:
{CAPABILITIES_MANIFEST.get("researcher").get("tools")}

Task:
- Use your tools to fulfill the user's request.

Rules:
- From the tools results, produce a concise summary of the results.
- Capture the main points.
- This will be consumed by someone synthesizing a report, so its vital you capture the essence and ignore any fluff.
- Do not include any additional commentary other than the summary itself.
- The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.
"""

    human_msg = f"Task:\n{current.task}"

    last_message = state.messages[-1]

    if isinstance(last_message, ToolMessage):
        return {
            "subtask_results": state.subtask_results + [last_message.content],
            "messages": [
                AIMessage(content=f"Research completed for task: {current.task}")
            ],
            "current_subtask_index": state.current_subtask_index + 1
        }

    llm_response = llm_with_tools.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=human_msg)
    ])

    print('[researcher]:', llm_response)

    if llm_response.tool_calls:
        return {
            "messages": [llm_response],
        }

    return {
        "subtask_results": state.subtask_results + [llm_response.content],
        "messages": [AIMessage(content=f"Research completed for task: {current.task}")],
        "current_subtask_index": state.current_subtask_index + 1
    }

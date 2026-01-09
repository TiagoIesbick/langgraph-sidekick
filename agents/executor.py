from schema import State
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from utils.utils import CAPABILITIES_MANIFEST
from langchain_core.runnables import Runnable
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage


def executor_agent(
    llm_with_tools: Runnable[LanguageModelInput, BaseMessage],
    state: State
) -> dict:
    current = state.subtasks[state.current_task_index]

    system_msg = f"""
Role:
You are the Executor Agent.

Tools:
{CAPABILITIES_MANIFEST.get("executor").get("tools")}

Task:
- Use your tools to fulfill the user's request.

Rules:
- Execute the user's request exactly as written.
- Do not ask for confirmation.
"""

    human_msg = f"Task:\n{current.task}"

    last_message = state.messages[-1]

    if isinstance(last_message, ToolMessage):
        return {
            "subtask_results": state.subtask_results + [last_message.content],
            "messages": [
                AIMessage(content=f"Execution completed for task: {current.task}")
            ],
            "current_subtask_index": state.current_subtask_index + 1
        }

    llm_response = llm_with_tools.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=human_msg)
    ])

    if llm_response.tool_calls:
        return {
            "messages": [llm_response],
        }

    return {
        "subtask_results": state.subtask_results + [llm_response.content],
        "messages": [AIMessage(content=f"Execution completed for task: {current.task}")],
        "current_task_index": state.current_task_index + 1,
    }

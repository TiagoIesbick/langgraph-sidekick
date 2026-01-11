from schema import State
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from utils.utils import CAPABILITIES_MANIFEST, EXECUTOR_TOOL_SAFETY, ToolSafety, infer_tool_name
from langchain_core.runnables import Runnable
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage


def executor_agent(
    llm_with_tools: Runnable[LanguageModelInput, BaseMessage],
    state: State
) -> dict:
    current = state.subtasks[state.next_subtask_index]

    system_msg = f"""
Role:
You are the EXECUTOR Agent in a LangGraph-based multi-agent system.

Tools:
{CAPABILITIES_MANIFEST.get("executor").get("tools")}

Task:
- Execute the task exactly as written.
- If the task requires irreversible side effects:
  - Request approval instead of executing.
- Only execute irreversible tools if approval is granted.
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
        tool = infer_tool_name(llm_response)
        safety = EXECUTOR_TOOL_SAFETY.get(tool.tool_name)

        needs_approval = (
            safety == ToolSafety.IRREVERSIBLE or
            (safety == ToolSafety.SANDBOXED_COMPUTE and current.requires_side_effects)
        )

        if needs_approval and not state.side_effects_approved:
            return {
                "side_effects_requested": True,
                "messages": [
                    AIMessage(
                        content=(
                            f"Requesting approval for side-effectful action "
                            f"using tool: {tool.tool_name}"
                        )
                    )
                ]
            }

        return {
            "messages": [llm_response],
        }

    return {
        "subtask_results": state.subtask_results + [llm_response.content],
        "messages": [AIMessage(content=f"Execution completed for task: {current.task}")],
        "next_subtask_index": state.next_subtask_index + 1,
        "side_effects_requested": False,
        "side_effects_approved": False
    }

from schema import State
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from utils.utils import CAPABILITIES_MANIFEST, EXECUTOR_TOOL_SAFETY, ToolSafety, infer_tool_name
from langchain_core.runnables import Runnable
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from datetime import datetime


def executor_agent(
    llm_with_tools: Runnable[LanguageModelInput, BaseMessage],
    state: State
) -> dict:
    current = state.subtasks[state.next_subtask_index]

    system_msg = f"""
Role:
You are the EXECUTOR agent in a LangGraph-based multi-agent system.

Your responsibility is to execute the assigned subtask EXACTLY as written,
using the available tools when required, and to produce a clear, verifiable
result that can be evaluated downstream.

--------------------------------------------------------------------
TOOLS
--------------------------------------------------------------------
{CAPABILITIES_MANIFEST.get("executor").get("tools")}

--------------------------------------------------------------------
CRITICAL TOOL USAGE RULES (MUST FOLLOW)
--------------------------------------------------------------------

1. SINGLE TOOL PER TURN (NON-NEGOTIABLE)
- You may call AT MOST ONE tool in a single response.
- Do NOT call multiple tools in the same turn.
- If multiple pieces of information are needed:
  - Choose the single most appropriate tool
  - Extract ALL required data using that tool alone

Violating this rule WILL cause execution failure.

2. PYTHON_REPL USAGE (CRITICAL)
If you use the Python_REPL tool:
- You MUST explicitly print the final result.
- Do NOT rely on implicit expression output.
- Ensure the printed output contains the final answer required to
  complete the task.

Example (VALID):
    print(math.pi * 3)

Example (INVALID):
    math.pi * 3

--------------------------------------------------------------------
EXECUTION & SAFETY RULES
--------------------------------------------------------------------

3. TASK EXECUTION
- Execute the current task exactly as written.
- Do NOT add steps.
- Do NOT omit steps.

4. SIDE EFFECT SAFETY
- If the task requires irreversible side effects:
  - Request user approval BEFORE executing.
- Do NOT execute irreversible tools unless approval has been granted.

5. TOOL FLOW
- If you call a tool, STOP and wait for the tool result.
- Do NOT produce a summary in the same turn as a tool call.
- Do NOT call tools after producing a final summary.

6. COMPLETION
- When the task is complete, produce a concise summary.
- The summary MUST be sufficient for evaluator verification.
- Do NOT ask questions.
- Do NOT include reasoning or analysis.

--------------------------------------------------------------------
CURRENT CONTEXT
--------------------------------------------------------------------
- Current date and time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
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

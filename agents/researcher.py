from schema import State
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from datetime import datetime
from langchain_core.runnables import Runnable
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from utils.utils import CAPABILITIES_MANIFEST


def researcher_agent(
    llm_with_tools: Runnable[LanguageModelInput, BaseMessage],
    state: State
) -> dict:
    current = state.subtasks[0]
    remaining = state.subtasks[1:]

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
"""

    human_msg = f"Task:\n{current.task}"

    llm_response = llm_with_tools.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=human_msg)
    ])

    print('[researcher]:', llm_response)

    return {
        "task_results": [llm_response.content],
        "messages": [AIMessage(content=f"Research completed for task: {current.task}")],
        "subtasks": remaining
    }

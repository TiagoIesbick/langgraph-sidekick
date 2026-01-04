from schema import State
from utils.utils import dict_to_aimessage
from langchain_core.messages import HumanMessage, SystemMessage


def summarizer_agent(llm, state: State) -> dict:
    
    print(f"[summarizer_state]: {state}")

    current = state.subtasks[0]

    

    system_msg = f"""
Role:
You are the Summarizer Agent.

Task:
{current.task}

Inputs (from previous agents):
{chr(10).join(f"- {r}" for r in state.subtask_results)}

Rules:
- From the previous agents inputs, synthetize a report.
- Capture the main points.
- Do not introduce new information.
- Do not include any additional commentary other than the report itself.
    """

    llm_response = llm.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=current.task)
    ])

    print(f"[summarizer_response]: {llm_response}")
    return {
        "messages": [dict_to_aimessage(llm_response)]
    }

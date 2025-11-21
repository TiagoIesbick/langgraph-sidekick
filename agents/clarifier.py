from schema import ClarifierOutput, State
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import Runnable
from langchain_core.language_models import LanguageModelInput
from langchain_openai.chat_models.base import _DictOrPydantic
from datetime import datetime


def clarifier_agent(
    llm_with_output: Runnable[LanguageModelInput, _DictOrPydantic],
    state: State
) -> State:

    system_message = f"""
Role:
You are a Clarification Agent.

Task:
- Evaluate whether the user's request is clear.
- If unclear, ask a clarifying question.
- If clear, acknowledge it.

Rules:
- You MUST output ONLY JSON.
- Structure the output EXACTLY like this:

{{
  "state_diff": {{
      "messages": [ {{ "type": "assistant", "content": "..." }} ],
      "user_input_needed": true or false
  }}
}}

- You may ONLY modify:
    - messages (append one assistant message)
    - user_input_needed (true/false)
- DO NOT include or modify any other State fields.
- DO NOT echo back fields not shown above.
- The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.
"""
    last_user_input = state.messages[-1].content if state.messages else "(no messages)"

    user_message = f"User request: {last_user_input}"

    llm_response: ClarifierOutput  = llm_with_output.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=user_message)
    ])

    diff = llm_response.state_diff

    updates = {}

    # Convert the raw dicts into AIMessage
    if diff.messages:
        updates["messages"] = [
            AIMessage(content=m["content"])
            for m in diff.messages
        ]

    if diff.user_input_needed is not None:
        updates["user_input_needed"] = diff.user_input_needed

    return state.model_copy(update=updates)

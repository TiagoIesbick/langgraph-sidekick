from schema import ClarifierOutput, State, ClarifierStateDiff
from utils.utils import dict_to_aimessage, format_conversation
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from langchain_core.language_models import LanguageModelInput
from langchain_openai.chat_models.base import _DictOrPydantic
from datetime import datetime


def clarifier_agent(
    llm_with_output: Runnable[LanguageModelInput, _DictOrPydantic],
    state: State
) -> dict:

    system_message = f"""
Role:
You are a Clarification Agent.

Task:
- Evaluate whether the user's request is clear based on the FULL conversation history.
- If unclear, ask a clarifying question.
- If clear, acknowledge the request clearly and do NOT ask any question.

Rules:
- "user_input_needed" MUST be:
    - true  → if you asked a clarifying question
    - false → if the user's request is already fully clear
- You may ONLY modify:
    - messages (append ONE assistant message)
    - user_input_needed (true/false)
- Do NOT modify, include, or reference any other fields.
- Do NOT repeat previous assistant questions.
- Do NOT ask for clarification if the answer is already present in prior user turns.
- The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.
"""
    last_user_input = state.messages[-1].content if state.messages else "(no messages)"

    human_message = f"""
Here is the full conversation so far:
{format_conversation(state.messages)}

The final user message is:
"{last_user_input}"

Your job:
- Decide whether the user's request is clear GIVEN THE ENTIRE CONVERSATION.
- If it is clear → set "user_input_needed": false and acknowledge.
- If it is unclear → ask ONE clarifying question and set "user_input_needed": true.
"""

    llm_response: ClarifierOutput  = llm_with_output.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=human_message)
    ])

    diff: ClarifierStateDiff = llm_response.state_diff

    updates: dict = {}

    if diff.messages:
        updates["messages"] = [dict_to_aimessage(m) for m in diff.messages]

    updates["user_input_needed"] = bool(diff.user_input_needed)

    return updates

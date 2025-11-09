from schema import State
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import Runnable
from langchain_core.language_models import LanguageModelInput
from langchain_openai.chat_models.base import _DictOrPydantic
from datetime import datetime


def clarifier_agent(llm_with_output: Runnable[LanguageModelInput, _DictOrPydantic], state: State) -> State:
    system_message = f"""
Role:
- You are a clarification agent.

Guidelines:
- Check if the user's request is clear and contains all needed info (goal, format, deadline, scope).
- If unclear, ask specific questions.
- The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Output **only valid JSON**, nothing else.
- Update only 'messages' and 'user_input_needed'.
- Set 'user_input_needed' to True if you ask a clarification question.
- The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.
"""
    user_message = f"User request: {state.messages[-1].content if state.messages else '(no messages)'}"
    response = llm_with_output.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=user_message)
    ])
    print('[response clarifier]:', response)
    clarified_state = response.dict(exclude_unset=True)
    updated_state = state.model_copy(update=clarified_state)
    return updated_state

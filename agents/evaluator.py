from schema import State, EvaluatorOutput
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.language_models import LanguageModelInput
from langchain_openai.chat_models.base import _DictOrPydantic
from utils.utils import dict_to_aimessage
from typing import Any, Callable
from datetime import datetime


def evaluator_agent(
    llm_with_output: Runnable[LanguageModelInput, _DictOrPydantic],
    state: State
) -> State:

    system_message = f"""
Role:
You are the EVALUATOR agent in a LangGraph-based multi-agent system.

Goal:
1. Decide whether the SUCCESS CRITERIA has been met based on the the task results.
2. Decide whether requested side effects are allowed

Decision Rules (STRICT):
- success_criteria_met:
  - Set to TRUE only if the success criteria is fully satisfied.
  - Set to FALSE if any required element is missing, ambiguous, or incorrect.

- user_input_needed:
  - Set to TRUE only if progress is blocked and clarification or missing information
    must be obtained from the user.
  - Set to FALSE if the system can continue autonomously.

- feedback:
  - Provide a concise, actionable explanation of your decision.
  - If success_criteria_met is FALSE, explain exactly what is missing or wrong.
  - If user_input_needed is TRUE, explicitly state what the user must provide.

Constraints:
- Do NOT suggest plans or actions.
- Do NOT invent missing data.
- Do NOT repeat the success criteria verbatim.

Current date/time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    human_msg = f"""
Success criteria:
{state.success_criteria}

Task results:
{chr(10).join(f"- {r}" for r in state.subtask_results)}

Side effects requested:
{state.side_effects_requested}
"""

    if state.feedback_on_work:
        human_msg += f"Also, note that in a prior attempt from the Assistant, you provided this feedback: {state.feedback_on_work}\n"
        human_msg += "If you're seeing the Assistant repeating the same mistakes, then consider responding that user input is required."

    llm_response: EvaluatorOutput = llm_with_output.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=human_msg)
    ])

    updates = {
        "messages": [dict_to_aimessage(llm_response.feedback)],
        "feedback_on_work": llm_response.feedback,
        "success_criteria_met": llm_response.success_criteria_met,
        "user_input_needed": llm_response.user_input_needed
    }

    if state.side_effects_requested:
        updates["side_effects_approved"] = bool(llm_response.side_effects_approved)

    return updates

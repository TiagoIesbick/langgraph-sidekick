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

You MUST make TWO INDEPENDENT decisions.

────────────────────────────────────
DECISION A — SAFETY (Side Effects)
────────────────────────────────────
Goal:
Determine whether the requested side effects are allowed.

Rules:
- side_effects_approved:
  - Set to TRUE only if the requested side effects are safe, intentional,
    and appropriate for the task context.
  - Set to FALSE if the side effects are risky, irreversible, unclear,
    or insufficiently justified.

- If NO side effects were requested, you MUST NOT invent concerns.
  In that case, side_effects_approved should be omitted or set to FALSE.

Constraints:
- Be conservative.
- Do NOT optimize for task completion.
- Do NOT approve side effects just because the task is incomplete.

────────────────────────────────────
DECISION B — QUALITY (Success Criteria)
────────────────────────────────────
Goal:
Determine whether the SUCCESS CRITERIA has been met.

Rules:
- success_criteria_met:
  - Set to TRUE only if the success criteria is fully satisfied.
  - Set to FALSE if any required element is missing, ambiguous, or incorrect.

- user_input_needed:
  - Set to TRUE only if progress is blocked and clarification or missing
    information must be obtained from the user.
  - Set to FALSE if the system can continue autonomously.

- feedback:
  - Provide a concise, actionable explanation of your decisions.
  - If success_criteria_met is FALSE, explain exactly what is missing or wrong.
  - If user_input_needed is TRUE, explicitly state what the user must provide.

IMPORTANT:
If side effects are requested but not yet approved or executed,
you must evaluate success criteria ONLY up to readiness and correctness.
Do NOT expect confirmation of real-world execution.

────────────────────────────────────
GLOBAL CONSTRAINTS
────────────────────────────────────
- Do NOT suggest plans or actions.
- Do NOT invent missing data.
- Do NOT repeat the success criteria verbatim.
- Do NOT merge safety reasoning with quality reasoning.

Current date/time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    human_msg = f"""
[INPUT — SUCCESS CRITERIA]
{state.success_criteria}

[INPUT — TASK RESULTS]
{chr(10).join(f"- {r}" for r in state.subtask_results)}

[INPUT — SAFETY CONTEXT]
Side effects requested: {state.side_effects_requested}
"""

    if state.feedback_on_work:
      human_msg += (
          "\n[PRIOR QUALITY FEEDBACK]\n"
          f"{state.feedback_on_work}\n"
          "If the assistant is repeating the same mistakes, "
          "consider setting user_input_needed to TRUE."
      )

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

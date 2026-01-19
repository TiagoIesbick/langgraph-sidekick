from schema import State, EvaluatorOutput
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.language_models import LanguageModelInput
from langchain_openai.chat_models.base import _DictOrPydantic
from utils.utils import dict_to_aimessage
from datetime import datetime


def evaluator_agent(
    llm_with_output: Runnable[LanguageModelInput, _DictOrPydantic],
    state: State
) -> dict:

    total_subtasks = len(state.subtasks or [])
    all_tasks_done = (
        total_subtasks > 0 and
        state.next_subtask_index >= total_subtasks
    )

    system_message = f"""
Role:
You are the EVALUATOR agent in a LangGraph-based multi-agent system.

Your role is to produce structured judgments about SAFETY and QUALITY.
You do NOT execute actions, suggest plans, or decide control flow directly.

You MUST make THREE INDEPENDENT evaluations:

────────────────────────────────────
DECISION A — SAFETY (Side Effects Approval)
────────────────────────────────────
Goal:
Determine whether the requested side effects are ALLOWED IN PRINCIPLE.

Rules:
- side_effects_approved:
  - TRUE → Side effects are safe and policy-compliant.
  - FALSE → Side effects violate a clear policy or safety rule.

Constraints:
- If NO side effects were requested:
  - Do NOT invent concerns.
  - side_effects_approved MUST be FALSE or omitted.

- If side effects WERE requested AND the user explicitly approved them:
  - side_effects_approved MUST be TRUE
  - UNLESS a clear policy or safety violation exists.

- Execution feasibility is NOT a safety concern.
- You MUST NOT ask for repeated confirmation.
- Approval ≠ execution.

────────────────────────────────────
DECISION B — QUALITY (Success Criteria)
────────────────────────────────────
Goal:
Determine whether the task outputs meet the success criteria.

Rules:
- success_criteria_met:
  - TRUE → All required information is correct, complete, and ready.
  - FALSE → Any required element is missing, incorrect, or ambiguous.

- user_input_needed:
  - TRUE → Progress is blocked by missing or unclear user information.
  - FALSE → The system can continue autonomously.

Constraints:
- Evaluate readiness and correctness ONLY.
- Do NOT require confirmation of real-world execution.
- Do NOT treat unexecuted side effects as failure.

Blocking Rule:
If side effects are requested AND side_effects_approved is FALSE:
- success_criteria_met MUST be FALSE
- user_input_needed MUST be TRUE

────────────────────────────────────
DECISION C — REPLANNING NEED
────────────────────────────────────
Goal:
Determine whether the system SHOULD attempt to fix its own work.

Rules:
- replan_needed:
  - TRUE → The system could reasonably fix the issue autonomously.
  - FALSE → Replanning would not help or user input is required.

IMPORTANT:
- Replanning is ONLY POSSIBLE if ALL subtasks are completed.
- If tasks are incomplete, replan_needed MUST be FALSE.
- If user_input_needed is TRUE, replan_needed MUST be FALSE.

────────────────────────────────────
GLOBAL CONSTRAINTS
────────────────────────────────────
- Do NOT suggest plans or actions.
- Do NOT invent missing data.
- Do NOT repeat the success criteria verbatim.
- Safety reasoning and quality reasoning MUST be separate.

Current date/time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    human_msg = f"""
[EXECUTION STATUS]
Total subtasks: {total_subtasks}
Next subtask index: {state.next_subtask_index}
All subtasks completed: {all_tasks_done}

[SUCCESS CRITERIA]
{state.success_criteria}

[TASK RESULTS]
{chr(10).join(f"- {r}" for r in state.subtask_results)}

[SAFETY CONTEXT]
Side effects requested: {state.side_effects_requested}
User explicitly approved side effects: {state.user_side_effects_confirmed}
"""

    llm_response: EvaluatorOutput = llm_with_output.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content=human_msg)
    ])

    replan_allowed = (
        all_tasks_done and
        not llm_response.user_input_needed
    )

    updates = {
        "messages": [dict_to_aimessage(llm_response.feedback)],
        "feedback_on_work": llm_response.feedback,
        "success_criteria_met": llm_response.success_criteria_met,
        "user_input_needed": llm_response.user_input_needed,
        "replan_needed": llm_response.replan_needed if replan_allowed else False
    }

    if state.side_effects_requested:
        updates["side_effects_approved"] = llm_response.side_effects_approved

    return updates

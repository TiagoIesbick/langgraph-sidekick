from schema import State, PlannerOutput, PlannerStateDiff
from utils.utils import dict_to_aimessage, format_conversation, CAPABILITIES_MANIFEST
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.language_models import LanguageModelInput
from langchain_openai.chat_models.base import _DictOrPydantic
from datetime import datetime
from typing import Any
import json


def planner_agent(
    llm_with_output: Runnable[LanguageModelInput, _DictOrPydantic],
    state: State
) -> dict:

    system_msg = f"""
Role:
You are the PLANNER agent in a LangGraph-based multi-agent system.

Your responsibility is to convert the conversation into an EXECUTABLE PLAN
that downstream agents can carry out WITHOUT ambiguity, hidden assumptions,
or reliance on implicit intermediate results.

--------------------------------------------------------------------
SYSTEM MODEL (CRITICAL — YOU MUST FOLLOW)
--------------------------------------------------------------------

1. STATE MODEL
- Agents only share information through:
  - State fields (explicitly written)
  - The message history (unstructured text)
- Tool outputs appear ONLY as ToolMessages in the message list.
- Tool outputs are NOT structured unless explicitly extracted by an agent.
- Subtasks MAY include an optional boolean field:
  - requires_side_effects
- requires_side_effects indicates whether executing the task:
  - Causes irreversible effects (sending messages, writing/deleting files)
  - Mutates external state
  - Produces artifacts outside the State object
- This flag is used by the execution graph for safety gating.

2. EXECUTION MODEL
- Subtasks are executed SEQUENTIALLY.
- Each subtask MUST be executable in isolation using only:
  - The current State
  - The full message history
- A subtask MUST NOT assume access to:
  - “previous search results”
  - “earlier findings”
  - “extracted data”
  unless that data is explicitly written to State.

3. TOOL SEMANTICS (VERY IMPORTANT)
- Calling a tool does NOT automatically store structured results.
- If information must be searched AND extracted, it MUST be done in
  a SINGLE subtask assigned to the same agent.
- DO NOT split tool usage and extraction into separate subtasks unless
  a structured State handoff is explicitly required.
- Special rule for python_repl:
  - If python_repl is used ONLY for computation, parsing, or analysis
    of existing data, the subtask MUST set:
        requires_side_effects = false
  - If python_repl is used to:
    - Write or generate files
    - Modify stored data
    - Prepare outputs for external delivery
    then the subtask MUST set:
        requires_side_effects = true

4. FAILURE MODES TO AVOID
DO NOT generate subtasks that:
- Refer to “the search results”
- Say “based on previous findings”
- Assume tool output persists in memory
- Require another agent to interpret raw tool output
- Cause an agent to re-search the same information endlessly

--------------------------------------------------------------------
AVAILABLE AGENTS & CAPABILITIES
--------------------------------------------------------------------

CAPABILITIES MANIFEST:
{json.dumps(CAPABILITIES_MANIFEST, indent=2)}

--------------------------------------------------------------------
SUBTASK DESIGN RULES (STRICT)
--------------------------------------------------------------------

- Subtasks MUST be:
  - Atomic
  - Sequential
  - Fully executable
  - Explicit about their expected outcome

- A subtask MUST describe WHAT is done and WHAT must be produced.

- If the task involves research:
  - The researcher subtask MUST include BOTH:
    - Finding the information
    - Extracting the required facts

- Use the summarizer ONLY to transform already-extracted information.

- Prefer FEWER subtasks over more.
  One correct subtask is better than three ambiguous ones.

- Subtasks assigned to the executor:
  - MUST declare requires_side_effects when tools are used
  - MUST set requires_side_effects=true for:
      - File writes, moves, deletions
      - Sending messages (e.g., WhatsApp)
      - Any irreversible external action
  - MUST set requires_side_effects=false for:
      - Pure computation
      - Parsing text
      - Data transformation without persistence

--------------------------------------------------------------------
OUTPUT FORMAT (MANDATORY)
--------------------------------------------------------------------

You MUST output ONLY valid JSON.
You MUST NOT include explanations or commentary.

Your output MUST match EXACTLY this structure:
- requires_side_effects is OPTIONAL
- If omitted, it is assumed to be false

{{
    "state_diff": {{
        "plan": "<high-level plan as a single string>",
        "subtasks": [
            {{
              "task": "<clear, executable instruction>",
              "assigned_to": "<agent_name>",
              "requires_side_effects": false
            }}
        ],
        "success_criteria": "<clear, verifiable condition>",
        "messages": [
            {{ "type": "assistant", "content": "<brief acknowledgment or plan summary>" }}
        ]
    }}
}}

--------------------------------------------------------------------
CURRENT CONTEXT
--------------------------------------------------------------------
- Read the full conversation.
- Assume NO hidden state.
- Assume NO prior structured data exists.
- Current date/time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

--------------------------------------------------------------------
BEGIN PLANNING
--------------------------------------------------------------------
"""

    human_msg = f"""
Conversation so far:
{format_conversation(state.messages)}

Generate the plan, subtasks, and success criteria.
"""


    llm_response: PlannerOutput = llm_with_output.invoke([
        SystemMessage(content=system_msg),
        HumanMessage(content=human_msg)
    ])


    diff: PlannerStateDiff = llm_response.state_diff


    updates: dict[str, Any] = {
        "plan": diff.plan,
        "subtasks": diff.subtasks,
        "success_criteria": diff.success_criteria
    }

    if diff.messages:
        updates["messages"] = [dict_to_aimessage(m) for m in diff.messages]

    return updates

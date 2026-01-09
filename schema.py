from typing import Annotated
from typing_extensions import Any, Optional, Literal
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class Subtask(BaseModel):
    task: str = Field(
        description=(
            "A single executable responsibility assigned to an agent. "
            "A task may involve multiple internal steps (including tool calls), "
            "but must result in a clear, explicit update to the shared state "
            "that signals task completion.\n\n"
            "Tasks MUST be defined in terms of outcomes, not intermediate actions. "
            "If tool usage and data extraction are required, they must be included "
            "in the same task unless the output is explicitly written to a named "
            "state field for subsequent tasks to consume.\n\n"
        )
    )
    assigned_to: Literal[
        "researcher",
        "executor",
        "summarizer",
        "evaluator"
    ]
    requires_side_effects: bool = False


class State(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    success_criteria: Optional[str] = None
    feedback_on_work: Optional[str] = None
    success_criteria_met: bool = False
    user_input_needed: bool = False
    plan: Optional[str] = None
    subtasks: Optional[list[Subtask]] = None
    next_subtask_index: int = 0
    subtask_results: list[str] = Field(default_factory=list)
    # research_context: Optional[str] = None
    # execution_results: Optional[str] = None
    final_answer: Optional[str] = None


class ClarifierStateDiff(BaseModel):
    messages: Optional[list[dict[str, Any]]] = None
    user_input_needed: Optional[bool] = None


class ClarifierOutput(BaseModel):
    state_diff: ClarifierStateDiff


class PlannerStateDiff(BaseModel):
    plan: str
    subtasks: list[Subtask]
    success_criteria: str
    messages: Optional[list[dict[str, Any]]] = None


class PlannerOutput(BaseModel):
    state_diff: PlannerStateDiff


class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(description="True if more input is needed from the user, or clarifications, or the assistant is stuck")


ToolName = Literal[
    "search",
    "wikipedia",
    "click",
    "navigate",
    "navigate_back",
    "extract_text",
    "extract_hyperlinks",
    "get_elements",
    "current_web_page",
    "python_repl",
    "copy_file",
    "delete_file",
    "file_search",
    "move_file",
    "read_file",
    "write_file",
    "list_directory",
    "send_whatsapp",
]

class ToolInference(BaseModel):
    tool_name: ToolName
    tool_call_id: str

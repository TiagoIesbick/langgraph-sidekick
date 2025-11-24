from typing import Annotated
from typing_extensions import Any, Optional, Literal
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class Subtask(BaseModel):
    task: str = Field(
        description=(
            "A single atomic action the assigned agent must perform. "
            "This should be a concise, executable instruction such as "
            "'search for the latest USD/BRL exchange rate', "
            "'extract the numeric rate from the page contents', or "
            "'summarize the research findings'. "
            "The task must be actionable, specific, and should not contain "
            "multiple independent steps."
        )
    )
    assigned_to: Literal[
        "researcher",
        "executor",
        "summarizer",
        "evaluator"
    ]


class State(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    success_criteria: Optional[str] = None
    feedback_on_work: Optional[str] = None
    success_criteria_met: bool = False
    user_input_needed: bool = False
    plan: Optional[str] = None
    subtasks: Optional[list[Subtask]] = None
    current_task_index: int = 0
    task_results: list[str] = Field(default_factory=list)
    research_context: Optional[str] = None
    execution_results: Optional[str] = None
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

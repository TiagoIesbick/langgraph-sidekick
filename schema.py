from typing import Annotated
from typing_extensions import Any, Optional, Union, Literal
from pydantic import BaseModel, Field, ConfigDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, AIMessage
import json


class State(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    success_criteria: str
    feedback_on_work: Optional[str] = None
    success_criteria_met: bool
    user_input_needed: bool
    plan: Optional[str] = None
    subtasks: Optional[list[str]] = None
    research_context: Optional[str] = None
    execution_results: Optional[str] = None
    final_answer: Optional[str] = None

    def to_json_str(self) -> str:
        """Compact JSON string for feeding into prompts."""
        return json.dumps(self.model_dump(exclude_none=True), separators=(",", ":"))


class ClarifierStateDiff(BaseModel):
    messages: Optional[list[dict]] = None
    user_input_needed: Optional[bool] = None


class ClarifierOutput(BaseModel):
    state_diff: ClarifierStateDiff


class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(description="True if more input is needed from the user, or clarifications, or the assistant is stuck")

from typing import Annotated
from typing_extensions import TypedDict, Any, Optional
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list[Any], add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool


class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(description="True if more input is needed from the user, or clarifications, or the assistant is stuck")

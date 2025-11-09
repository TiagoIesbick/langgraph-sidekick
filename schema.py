from typing import Annotated
from typing_extensions import Any, Optional, Union
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
import json


class State(BaseModel):
    model_config = {
        "extra": "forbid",               # 🚨 Required for OpenAI structured outputs
        "json_schema_extra": {
            "additionalProperties": False  # 🚨 This explicitly satisfies the OpenAI validator
        }
    }
    messages: Annotated[list[Union[BaseMessage, dict[str, Any]]], add_messages] = Field(default_factory=list)
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


class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(description="True if more input is needed from the user, or clarifications, or the assistant is stuck")

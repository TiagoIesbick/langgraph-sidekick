from schema import State
from langchain_core.messages import SystemMessage
from datetime import datetime
from langchain_core.runnables import Runnable
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage


def worker_agent(llm_with_tools: Runnable[LanguageModelInput, BaseMessage], state: State) -> dict[str, list[BaseMessage]]:
    system_message = f"""
You are a helpful assistant that can use tools to complete tasks.
You keep working on a task until either you have a question or clarification for the user, or the success criteria is met.
You have many tools to help you, including tools to browse the internet, navigating and retrieving web pages.
You have a tool to run python code, but note that you would need to include a print() statement if you wanted to receive output.
The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This is the success criteria:
{state.success_criteria}
You should reply either with a question for the user about this assignment, or with your final response.
If you have a question for the user, you need to reply by clearly stating your question. An example might be:

Question: please clarify whether you want a summary or a detailed answer

If you've finished, reply with the final answer, and don't ask a question; simply reply with the answer.
"""

    if state.feedback_on_work:
        system_message += f"""
Previously you thought you completed the assignment, but your reply was rejected because the success criteria was not met.
Here is the feedback on why this was rejected:
{state.feedback_on_work}
With this feedback, please continue the assignment, ensuring that you meet the success criteria or have a question for the user.
"""

    # Add in the system message

    found_system_message = False
    messages = state.messages
    for message in messages:
        if isinstance(message, SystemMessage):
            message.content = system_message
            found_system_message = True

    if not found_system_message:
        messages = [SystemMessage(content=system_message)] + messages

    # Invoke the LLM with tools
    response = llm_with_tools.invoke(messages)

    # Return updated state
    return {
        "messages": [response],
    }

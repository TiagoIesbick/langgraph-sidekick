from schema import State
from langchain_core.messages import SystemMessage, HumanMessage


def planner_agent(llm, state: State):
    system = "You are a planner that turns user goals into actionable subtasks and clear success criteria."
    user = f"User request: {state['messages'][-1].content}"
    result = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return {"plan": result.content, }

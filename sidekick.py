from schema import State, EvaluatorOutput
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from typing import Any
from tools.file_code import file_code_tools
from tools.navigation import playwright_tools
from tools.search import search_tools
from tools.notifications import whatsapp_tool
from agents.worker import worker_agent
from agents.clarifier import clarifier_agent
from agents.evaluator import evaluator_agent
from db.sql_memory import setup_memory
import uuid
import asyncio


class Sidekick:
    def __init__(self):
        self.clarifier_llm_with_output = None
        self.worker_llm_with_tools = None
        self.evaluator_llm_with_output = None
        self.tools = None
        self.llm_with_tools = None
        self.graph = None
        self.sidekick_id = str(uuid.uuid4())
        self.memory = None
        self.browser = None
        self.playwright = None

    async def setup(self):
        self.memory = await setup_memory()
        self.tools, self.browser, self.playwright = await playwright_tools()
        self.tools += await file_code_tools()
        self.tools += await search_tools()
        self.tools.append(whatsapp_tool)
        self.clarifier_llm_with_output = ChatOpenAI(model="gpt-4o-mini").with_structured_output(State)
        self.worker_llm_with_tools = ChatOpenAI(model="gpt-4o-mini").bind_tools(self.tools)
        self.evaluator_llm_with_output = ChatOpenAI(model="gpt-4o-mini").with_structured_output(EvaluatorOutput)
        await self.build_graph()

    def clarifier(self, state: State) -> State:
        return clarifier_agent(self.clarifier_llm_with_output, state)

    def worker(self, state: State) -> dict[str, list[BaseMessage]]:
        return worker_agent(self.worker_llm_with_tools, state)

    def worker_router(self, state: State) -> str:
        last_message = state.messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        else:
            return "evaluator"

    def format_conversation(self, messages: list[Any]) -> str:
        conversation = "Conversation history:\n\n"
        for message in messages:
            if isinstance(message, HumanMessage):
                conversation += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                text = message.content or "[Tools use]"
                conversation += f"Assistant: {text}\n"
        return conversation

    def evaluator(self, state: State) -> State:
        return evaluator_agent(self.evaluator_llm_with_output, self.format_conversation, state)

    def route_based_on_evaluation(self, state: State) -> str:
        if state["success_criteria_met"] or state["user_input_needed"]:
            return "END"
        else:
            return "worker"


    async def build_graph(self):
        # Set up Graph Builder with State
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("clarifier", self.clarifier)
        graph_builder.add_node("worker", self.worker)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("evaluator", self.evaluator)

        # Add edges
        graph_builder.add_edge(START, "clarifier")
        graph_builder.add_conditional_edges("worker", self.worker_router, {"tools": "tools", "evaluator": "evaluator"})
        graph_builder.add_edge("tools", "worker")
        graph_builder.add_conditional_edges("evaluator", self.route_based_on_evaluation, {"worker": "worker", "END": END})
        graph_builder.add_edge(START, "worker")

        # Compile the graph
        self.graph = graph_builder.compile(checkpointer=self.memory)

    async def run_superstep(self, message, success_criteria, history):
        config = {"configurable": {"thread_id": self.sidekick_id}}

        if isinstance(message, str):
            message = [HumanMessage(content=message)]
        elif isinstance(message, BaseMessage):
            message = [message]

        state = State(
            messages=message,
            success_criteria=success_criteria or "The answer should be clear and accurate",
            success_criteria_met=False,
            user_input_needed=False,
        )

        result = await self.graph.ainvoke(state, config=config)
        user = {"role": "user", "content": message}
        reply = {"role": "assistant", "content": result["messages"][-2].content}
        feedback = {"role": "assistant", "content": result["messages"][-1].content}
        return history + [user, reply, feedback]

    async def cleanup(self):
        if self.browser:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.browser.close())
                if self.playwright:
                    loop.create_task(self.playwright.stop())
            except RuntimeError:
                # If no loop is running, do a direct run
                asyncio.run(self.browser.close())
                if self.playwright:
                    asyncio.run(self.playwright.stop())
        if self.memory and hasattr(self.memory, "conn"):
            await self.memory.conn.close()

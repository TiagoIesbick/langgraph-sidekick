from schema import PlannerOutput, State, EvaluatorOutput, ClarifierOutput
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from typing import Any
from tools.file_code import file_code_tools
from tools.navigation import playwright_tools
from tools.search import search_tools
from tools.notifications import whatsapp_tool
from agents.worker import worker_agent
from agents.clarifier import clarifier_agent
from agents.planner import planner_agent
from agents.researcher import researcher_agent
from agents.summarizer import summarizer_agent
from agents.evaluator import evaluator_agent
from db.sql_memory import setup_memory
import uuid
import asyncio


class Sidekick:
    def __init__(self):
        self.clarifier_llm_with_output = None
        self.planner_llm_with_output = None
        self.researcher_llm_with_tools = None
        self.summarizer_llm = None
        self.evaluator_llm_with_output = None
        self.researcher_tools = None
        self.llm_with_tools = None
        self.graph = None
        self.sidekick_id = str(uuid.uuid4())
        self.memory = None
        self.browser = None
        self.playwright = None

    async def setup(self):
        self.memory = await setup_memory()
        self.researcher_tools, self.browser, self.playwright = await playwright_tools()
        # self.tools += await file_code_tools()
        self.researcher_tools += await search_tools()
        # self.tools.append(whatsapp_tool)
        self.clarifier_llm_with_output = ChatOpenAI(model="gpt-4o-mini").with_structured_output(ClarifierOutput, method="function_calling")
        self.planner_llm_with_output = ChatOpenAI(model="gpt-4o-mini").with_structured_output(PlannerOutput, method="function_calling")
        self.researcher_llm_with_tools = ChatOpenAI(model="gpt-4o-mini").bind_tools(self.researcher_tools)
        self.summarizer_llm = ChatOpenAI(model="gpt-4o-mini")
        self.evaluator_llm_with_output = ChatOpenAI(model="gpt-4o-mini").with_structured_output(EvaluatorOutput)
        await self.build_graph()

    def clarifier(self, state: State) -> State:
        return clarifier_agent(self.clarifier_llm_with_output, state)

    def planner(self, state: State) -> State:
        return planner_agent(self.planner_llm_with_output, state)

    def researcher(self, state: State) -> State:
        return researcher_agent(self.researcher_llm_with_tools, state)

    def summarizer(self, state: State) -> State:
        return summarizer_agent(self.summarizer_llm, state)

    def researcher_router(self, state: State) -> str:
        print("[researcher_router] index:", state.current_subtask_index)

        # 1. No plan yet
        if not state.subtasks:
            return "planner"

        # 2. All tasks done
        if state.current_subtask_index >= len(state.subtasks):
            return "evaluator"

        current = state.subtasks[state.current_subtask_index]

        # 3. Task not for researcher → hand off
        if current.assigned_to != "researcher":
            return current.assigned_to

        # 4. Tool call in progress
        if state.messages:
            last = state.messages[-1]
            if getattr(last, "tool_calls", None):
                return "researcher_tools"

        # 5. Otherwise keep researching
        return "researcher"
        # print(f"[researcher_router]: {state}")
        # last_message = state.messages[-1]

        # if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        #     return "researcher_tools"
        # else:
        #     return "researcher"

    def worker(self, state: State) -> dict[str, list[BaseMessage]]:
        return worker_agent(self.worker_llm_with_tools, state)

    def worker_router(self, state: State) -> str:
        last_message = state.messages[-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        else:
            return "evaluator"

    def evaluator(self, state: State) -> State:
        return evaluator_agent(self.evaluator_llm_with_output, state)

    def route_based_on_evaluation(self, state: State) -> str:
        if state["success_criteria_met"] or state["user_input_needed"]:
            return "END"
        else:
            return "worker"

    def route_based_on_clarifition(self, state: State) -> str:
        return "END" if state.user_input_needed else "planner"

    def route_based_on_planner_subtasks(self, state: State) -> str:
        if not state.subtasks:
            return "evaluator"
        next_task = state.subtasks[0]
        return next_task.assigned_to

    async def build_graph(self):
        # Set up Graph Builder with State
        graph_builder = StateGraph(State)

        # Add nodes
        graph_builder.add_node("clarifier", self.clarifier)
        graph_builder.add_node("planner", self.planner)
        graph_builder.add_node("researcher", self.researcher)
        graph_builder.add_node("summarizer", self.summarizer)
        # graph_builder.add_node("worker", self.worker)
        graph_builder.add_node("researcher_tools", ToolNode(tools=self.researcher_tools))
        # graph_builder.add_node("evaluator", self.evaluator)

        # Add edges
        graph_builder.add_edge(START, "clarifier")
        graph_builder.add_conditional_edges(
            "clarifier",
            self.route_based_on_clarifition,
            {"END": END, "planner": "planner"}
        )
        graph_builder.add_conditional_edges(
            "planner",
            self.route_based_on_planner_subtasks,
            {
                "researcher": "researcher",
                # "executor": "executor",
                "summarizer": "summarizer",
                # "evaluator": "evaluator"
            }
        )
        graph_builder.add_conditional_edges(
            "researcher",
            self.researcher_router,
            {
                "researcher_tools": "researcher_tools",
                "planner": "planner",
                "researcher": "researcher",
                # "executor": "executor",
                "summarizer": "summarizer",
                # "evaluator": "evaluator"
            }
        )
        graph_builder.add_edge("researcher_tools", "researcher")
        # graph_builder.add_conditional_edges("worker", self.worker_router, {"tools": "tools", "evaluator": "evaluator"})
        # graph_builder.add_edge("tools", "worker")
        # graph_builder.add_conditional_edges("evaluator", self.route_based_on_evaluation, {"worker": "worker", "END": END})
        # graph_builder.add_edge(START, "worker")

        # Compile the graph
        self.graph = graph_builder.compile(checkpointer=self.memory)

    async def run_superstep(self, message, success_criteria, history):
        config = {"configurable": {"thread_id": self.sidekick_id}}

        if isinstance(message, str):
            message = [HumanMessage(content=message)]
        elif isinstance(message, BaseMessage):
            message = [message]

        initial_state = State(
            messages=message
        )

        result = await self.graph.ainvoke(initial_state, config=config)
        user = {"role": "user", "content":  message[0].content}
        reply = {"role": "assistant", "content": result["messages"][-1].content}
        # feedback = {"role": "assistant", "content": result["messages"][-1].content}
        return history + [user, reply], result.get("user_input_needed", False)

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

import os
import functools
from typing import Literal

import chainlit as cl
from chainlit import make_async

from langchain_openai import ChatOpenAI

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from langchain_core.messages import (
    HumanMessage,
)
from langgraph.prebuilt import ToolNode

from lib.custom_tools import python_repl
from lib.agent_utils import (
    AgentState,
    create_manager_agent,
    create_agent,
    create_agent_node,
)


# Define the model to use
chat_model = os.environ.get("CHAT_MODEL", "gpt-3.5-turbo")
llm = ChatOpenAI(model=chat_model, temperature=0.5)


# Create agents
manager_agent = create_manager_agent(
    llm,
    system_message="""Route the user's input to the correct agent based on the user's input.""",
    route_options=["Calculate", "Elaborate"],
)

calculate_expression_agent = create_agent(
    llm,
    system_message="""Calculate the expression based on the user's input""",
    tools=[python_repl],
)

elaborate_agent = create_agent(
    llm,
    system_message="""Elaborate the user's input for a better understanding""",
    tools=[],
)

# Create Nodes
manager_node = functools.partial(create_agent_node, agent=manager_agent, name="manager")
calculate_expression_node = functools.partial(
    create_agent_node, agent=calculate_expression_agent, name="calculate"
)
elaborate_node = functools.partial(
    create_agent_node, agent=elaborate_agent, name="elaborate"
)
tool_node = ToolNode([python_repl])


# Router
def router(state) -> Literal[
    "call_tool",
    "__end__",
]:
    # This is the router
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        # The previous agent is invoking a tool
        return "call_tool"

    # Any agent decided the work is done
    return "__end__"


def assistent_router(state) -> Literal["Calculate", "Elaborate"]:
    messages = state["messages"]
    last_message = messages[-1]

    return last_message.content


# Create the Graph
workflow = StateGraph(AgentState)
workflow.add_node("manager", manager_node)
workflow.add_node("calculate", calculate_expression_node)
workflow.add_node("elaborate", elaborate_node)
workflow.add_node("call_tool", tool_node)

workflow.set_entry_point("manager")

workflow.add_conditional_edges(
    "manager",
    assistent_router,
    {
        "Calculate": "calculate",
        "Elaborate": "elaborate",
    },
)

workflow.add_conditional_edges(
    "calculate",
    router,
)

workflow.add_conditional_edges(
    "elaborate",
    router,
)

workflow.add_conditional_edges(
    "call_tool",
    lambda x: x["sender"],
)


# Here we only save in-memory
memory = SqliteSaver.from_conn_string(":memory:")

# Compile the graph
app = workflow.compile(checkpointer=memory)

# Draw the graph image
graph_image = app.get_graph().draw_png()


# Chainlit Interface Hooks
@cl.on_chat_start
async def on_start():
    image = cl.Image(content=graph_image, name="Current Graph", display="inline")

    await cl.Message(
        content="Here is the graph being executed", elements=[image], author="system"
    ).send()


@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": "inventory_manager_thread"}}

    # define the messages for the input
    inputs = {
        "messages": [HumanMessage(content=msg.content)],
    }

    response: AgentState = await make_async(app.invoke)(inputs, config, debug=True)

    await cl.Message(content=response["messages"][-1].content, author="system").send()

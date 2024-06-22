import os

import chainlit as cl
from chainlit import make_async

from langchain_openai import ChatOpenAI

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessageGraph
from langgraph.prebuilt.tool_node import ToolNode

from lib.custom_tools import get_titanic_character_info


chat_model = os.environ.get("CHAT_MODEL", "gpt-3.5-turbo")


# Define the function that determines whether to continue or not
def should_continue(messages):
    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return END
    else:
        return "titanic_data_fetcher"


# agent tools
tools = [get_titanic_character_info]


# Define a new graph
workflow = MessageGraph()


llm = ChatOpenAI(
    model=chat_model,
    temperature=0.5,
)
llm_with_tools = llm.bind_tools(tools=[get_titanic_character_info])

workflow.add_node("agent", llm_with_tools)
workflow.add_node("titanic_data_fetcher", ToolNode(tools))

workflow.set_entry_point("agent")

# Whether or to not call the action node
workflow.add_conditional_edges(
    "agent",
    should_continue,
)

# Transition after action has been called for futher processing.
# If we don't add this edge the workflow will return, and the last message
# will be the action node's return value, without any data interpretation.
# IMPORTANT: This is why we need the conditional edge to determine whether to continue or not,
# after the action node has been called and returned back to the agent.
workflow.add_edge("titanic_data_fetcher", "agent")

# Here we only save in-memory
memory = SqliteSaver.from_conn_string(":memory:")

app = workflow.compile(checkpointer=memory)

graph_image = app.get_graph().draw_png()


@cl.on_chat_start
async def on_start():
    image = cl.Image(content=graph_image, name="Current Graph", display="inline")

    await cl.Message(
        content="Here is the graph being executed", elements=[image], author="system"
    ).send()


@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": "4"}}

    response = await make_async(app.invoke)(msg.content, config, debug=True)

    await cl.Message(content=response[-1].content, author="system").send()

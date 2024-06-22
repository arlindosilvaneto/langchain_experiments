import os

import chainlit as cl
from chainlit import make_async

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


from lib.custom_tools import get_titanic_character_info


# Define the model to use
chat_model = os.environ.get("CHAT_MODEL", "gpt-3.5-turbo")

# agent tools
tools = [get_titanic_character_info]

# Define the language model and the tool powered language model
llm = ChatOpenAI(
    model=chat_model,
    temperature=0.5,
)

app = create_react_agent(llm, tools)

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
    config = {"configurable": {"thread_id": "4"}}

    # define the messages for the input
    inputs = {"messages": [("user", msg.content)]}

    response = await make_async(app.invoke)(inputs, config, debug=True)

    await cl.Message(content=response["messages"][-1].content, author="system").send()

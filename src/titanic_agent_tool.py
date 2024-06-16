import os
import chainlit as cl
from chainlit import make_async
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent

from lib.custom_tools import get_titanic_character_info

chat_model = os.environ.get("CHAT_MODEL", "gpt-3.5-turbo")

llm = ChatOpenAI(
    model=chat_model,
    temperature=0.5,
)

prompt = hub.pull("hwchase17/openai-tools-agent")

tools = [get_titanic_character_info]
agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


@cl.on_message
async def on_message(msg: cl.Message):
    response = await make_async(agent_executor.invoke)({"input": msg.content})

    await cl.Message(content=response.get("output"), author="system").send()

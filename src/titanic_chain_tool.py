import os
import chainlit as cl
from chainlit import make_async
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.passthrough import RunnablePassthrough

from lib.custom_tools import get_titanic_character_info

chat_model = os.environ.get("CHAT_MODEL", "gpt-3.5-turbo")

llm = ChatOpenAI(
    model=chat_model,
    temperature=0.5,
)

llm_with_tools = llm.bind_tools([get_titanic_character_info])

prompt = ChatPromptTemplate.from_template(
    "What can you tell me about the Titanic movie character {input}?"
)


def debug_method_call(result):
    print(f"\n\n{result.tool_calls}\n\n")
    return result


chain = (
    RunnablePassthrough()
    | prompt
    | llm_with_tools
    | (lambda x: debug_method_call(x))
    | (lambda x: x.tool_calls[0]["args"])
    | get_titanic_character_info
    | StrOutputParser()
)


@cl.on_chat_start
async def on_start():
    await cl.Message(
        content="Enter the name of the character you want to know about.",
        author="system",
    ).send()


@cl.on_message
async def on_message(msg: cl.Message):
    response = await make_async(chain.invoke)({"input": msg.content})
    print(response)

    await cl.Message(content=response, author="system").send()

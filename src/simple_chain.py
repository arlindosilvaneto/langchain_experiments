import os
import chainlit as cl
from chainlit import make_async
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chat_model = os.environ.get("CHAT_MODEL", "gpt-3.5-turbo")

llm = ChatOpenAI(
    model=chat_model,
    temperature=0.5,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a python developer assistant"),
    ("user", "Elaborate a simple code snipped for the subject {input}")
])

chain = prompt | llm | StrOutputParser()

@cl.on_message
async def on_message(msg: cl.Message):
    response = await make_async(chain.invoke)({"input": msg.content})

    await cl.Message(content=response, author="system").send()

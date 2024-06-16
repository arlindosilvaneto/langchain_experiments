import os
import chainlit as cl
from chainlit import make_async
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.documents import Document

from lib.rag_utils import get_db_from_documents

chat_model = os.environ.get("CHAT_MODEL", "gpt-3.5-turbo")

llm = ChatOpenAI(
    model=chat_model,
    temperature=0.5,
)

prompt = ChatPromptTemplate.from_template(
    "Based on the {context} tell me everything you know about {input}"
)

documents = [
    Document(
        page_content="I sold 1 shirt to a customer called John",
        metadata={"source": "sales", "date": "2024-06-16", "customer": "John"},
    ),
    Document(
        page_content="I sold 2 shirts to a customer called Marcel",
        metadata={"source": "sales", "date": "2024-06-16", "customer": "Marcel"},
    ),
    Document(
        page_content="I bought 10 shirts from Fabric Co",
        metadata={"source": "purchases", "date": "2024-06-16", "vendor": "Fabric Co"},
    ),
]

retriever = get_db_from_documents(documents)

chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


@cl.on_message
async def on_message(msg: cl.Message):
    response = await make_async(chain.invoke)(msg.content)

    await cl.Message(content=response, author="system").send()

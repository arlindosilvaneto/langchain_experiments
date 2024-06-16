import os
import chainlit as cl
from chainlit import make_async
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.documents import Document
from langchain_community.document_loaders import WikipediaLoader

from lib.rag_utils import get_db_from_documents

chat_model = os.environ.get("CHAT_MODEL", "gpt-3.5-turbo")

llm = ChatOpenAI(
    model=chat_model,
    temperature=0.5,
)

documents = WikipediaLoader(
    query="tour de france 2023", load_max_docs=2, lang="en"
).load()

retriever = get_db_from_documents(documents)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


prompt = ChatPromptTemplate.from_template(
    "Based on the {context} I'd like to know more about the Tour de France 2023. Can you tell me about {question}?"
)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


@cl.on_message
async def on_message(msg: cl.Message):
    response = await make_async(chain.invoke)(msg.content)

    await cl.Message(content=response, author="system").send()

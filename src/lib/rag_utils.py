from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS

def get_db_from_documents(documents: list[Document]) -> VectorStoreRetriever:
    embeddings = OpenAIEmbeddings()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )

    chunks = text_splitter.split_documents(documents)

    db = FAISS.from_documents(chunks, embeddings)

    return db.as_retriever()

import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


KNOWLEDGE_DIR = "knowledge_base"
DB_DIR = "chroma_db"


def build_vectorstore():
    """
    Build Chroma vector database from all PDFs.
    """

    documents = []

    for file in os.listdir(KNOWLEDGE_DIR):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(KNOWLEDGE_DIR, file))
            documents.extend(loader.load())

    if len(documents) == 0:
        raise Exception("No PDFs found in knowledge_base folder.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    print(f"Stored {len(docs)} chunks.")

    return vectordb
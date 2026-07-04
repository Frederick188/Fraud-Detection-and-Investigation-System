from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings


DB_DIR = "chroma_db"


def load_vectorstore():

    embeddings = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    return vectordb


def retrieve(query, k=4):

    vectordb = load_vectorstore()

    docs = vectordb.similarity_search(
        query,
        k=k
    )

    return docs
from dotenv import load_dotenv
import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

from src.prompts import INVESTIGATION_PROMPT

load_dotenv()

DB_DIR = "chroma_db"


class FraudAgent:

    def __init__(self):

        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = Chroma(
            persist_directory=DB_DIR,
            embedding_function=embeddings
        )

        self.retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4}
        )

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            temperature=0.2
        )

        self.chain = (
            INVESTIGATION_PROMPT
            | self.llm
            | StrOutputParser()
        )

    def investigate(self, question, investigation):

        docs = self.retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        return self.chain.invoke({

            "transaction":
                investigation["transaction"],

            "processed_features":
                investigation["processed_features"],

            "prediction":
                investigation["prediction"],

            "fraud_probability":
                investigation["fraud_probability"],

            "lime":
                investigation["lime"],

            "context":
                context,

            "question":
                question
        })
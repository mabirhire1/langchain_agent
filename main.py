import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple 

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.tools import StructuredTool

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
)

from langchain_classic.retrievers import EnsembleRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Load .env
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY", "")

# Configuration
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
VECTOR_DB_DIR: Path = BASE_DIR / "vector_db"
VECTOR_DB_DIR.mkdir(exist_ok=True)


# Tool Input Schemas
class FlightScheduleInput(BaseModel):
    origin: str = Field(..., description="Departure city name")
    destination: str = Field(..., description="Arrival city name")

class HotelScheduleInput(BaseModel):
    city: str = Field(..., description="City name")

class ConvertCurrencyInput(BaseModel):
    amount: float
    from_currency: str
    to_currency: str

class RAGInput(BaseModel):
    query: str

# Tools
@tool(args_schema=FlightScheduleInput)
def get_flight_schedule(origin: str, destination: str) -> Dict:
    """
    Use this tool ONLY when the user wants flight information between two cities.
    """
    return {
        "type": "flight_schedule",
        "origin": origin,
        "destination": destination,
        "flight_time_hours": 5.5,
        "price_usd": 920,
    }

@tool(args_schema=HotelScheduleInput)
def get_hotel_schedule(city: str) -> Dict:
    """
    Use this tool ONLY when the user wants hotel information in a specific city.
    """
    return {
        "city": city,
        "hotels": [
            {"name": "Nairobi Serena", "price_usd": 250},
            {"name": "Radisson Blu", "price_usd": 200},
        ],
    }

@tool(args_schema=ConvertCurrencyInput)
def convert_currency(amount: float, from_currency: str, to_currency: str) -> Dict:
    """Use ONLY when user requests hotel options in a city."""
    exchange_rates: Dict[Tuple[str, str], float] = {
        ("USD", "NGN"): 1400.0,
        ("NGN", "USD"): 1 / 1400.0,
    }

    key = (from_currency.upper(), to_currency.upper())
    if key not in exchange_rates:
        raise ValueError(f"Exchange rate not available for {from_currency} → {to_currency}")

    return {
        "type": "currency_conversion",
        "original_amount": amount,
        "amount_converted": amount * exchange_rates[key],
        "currency": to_currency.upper(),
    }

# Hybrid RAG System
class HybridRAGSystem:

    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )

        self.documents: List[Document] = []

        self.vector_store: Optional[Chroma] = None
        self.bm25_retriever: Optional[BM25Retriever] = None

    def load_all_documents(self):
        if not DATA_DIR.exists():
            return

        loader_map = {
            ".txt": TextLoader,
            ".pdf": PyPDFLoader,
            ".csv": CSVLoader,
            ".docx": Docx2txtLoader,
        }

        for file_path in DATA_DIR.glob("*"):
            loader_cls = loader_map.get(file_path.suffix.lower())
            if not loader_cls:
                continue

            loader = loader_cls(str(file_path))
            docs = loader.load()
            split_docs = self.text_splitter.split_documents(docs)
            self.documents.extend(split_docs)

    def index_documents(self):
        if not self.documents:
            return

        self.vector_store = Chroma.from_documents(
            self.documents,
            embedding=self.embeddings,
            persist_directory=str(VECTOR_DB_DIR),
        )

        self.bm25_retriever = BM25Retriever.from_documents(self.documents, k=5)

    def get_hybrid_retriever(self):
        if not self.vector_store or not self.bm25_retriever:
            return None

        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        return EnsembleRetriever(
            retrievers=[self.bm25_retriever, vector_retriever],
            weights=[0.5, 0.5],
        )

    def add_conversation_to_store(self, conversation_text: str):
        if not self.vector_store:
            return

        doc = Document(page_content=conversation_text)
        self.vector_store.add_documents([doc])

# Build Agent
def build_agent():

    rag_system = HybridRAGSystem()
    rag_system.load_all_documents()
    rag_system.index_documents()

    hybrid_retriever = rag_system.get_hybrid_retriever()

    tools = [get_flight_schedule, get_hotel_schedule, convert_currency]

    if hybrid_retriever:

        def search_internal_docs(query: str) -> str:
            docs = hybrid_retriever.invoke(query)
            if not docs:
                return "No relevant documents found."
            return "\n\n".join([doc.page_content for doc in docs])

        rag_tool = StructuredTool.from_function(
            name="search_internal_documents",
            description="Search internal documents and previous conversations.",
            func=search_internal_docs,
            args_schema=RAGInput,
        )

        tools.append(rag_tool)

    llm = ChatOpenAI(
        model="nvidia/nemotron-3-nano-30b-a3b:free",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0.7,
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a professional AI travel assistant.",
    )

    return agent, rag_system

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime
from typing import Annotated, Any, Dict, Optional, TypedDict

from dotenv import load_dotenv
load_dotenv()  # Load OLLAMA_API_KEY from .env

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
import requests

# -------------------
# 1. LLM + embeddings
# LLM  : minimax-m3 via Ollama cloud (https://ollama.com)
# Embed: nomic-embed-text via local Ollama
# -------------------
_OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
if not _OLLAMA_API_KEY:
    raise EnvironmentError(
        "OLLAMA_API_KEY is not set. "
        "Add it to the .env file in this directory."
    )

llm = ChatOllama(
    model="minimax-m3",
    base_url="https://ollama.com",
    headers={"Authorization": f"Bearer {_OLLAMA_API_KEY}"},
)

# Embeddings stay local — no API key needed
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.

    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        # The FAISS store keeps copies of the text, so the temp file is safe to remove.
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=C9PE94QUEW9VWGFM"
    )
    r = requests.get(url)
    return r.json()


@tool
def get_current_datetime() -> dict:
    """
    Returns the current date and time. Use this when the user asks about
    today's date, current time, day of the week, or anything time-related.
    """
    now = datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "full": now.strftime("%A, %B %d, %Y %I:%M %p"),
    }


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [get_current_datetime, search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    now = datetime.now()
    current_datetime = now.strftime("%A, %B %d, %Y %I:%M %p")

    system_message = SystemMessage(
        content=(
            f"You are a smart, helpful AI assistant.\n\n"
            f"CURRENT DATE & TIME: {current_datetime}\n\n"
            "You have access to the following tools:\n"
            "- `get_current_datetime`: Get exact current date and time\n"
            "- `duckduckgo_search`: Search the web for real-time information\n"
            "- `get_stock_price`: Get live stock prices (e.g. AAPL, TSLA)\n"
            "- `calculator`: Perform math operations (add, sub, mul, div)\n"
            f"- `rag_tool`: Search the uploaded PDF document (always pass thread_id='{thread_id}')\n\n"
            "RULES:\n"
            "1. For date/time questions, you already know the current datetime above — answer directly or use get_current_datetime tool.\n"
            "2. For math questions, always use the calculator tool.\n"
            "3. For questions about an uploaded PDF, use rag_tool with the thread_id.\n"
            "4. For real-time news or web info, use duckduckgo_search.\n"
            "5. For stock prices, use get_stock_price.\n"
            "6. Always respond in plain, clear English after getting tool results.\n"
            "7. If no PDF is uploaded and user asks about a document, tell them to upload one."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)

    # minimax-m3 uses the standard Ollama structured tool_calls format — no workaround needed.

    return {"messages": [response]}


tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------
_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatbot.db")
conn = sqlite3.connect(database=_DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)
checkpointer.setup()

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


import threading
_DB_LOCK = threading.Lock()

def delete_thread(thread_id: str) -> None:
    tid = str(thread_id)
    with _DB_LOCK:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM checkpoints WHERE thread_id = ?", (tid,))
        cursor.execute("DELETE FROM writes WHERE thread_id = ?", (tid,))
        conn.commit()
        cursor.close()
    _THREAD_RETRIEVERS.pop(tid, None)
    _THREAD_METADATA.pop(tid, None)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})

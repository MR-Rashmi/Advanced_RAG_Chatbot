
from __future__ import annotations

import base64
import io
import os
import sqlite3
import tempfile
import threading
from datetime import datetime
from typing import Annotated, Any, Dict, Optional, TypedDict

import pdfplumber
import pymupdf
import pytesseract
import requests
from dotenv import load_dotenv
from PIL import Image

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# ============================================================
# 1. ENVIRONMENT CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Local Ollama chat model
llm = ChatOllama(
    model=os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:3b"),
    temperature=0,
)

# Local embedding model
embeddings = OllamaEmbeddings(
    model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
)

# Local vision model for image/chart understanding
VISION_MODEL_NAME = os.getenv(
    "OLLAMA_VISION_MODEL",
    "qwen2.5vl:3b"
)

# Set ENABLE_VISION_ANALYSIS=false in .env to disable vision processing
ENABLE_VISION_ANALYSIS = (
    os.getenv("ENABLE_VISION_ANALYSIS", "true").lower() == "true"
)

# Tesseract OCR configuration
TESSERACT_PATH = os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

# Maximum number of rendered pages sent to the vision model
MAX_VISUAL_PAGES = int(os.getenv("MAX_VISUAL_PAGES", "12"))

# Limit image dimensions before sending to the vision model
MAX_IMAGE_DIMENSION = 1400


# ============================================================
# 2. PER-THREAD DOCUMENT STORAGE
# ============================================================

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

_VISION_LLM = None
_VISION_AVAILABLE = None


def _get_retriever(thread_id: Optional[str]):
    """Return the document retriever for a conversation."""
    if thread_id:
        return _THREAD_RETRIEVERS.get(str(thread_id))
    return None


def _get_vision_model():
    """Load the local vision model only when it is needed."""
    global _VISION_LLM, _VISION_AVAILABLE

    if not ENABLE_VISION_ANALYSIS:
        return None

    if _VISION_AVAILABLE is False:
        return None

    if _VISION_LLM is None:
        try:
            _VISION_LLM = ChatOllama(
                model=VISION_MODEL_NAME,
                temperature=0,
            )
            _VISION_AVAILABLE = True
        except Exception as exc:
            _VISION_AVAILABLE = False
            print(f"Vision model initialization failed: {exc}")
            return None

    return _VISION_LLM


# ============================================================
# 3. IMAGE PREPARATION AND VISION ANALYSIS
# ============================================================

def _prepare_image(image: Image.Image) -> bytes:
    """Resize and convert an image to PNG bytes."""
    image = image.convert("RGB")
    image.thumbnail(
        (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION)
    )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _describe_image(
    image_bytes: bytes,
    description_type: str = "image",
) -> Optional[str]:
    """
    Generate a text description of an image using local Ollama vision.

    The description becomes a normal text document in FAISS.
    """
    global _VISION_AVAILABLE

    vision_model = _get_vision_model()

    if vision_model is None:
        return None

    try:
        image = Image.open(io.BytesIO(image_bytes))
        prepared_bytes = _prepare_image(image)

        encoded_image = base64.b64encode(
            prepared_bytes
        ).decode("utf-8")

        prompt = (
            "Analyze this PDF content carefully. "
            "Describe all meaningful visual information. "
            "If it contains a chart, identify the chart type, "
            "title, axes, labels, legends, trends, comparisons, "
            "and visible values. "
            "If it contains a table, describe its structure and "
            "important values. "
            "If it contains a diagram, explain its components "
            "and relationships. "
            "Do not invent values that are not visible. "
            "Return a detailed, factual description suitable "
            "for searching in a document question-answering system."
        )

        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": f"Content type: {description_type}\n{prompt}",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            "data:image/png;base64,"
                            + encoded_image
                        )
                    },
                },
            ]
        )

        response = vision_model.invoke([message])

        if isinstance(response.content, str):
            return response.content.strip()

        return str(response.content).strip()

    except Exception as exc:
        # Avoid repeatedly trying a vision model that is unavailable.
        if "not found" in str(exc).lower():
            _VISION_AVAILABLE = False

        print(f"Vision analysis failed: {exc}")
        return None


# ============================================================
# 4. PDF IMAGE EXTRACTION
# ============================================================

def _extract_pdf_images(
    pdf_document,
    filename: str,
    extracted_docs: list,
) -> int:
    """Extract embedded images from PDF pages and describe them."""
    image_count = 0
    processed_xrefs = set()

    for page_index, page in enumerate(pdf_document):

        for image_info in page.get_images(full=True):
            xref = image_info[0]

            if xref in processed_xrefs:
                continue

            processed_xrefs.add(xref)

            try:
                image_data = pdf_document.extract_image(xref)
                raw_bytes = image_data.get("image")

                if not raw_bytes:
                    continue

                image = Image.open(io.BytesIO(raw_bytes))

                if image.width < 80 or image.height < 80:
                    continue

                prepared_bytes = _prepare_image(image)

                description = _describe_image(
                    prepared_bytes,
                    description_type="Embedded PDF image",
                )

                if description:
                    extracted_docs.append(
                        Document(
                            page_content=(
                                "Visual content extracted from PDF.\n"
                                + description
                            ),
                            metadata={
                                "source": filename,
                                "page": page_index,
                                "content_type": "image_description",
                                "image_xref": xref,
                            },
                        )
                    )

                image_count += 1

            except Exception as exc:
                print(
                    f"Could not process image on page "
                    f"{page_index + 1}: {exc}"
                )

    return image_count


# ============================================================
# 5. CHART AND DIAGRAM PAGE ANALYSIS
# ============================================================

def _analyze_visual_pages(
    pdf_document,
    filename: str,
    extracted_docs: list,
) -> int:
    """
    Render pages containing images or substantial vector drawings.

    This helps detect charts and diagrams that may not be stored
    as separate embedded image files.
    """
    visual_pages = 0

    for page_index, page in enumerate(pdf_document):

        if visual_pages >= MAX_VISUAL_PAGES:
            break

        try:
            images = page.get_images(full=True)
            drawings = page.get_drawings()

            # A page is considered visually interesting if it contains
            # embedded images or a substantial number of vector drawings.
            has_visual_content = (
                len(images) > 0 or len(drawings) >= 8
            )

            if not has_visual_content:
                continue

            pixmap = page.get_pixmap(
                matrix=pymupdf.Matrix(1.5, 1.5),
                alpha=False,
            )

            image = Image.open(
                io.BytesIO(pixmap.tobytes("png"))
            )

            image_bytes = _prepare_image(image)

            description = _describe_image(
                image_bytes,
                description_type="PDF page containing charts, "
                                 "graphs, diagrams, or images",
            )

            if description:
                extracted_docs.append(
                    Document(
                        page_content=(
                            "Visual analysis of PDF page.\n"
                            + description
                        ),
                        metadata={
                            "source": filename,
                            "page": page_index,
                            "content_type": "chart_or_diagram",
                        },
                    )
                )

            visual_pages += 1

        except Exception as exc:
            print(
                f"Visual page analysis failed on page "
                f"{page_index + 1}: {exc}"
            )

    return visual_pages


# ============================================================
# 6. PDF INGESTION
# ============================================================

def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:
    """
    Extract text, OCR content, tables, images, charts, and diagrams
    from a PDF. Split the content into chunks and index it in FAISS.
    """

    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    if not file_bytes.startswith(b"%PDF"):
        raise ValueError("The uploaded file does not appear to be a PDF.")

    temp_path = None
    source_name = filename or "uploaded.pdf"

    try:
        # ----------------------------------------------------
        # Save PDF temporarily
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:
            temp_file.write(file_bytes)
            temp_path = temp_file.name

        extracted_docs = []

        # ----------------------------------------------------
        # 1. Extract normal PDF text
        # ----------------------------------------------------

        pdf_loader = PyPDFLoader(temp_path)
        pdf_docs = pdf_loader.load()

        for doc in pdf_docs:
            text = doc.page_content.strip()

            if text:
                extracted_docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": source_name,
                            "page": doc.metadata.get("page", 0),
                            "content_type": "text",
                        },
                    )
                )

        # ----------------------------------------------------
        # 2. OCR scanned or image-based PDF pages
        # ----------------------------------------------------

        ocr_pages = 0

        pdf_document = pymupdf.open(temp_path)

        try:
            for page_index, page in enumerate(pdf_document):

                page_text = (
                    pdf_docs[page_index].page_content.strip()
                    if page_index < len(pdf_docs)
                    else ""
                )

                if len(page_text) < 30:

                    pixmap = page.get_pixmap(
                        matrix=pymupdf.Matrix(2, 2),
                        alpha=False,
                    )

                    image = Image.open(
                        io.BytesIO(pixmap.tobytes("png"))
                    )

                    ocr_text = pytesseract.image_to_string(
                        image,
                        lang="eng",
                    ).strip()

                    if ocr_text:
                        extracted_docs.append(
                            Document(
                                page_content=ocr_text,
                                metadata={
                                    "source": source_name,
                                    "page": page_index,
                                    "content_type": "ocr",
                                },
                            )
                        )

                        ocr_pages += 1

        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "Tesseract OCR was not found. Check the TESSERACT_CMD "
                "path or install Tesseract OCR."
            ) from exc

        # ----------------------------------------------------
        # 3. Extract PDF tables
        # ----------------------------------------------------

        table_count = 0

        with pdfplumber.open(temp_path) as pdf:

            for page_index, page in enumerate(pdf.pages):

                for table in page.extract_tables() or []:

                    if not table:
                        continue

                    rows = []

                    for row in table:
                        cleaned_row = [
                            str(cell).strip()
                            if cell is not None
                            else ""
                            for cell in row
                        ]

                        if any(cleaned_row):
                            rows.append(
                                " | ".join(cleaned_row)
                            )

                    if rows:
                        table_text = "\n".join(rows)

                        extracted_docs.append(
                            Document(
                                page_content=(
                                    "Extracted table:\n"
                                    + table_text
                                ),
                                metadata={
                                    "source": source_name,
                                    "page": page_index,
                                    "content_type": "table",
                                },
                            )
                        )

                        table_count += 1

        # ----------------------------------------------------
        # 4. Extract embedded images
        # ----------------------------------------------------

        image_count = _extract_pdf_images(
            pdf_document,
            source_name,
            extracted_docs,
        )

        # ----------------------------------------------------
        # 5. Analyze charts and diagrams
        # ----------------------------------------------------

        visual_pages = _analyze_visual_pages(
            pdf_document,
            source_name,
            extracted_docs,
        )

        pdf_document.close()

        # ----------------------------------------------------
        # 6. Validate extracted content
        # ----------------------------------------------------

        if not extracted_docs:
            raise ValueError(
                "No readable text, tables, OCR content, "
                "or visual descriptions could be extracted."
            )

        # ----------------------------------------------------
        # 7. Split documents into chunks
        # ----------------------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = splitter.split_documents(extracted_docs)

        if not chunks:
            raise ValueError(
                "Content was extracted, but no chunks were generated."
            )

        # ----------------------------------------------------
        # 8. Create FAISS vector index
        # ----------------------------------------------------

        vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=embeddings,
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6},
        )

        # Store retriever for this conversation
        tid = str(thread_id)
        _THREAD_RETRIEVERS[tid] = retriever

        # ----------------------------------------------------
        # 9. Store document metadata
        # ----------------------------------------------------

        summary = {
            "filename": source_name,
            "documents": len(extracted_docs),
            "chunks": len(chunks),
            "tables": table_count,
            "ocr_pages": ocr_pages,
            "images": image_count,
            "visual_pages": visual_pages,
            "vision_model": (
                VISION_MODEL_NAME
                if ENABLE_VISION_ANALYSIS
                else "disabled"
            ),
        }

        _THREAD_METADATA[tid] = summary

        return summary

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# 7. TOOLS
# ============================================================

# Keep your existing DuckDuckGo integration unchanged.
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str,
) -> dict:
    """Perform addition, subtraction, multiplication, or division."""

    try:
        if operation == "add":
            result = first_num + second_num

        elif operation == "sub":
            result = first_num - second_num

        elif operation == "mul":
            result = first_num * second_num

        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed."}
            result = first_num / second_num

        else:
            return {
                "error": f"Unsupported operation: {operation}"
            }

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }

    except Exception as exc:
        return {"error": str(exc)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest available stock quote using Alpha Vantage.
    Requires ALPHA_VANTAGE_API_KEY in the .env file.
    """

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not api_key:
        return {
            "error": "Alpha Vantage API key is not configured."
        }

    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": api_key,
            },
            timeout=15,
        )

        response.raise_for_status()
        return response.json()

    except requests.RequestException as exc:
        return {
            "error": f"Stock API request failed: {str(exc)}"
        }


@tool
def get_current_datetime() -> dict:
    """Return the current local date and time."""

    now = datetime.now()

    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "day_of_week": now.strftime("%A"),
        "full": now.strftime("%A, %B %d, %Y %I:%M %p"),
    }


@tool
def rag_tool(
    query: str,
    thread_id: Optional[str] = None,
) -> dict:
    """
    Retrieve relevant text, tables, OCR content, and visual
    descriptions from the PDF uploaded in this conversation.
    Always pass the current thread_id.
    """

    retriever = _get_retriever(thread_id)

    if retriever is None:
        return {
            "error": (
                "No document is indexed for this chat. "
                "Upload a PDF first."
            ),
            "query": query,
        }

    try:
        results = retriever.invoke(query)

        context = []

        for doc in results:
            context.append({
                "content": doc.page_content,
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "content_type": doc.metadata.get("content_type"),
            })

        return {
            "query": query,
            "context": context,
            "source_file": _THREAD_METADATA.get(
                str(thread_id), {}
            ).get("filename"),
        }

    except Exception as exc:
        return {
            "error": f"Document retrieval failed: {str(exc)}",
            "query": query,
        }


tools = [
    get_current_datetime,
    search_tool,
    get_stock_price,
    calculator,
    rag_tool,
]

llm_with_tools = llm.bind_tools(tools)


# ============================================================
# 8. LANGGRAPH STATE
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# 9. LANGGRAPH CHAT NODE
# ============================================================

def chat_node(state: ChatState, config=None):
    """LLM node that answers or requests a tool call."""

    thread_id = None

    if config and isinstance(config, dict):
        thread_id = (
            config.get("configurable", {}).get("thread_id")
        )

    now = datetime.now()

    current_datetime = now.strftime(
        "%A, %B %d, %Y %I:%M %p"
    )

    system_message = SystemMessage(
        content=(
            "You are a smart, helpful AI assistant.\n\n"

            f"CURRENT DATE & TIME: {current_datetime}\n\n"

            "You have access to these tools:\n"
            "- get_current_datetime: current date and time\n"
            "- duckduckgo_search: web search\n"
            "- get_stock_price: stock quotes\n"
            "- calculator: arithmetic operations\n"
            "- rag_tool: search uploaded PDF content\n\n"

            "RULES:\n"
            "1. Answer date and time questions using the current "
            "date and time above or the date/time tool.\n"
            "2. Always use calculator for arithmetic questions.\n"
            "3. For questions about an uploaded PDF, use rag_tool.\n"
            "4. When calling rag_tool, pass the current thread ID: "
            f"'{thread_id}'.\n"
            "5. PDF retrieval can contain normal text, OCR text, "
            "tables, and descriptions of images, charts, and diagrams.\n"
            "6. Use the retrieved context to answer document questions. "
            "Do not invent information that is not in the context.\n"
            "7. Mention the source page when page metadata is available.\n"
            "8. For real-time web information, use DuckDuckGo search.\n"
            "9. For stock prices, use get_stock_price.\n"
            "10. After using tools, explain the results in clear English.\n"
            "11. If no PDF is uploaded and the user asks about a "
            "specific document, ask them to upload it.\n"
        )
    )

    messages = [
        system_message,
        *state["messages"],
    ]

    response = llm_with_tools.invoke(
        messages,
        config=config,
    )

    return {"messages": [response]}


# ============================================================
# 10. LANGGRAPH TOOL NODE
# ============================================================

tool_node = ToolNode(tools)


# ============================================================
# 11. SQLITE CHECKPOINTER
# ============================================================

_DB_PATH = os.path.join(
    BASE_DIR,
    "chatbot.db",
)

conn = sqlite3.connect(
    database=_DB_PATH,
    check_same_thread=False,
)

checkpointer = SqliteSaver(conn=conn)
checkpointer.setup()

_DB_LOCK = threading.Lock()


# ============================================================
# 12. BUILD LANGGRAPH WORKFLOW
# ============================================================

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges(
    "chat_node",
    tools_condition,
)

graph.add_edge("tools", "chat_node")

chatbot = graph.compile(
    checkpointer=checkpointer,
)


# ============================================================
# 13. THREAD MANAGEMENT HELPERS
# ============================================================

def retrieve_all_threads():
    """Return thread IDs stored in SQLite checkpoints."""

    all_threads = set()

    for checkpoint in checkpointer.list(None):
        thread_id = (
            checkpoint.config["configurable"]["thread_id"]
        )
        all_threads.add(thread_id)

    return list(all_threads)


def delete_thread(thread_id: str) -> None:
    """Delete a conversation's SQLite checkpoints and document cache."""

    tid = str(thread_id)

    with _DB_LOCK:
        cursor = conn.cursor()

        cursor.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?",
            (tid,),
        )

        cursor.execute(
            "DELETE FROM writes WHERE thread_id = ?",
            (tid,),
        )

        conn.commit()
        cursor.close()

    _THREAD_RETRIEVERS.pop(tid, None)
    _THREAD_METADATA.pop(tid, None)


def thread_has_document(thread_id: str) -> bool:
    """Check whether a document is indexed for this thread."""

    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    """Return metadata for the document uploaded in this thread."""

    return _THREAD_METADATA.get(str(thread_id), {})

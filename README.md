AI RAG Chatbot with Multi-Tool Agent using LangGraph & Ollama

Developed an AI-powered conversational chatbot using Python, LangGraph, LangChain, and Ollama, capable of answering user queries through Retrieval-Augmented Generation (RAG), multimodal PDF understanding, web search, and custom tools.

Built an advanced RAG pipeline by extracting text, tables, scanned content, images, charts, and diagrams from PDF documents using PyMuPDF, pdfplumber, and Tesseract OCR. Integrated Qwen2.5-VL through Ollama to analyze visual content and generate descriptions, which are indexed alongside extracted document text using Nomic Embed embeddings and FAISS for context-aware question answering.

Integrated multiple AI tools including DuckDuckGo Search, stock price retrieval, calculator, date/time utilities, and document-based question answering using LangChain tool calling to extend chatbot capabilities.

Implemented a stateful multi-agent workflow using LangGraph with conditional tool execution, persistent conversation checkpoints using SQLite, and thread-based document management for organized user interactions.

Designed a modular architecture supporting local LLM integration, vision-based document analysis, environment variable management, efficient document chunking, vector search, and reusable components for extensible deployment.

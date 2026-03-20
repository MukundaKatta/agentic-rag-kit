"""Agentic RAG Kit - A framework for building agentic RAG pipelines."""

from .document import Document, Chunk, SearchResult, ReasoningStep
from .chunker import (
    BaseChunker,
    FixedSizeChunker,
    SentenceChunker,
    ParagraphChunker,
    RecursiveChunker,
)
from .retriever import BaseRetriever, SimpleRetriever, SemanticRetriever, HybridRetriever
from .knowledge_base import KnowledgeBase
from .agent import RAGAgent

__version__ = "0.1.0"
__author__ = "Mukunda Katta"

__all__ = [
    # Core classes
    "RAGAgent",
    "KnowledgeBase",
    # Data models
    "Document",
    "Chunk",
    "SearchResult",
    "ReasoningStep",
    # Chunkers
    "BaseChunker",
    "FixedSizeChunker",
    "SentenceChunker",
    "ParagraphChunker",
    "RecursiveChunker",
    # Retrievers
    "BaseRetriever",
    "SimpleRetriever",
    "SemanticRetriever",
    "HybridRetriever",
]

"""Data models for documents, chunks, and search results."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class Document:
    """Represents a source document."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None

    def __post_init__(self):
        """Validate document on creation."""
        if not self.id:
            raise ValueError("Document id cannot be empty")
        if not self.content:
            raise ValueError("Document content cannot be empty")


@dataclass
class Chunk:
    """Represents a chunk of a document."""

    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_doc_id: str = ""
    position: int = 0  # Position in the document (0-indexed)

    def __post_init__(self):
        """Validate chunk on creation."""
        if not self.id:
            raise ValueError("Chunk id cannot be empty")
        if not self.content:
            raise ValueError("Chunk content cannot be empty")


@dataclass
class SearchResult:
    """Represents a search result."""

    chunk: Chunk
    score: float  # Relevance score (0-1)
    retriever_name: str = "unknown"


@dataclass
class ReasoningStep:
    """Represents a step in the agent's reasoning trace."""

    step_number: int
    action: str  # e.g., "initial_query", "refine_query", "synthesize"
    query: str
    results: List[SearchResult] = field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.0  # Confidence in having enough information

    def __str__(self) -> str:
        """Return human-readable string representation."""
        return (
            f"Step {self.step_number}: {self.action}\n"
            f"  Query: {self.query}\n"
            f"  Results found: {len(self.results)}\n"
            f"  Confidence: {self.confidence:.2f}\n"
            f"  Reasoning: {self.reasoning}"
        )

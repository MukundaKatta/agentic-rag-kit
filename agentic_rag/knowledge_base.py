"""Knowledge base management and document indexing."""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

from .document import Document, Chunk
from .chunker import RecursiveChunker, BaseChunker


class KnowledgeBase:
    """Manages documents and chunks for RAG."""

    def __init__(self, chunker: Optional[BaseChunker] = None):
        """
        Initialize knowledge base.

        Args:
            chunker: Document chunker to use (defaults to RecursiveChunker)
        """
        self.chunker = chunker or RecursiveChunker()
        self.documents: List[Document] = []
        self.chunks: List[Chunk] = []
        self._doc_id_counter = 0
        self._chunk_id_counter = 0

    def add_text(
        self, content: str, metadata: Optional[Dict[str, Any]] = None, source: Optional[str] = None
    ) -> Document:
        """
        Add a text document to the knowledge base.

        Args:
            content: Text content
            metadata: Optional metadata dictionary
            source: Optional source identifier

        Returns:
            The created Document
        """
        if not content or not content.strip():
            raise ValueError("Content cannot be empty")

        doc_id = f"doc_{self._doc_id_counter}"
        self._doc_id_counter += 1

        document = Document(
            id=doc_id, content=content, metadata=metadata or {}, source=source
        )

        self.documents.append(document)
        self._chunk_document(document)

        return document

    def add_file(self, file_path: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        """
        Add a document from a file.

        Args:
            file_path: Path to the file
            metadata: Optional metadata dictionary

        Returns:
            The created Document
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read file content
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="latin-1") as f:
                content = f.read()

        # Add metadata about file
        file_metadata = metadata or {}
        file_metadata.setdefault("filename", path.name)
        file_metadata.setdefault("file_path", str(path.absolute()))

        return self.add_text(content, metadata=file_metadata, source=str(path.name))

    def _chunk_document(self, document: Document) -> None:
        """
        Chunk a document and add chunks to knowledge base.

        Args:
            document: Document to chunk
        """
        chunks = self.chunker.chunk(document)
        # Update chunk IDs with global counter
        for chunk in chunks:
            chunk.id = f"chunk_{self._chunk_id_counter}"
            self._chunk_id_counter += 1
        self.chunks.extend(chunks)

    def remove_document(self, doc_id: str) -> bool:
        """
        Remove a document and its chunks.

        Args:
            doc_id: Document ID to remove

        Returns:
            True if document was removed, False if not found
        """
        # Find and remove document
        doc_index = None
        for i, doc in enumerate(self.documents):
            if doc.id == doc_id:
                doc_index = i
                break

        if doc_index is None:
            return False

        self.documents.pop(doc_index)

        # Remove associated chunks
        self.chunks = [chunk for chunk in self.chunks if chunk.source_doc_id != doc_id]

        return True

    def clear(self) -> None:
        """Clear all documents and chunks."""
        self.documents.clear()
        self.chunks.clear()
        self._chunk_id_counter = 0

    def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get a document by ID.

        Args:
            doc_id: Document ID

        Returns:
            Document or None if not found
        """
        for doc in self.documents:
            if doc.id == doc_id:
                return doc
        return None

    def get_chunks_for_document(self, doc_id: str) -> List[Chunk]:
        """
        Get all chunks for a document.

        Args:
            doc_id: Document ID

        Returns:
            List of chunks from this document
        """
        return [chunk for chunk in self.chunks if chunk.source_doc_id == doc_id]

    def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.

        Returns:
            Dictionary with stats
        """
        total_chars = sum(len(doc.content) for doc in self.documents)
        total_words = sum(len(doc.content.split()) for doc in self.documents)

        return {
            "num_documents": len(self.documents),
            "num_chunks": len(self.chunks),
            "total_characters": total_chars,
            "total_words": total_words,
            "avg_chunk_size": total_chars / len(self.chunks) if self.chunks else 0,
        }

    def info(self) -> str:
        """
        Get human-readable information about the knowledge base.

        Returns:
            Information string
        """
        stats = self.get_stats()
        return (
            f"Knowledge Base Info:\n"
            f"  Documents: {stats['num_documents']}\n"
            f"  Chunks: {stats['num_chunks']}\n"
            f"  Total Characters: {stats['total_characters']:+}\n"
            f"  Total Words: {stats['total_words']:+}\n"
            f"  Avg Chunk Size: {stats['avg_chunk_size']:.0f} chars"
        )

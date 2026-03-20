"""Document chunking strategies."""

import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from .document import Chunk, Document


class BaseChunker(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(
        self, document: Document, chunk_size: int = 512, overlap: int = 50
    ) -> List[Chunk]:
        """
        Chunk a document.

        Args:
            document: Document to chunk
            chunk_size: Target chunk size (meaning varies by strategy)
            overlap: Overlap between chunks (meaning varies by strategy)

        Returns:
            List of chunks
        """
        pass


class FixedSizeChunker(BaseChunker):
    """Chunks documents into fixed-size pieces by word or character count."""

    def __init__(self, by: str = "words"):
        """
        Initialize the chunker.

        Args:
            by: "words" for word-based chunking, "chars" for character-based
        """
        if by not in ("words", "chars"):
            raise ValueError("by must be 'words' or 'chars'")
        self.by = by

    def chunk(
        self, document: Document, chunk_size: int = 512, overlap: int = 50
    ) -> List[Chunk]:
        """Chunk document by fixed size."""
        chunks = []
        chunk_counter = 0

        if self.by == "words":
            words = document.content.split()
            for i in range(0, len(words), chunk_size - overlap):
                chunk_words = words[i : i + chunk_size]
                if not chunk_words:
                    continue

                content = " ".join(chunk_words)
                chunk = Chunk(
                    id=f"{document.id}_chunk_{chunk_counter}",
                    content=content,
                    metadata=document.metadata.copy(),
                    source_doc_id=document.id,
                    position=chunk_counter,
                )
                chunks.append(chunk)
                chunk_counter += 1
        else:  # by == "chars"
            content = document.content
            for i in range(0, len(content), chunk_size - overlap):
                chunk_content = content[i : i + chunk_size]
                if not chunk_content.strip():
                    continue

                chunk = Chunk(
                    id=f"{document.id}_chunk_{chunk_counter}",
                    content=chunk_content,
                    metadata=document.metadata.copy(),
                    source_doc_id=document.id,
                    position=chunk_counter,
                )
                chunks.append(chunk)
                chunk_counter += 1

        return chunks


class SentenceChunker(BaseChunker):
    """Chunks documents by sentences, respecting sentence boundaries."""

    def chunk(
        self, document: Document, chunk_size: int = 3, overlap: int = 1
    ) -> List[Chunk]:
        """
        Chunk document by sentences.

        Args:
            document: Document to chunk
            chunk_size: Number of sentences per chunk
            overlap: Number of sentences to overlap

        Returns:
            List of chunks
        """
        # Split into sentences (simple approach)
        sentences = re.split(r"(?<=[.!?])\s+", document.content.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        chunk_counter = 0

        for i in range(0, len(sentences), chunk_size - overlap):
            chunk_sentences = sentences[i : i + chunk_size]
            if not chunk_sentences:
                continue

            content = " ".join(chunk_sentences)
            chunk = Chunk(
                id=f"{document.id}_chunk_{chunk_counter}",
                content=content,
                metadata=document.metadata.copy(),
                source_doc_id=document.id,
                position=chunk_counter,
            )
            chunks.append(chunk)
            chunk_counter += 1

        return chunks


class ParagraphChunker(BaseChunker):
    """Chunks documents by paragraphs, respecting paragraph boundaries."""

    def chunk(
        self, document: Document, chunk_size: int = 2, overlap: int = 0
    ) -> List[Chunk]:
        """
        Chunk document by paragraphs.

        Args:
            document: Document to chunk
            chunk_size: Number of paragraphs per chunk
            overlap: Number of paragraphs to overlap (usually 0)

        Returns:
            List of chunks
        """
        # Split into paragraphs (separated by blank lines)
        paragraphs = re.split(r"\n\s*\n", document.content.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        chunk_counter = 0

        for i in range(0, len(paragraphs), chunk_size - overlap):
            chunk_paragraphs = paragraphs[i : i + chunk_size]
            if not chunk_paragraphs:
                continue

            content = "\n\n".join(chunk_paragraphs)
            chunk = Chunk(
                id=f"{document.id}_chunk_{chunk_counter}",
                content=content,
                metadata=document.metadata.copy(),
                source_doc_id=document.id,
                position=chunk_counter,
            )
            chunks.append(chunk)
            chunk_counter += 1

        return chunks


class RecursiveChunker(BaseChunker):
    """
    Recursively chunks documents, trying to preserve document structure.

    Attempts chunking at multiple levels (paragraphs, sentences, words)
    until reaching the target size.
    """

    def chunk(
        self, document: Document, chunk_size: int = 512, overlap: int = 50
    ) -> List[Chunk]:
        """
        Recursively chunk document trying to preserve structure.

        Args:
            document: Document to chunk
            chunk_size: Target chunk size in words
            overlap: Overlap in words

        Returns:
            List of chunks
        """
        separators = [
            "\n\n",  # Paragraph breaks
            "\n",  # Line breaks
            ". ",  # Sentences
            " ",  # Words
            "",  # Characters (fallback)
        ]

        return self._recursive_split(
            document.content, document.id, document.metadata, separators, chunk_size, overlap
        )

    def _recursive_split(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any],
        separators: List[str],
        chunk_size: int,
        overlap: int,
        chunk_counter: int = 0,
    ) -> List[Chunk]:
        """
        Recursively split text using separators.

        Args:
            text: Text to split
            doc_id: Document ID
            metadata: Document metadata
            separators: List of separators to try in order
            chunk_size: Target chunk size
            overlap: Overlap between chunks
            chunk_counter: Counter for chunk numbering

        Returns:
            List of chunks
        """
        chunks = []
        good_splits = []
        separator = separators[-1]

        for _s in separators:
            if _s == "":
                break
            if _s in text:
                separator = _s
                break

        # Split by separator
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        # Filter out empty splits
        good_splits = [s for s in splits if s.strip()]

        # Merge splits if they're too small
        merged_text = []
        separator_text = separator if separator else ""

        for split in good_splits:
            if not merged_text:
                merged_text.append(split)
            else:
                # Check if merged would exceed chunk size
                merged = separator_text.join(merged_text + [split])
                if len(merged.split()) <= chunk_size:
                    merged_text.append(split)
                else:
                    # Flush current merged text
                    if merged_text:
                        combined = separator_text.join(merged_text)
                        chunk = Chunk(
                            id=f"{doc_id}_chunk_{chunk_counter}",
                            content=combined,
                            metadata=metadata.copy(),
                            source_doc_id=doc_id,
                            position=chunk_counter,
                        )
                        chunks.append(chunk)
                        chunk_counter += 1
                    merged_text = [split]

        # Add remaining merged text
        if merged_text:
            combined = separator_text.join(merged_text)
            chunk = Chunk(
                id=f"{doc_id}_chunk_{chunk_counter}",
                content=combined,
                metadata=metadata.copy(),
                source_doc_id=doc_id,
                position=chunk_counter,
            )
            chunks.append(chunk)

        return chunks

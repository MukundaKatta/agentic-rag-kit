"""Retriever implementations for searching the knowledge base."""

from abc import ABC, abstractmethod
from typing import List, Dict, Set
import math

import numpy as np

from .document import Chunk, SearchResult


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""

    def __init__(self, knowledge_base):
        """
        Initialize retriever.

        Args:
            knowledge_base: KnowledgeBase instance to search
        """
        self.knowledge_base = knowledge_base

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of SearchResult objects ranked by relevance
        """
        pass


class SimpleRetriever(BaseRetriever):
    """
    Simple keyword-based retriever using TF-IDF style scoring.

    Ranks documents by term frequency matching without external dependencies.
    """

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Retrieve chunks using keyword matching.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of SearchResult objects ranked by relevance
        """
        if not self.knowledge_base.chunks:
            return []

        # Normalize and tokenize query
        query_terms = self._tokenize(query)
        if not query_terms:
            return []

        # Score each chunk
        scored_chunks = []
        for chunk in self.knowledge_base.chunks:
            score = self._score_chunk(chunk, query_terms)
            if score > 0:
                scored_chunks.append((chunk, score))

        # Sort by score (descending) and return top-k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        results = [
            SearchResult(chunk=chunk, score=score, retriever_name="SimpleRetriever")
            for chunk, score in scored_chunks[:top_k]
        ]

        return results

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        """
        Tokenize text into terms.

        Args:
            text: Text to tokenize

        Returns:
            Set of lowercase terms
        """
        # Convert to lowercase and split on non-alphanumeric
        import re

        terms = re.findall(r"\b\w+\b", text.lower())
        # Remove common stopwords
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "is",
            "was",
            "are",
            "be",
        }
        return {term for term in terms if term not in stopwords}

    @staticmethod
    def _score_chunk(chunk: Chunk, query_terms: Set[str]) -> float:
        """
        Score a chunk based on query term matches.

        Args:
            chunk: Chunk to score
            query_terms: Set of query terms

        Returns:
            Relevance score between 0 and 1
        """
        import re

        chunk_terms = set(re.findall(r"\b\w+\b", chunk.content.lower()))
        chunk_terms = {term for term in chunk_terms}

        if not query_terms:
            return 0.0

        # Calculate Jaccard similarity
        intersection = len(query_terms & chunk_terms)
        union = len(query_terms | chunk_terms)

        if union == 0:
            return 0.0

        return intersection / union


class SemanticRetriever(BaseRetriever):
    """
    Semantic retriever using simple embedding-based similarity.

    Uses a basic embedding approach (counting unique words/features)
    and cosine similarity without external APIs.
    """

    def __init__(self, knowledge_base):
       """
        Initialize semantic retriever.

        Args:
            knowledge_base: KnowledgeBase instance to search
        """
        super().__init__(knowledge_base)
        self._embedding_cache = {}

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Retrieve chunks using semantic similarity.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of SearchResult objects ranked by relevance
        """
        if not self.knowledge_base.chunks:
            return []

        # Get query embedding
        query_embedding = self._embed(query)
        if query_embedding is None:
            return []

        # Score each chunk
        scored_chunks = []
        for chunk in self.knowledge_base.chunks:
            chunk_embedding = self._embed(chunk.content)
            if chunk_embedding is not None:
                score = self._cosine_similarity(query_embedding, chunk_embedding)
                if score > 0:
                    scored_chunks.append((chunk, score))

        # Sort by score (descending) and return top-k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        results = [
            SearchResult(chunk=chunk, score=score, retriever_name="SemanticRetriever")
            for chunk, score in scored_chunks[:top_k]
        ]

        return results

    def _embed(self, text: str):
        """
        Create a simple embedding from text.

        Uses a simple bag-of-words approach with TF weighting.

        Args:
            text: Text to embed

        Returns:
            Embedding as numpy array
        """
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        import re

        terms = re.findall(r"\b\w+\b", text.lower())
        if not terms:
            return None

        # Create term frequency vector
        term_counts = {}
        for term in terms:
            term_counts[term] = term_counts.get(term, 0) + 1

        # Create fixed-size embedding (hash to bins)
        embedding_size = 100
        embedding = np.zeros(embedding_size)

        for term, count in term_counts.items():
            bin_idx = hash(term) % embedding_size
            embedding[bin_idx] += count

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        self._embedding_cache[text] = embedding
        return embedding

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score between 0 and 1
        """
        if vec1 is None or vec2 is None:
            return 0.0

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining keyword and semantic search.

    Combines SimpleRetriever and SemanticRetriever scores using weighted average.
    """

    def __init__(self, knowledge_base, keyword_weight: float = 0.4, semantic_weight: float = 0.6):
       """
        Initialize hybrid retriever.

        Args:
            knowledge_base: KnowledgeBase instance to search
            keyword_weight: Weight for keyword retriever (0-1)
            semantic_weight: Weight for semantic retriever (0-1)
        """
        super().__init__(knowledge_base)

        total = keyword_weight + semantic_weight
        self.keyword_weight = keyword_weight / total
        self.semantic_weight = semantic_weight / total

        self.keyword_retriever = SimpleRetriever(knowledge_base)
        self.semantic_retriever = SemanticRetriever(knowledge_base)

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
       """
        Retrieve chunks using hybrid approach.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of SearchResult objects ranked by relevance
        """
        if not self.knowledge_base.chunks:
            return []

        # Get results from both retrievers
        keyword_results = self.keyword_retriever.retrieve(query, top_k=top_k * 2)
        semantic_results = self.semantic_retriever.retrieve(query, top_k=top_k * 2)

        # Create score map: chunk_id -> score
        chunk_scores: Dict[str, float] = {}

        # Add keyword scores
        for result in keyword_results:
            chunk_id = result.chunk.id
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + result.score * self.keyword_weight

        # Add semantic scores
        for result in semantic_results:
            chunk_id = result.chunk.id
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + result.score * self.semantic_weight

        # Create result list sorted by combined score
        chunk_map = {chunk.id: chunk for chunk in self.knowledge_base.chunks}
        scored_chunks = [
            (chunk_map[chunk_id], score)
            for chunk_id, score in chunk_scores.items()
            if chunk_id in chunk_map
        ]
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        results = [
            SearchResult(chunk=chunk, score=score, retriever_name="HybridRetriever")
            for chunk, score in scored_chunks[:top_k]
        ]

        return results

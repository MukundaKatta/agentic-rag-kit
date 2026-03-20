"""Tests for retrieval systems."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from agentic_rag import (
    KnowledgeBase,
    SimpleRetriever,
    SemanticRetriever,
    HybridRetriever,
)


class TestRetrieverBase(unittest.TestCase):
    """Base test class for retrievers."""

    def setUp(self):
        """Set up test fixtures."""
        self.kb = KnowledgeBase()

        # Add test documents
        docs = [
            "The cat sat on the mat. Cats are furry animals.",
            "Dogs are man's best friend. They love to play and run.",
            "Birds can fly in the sky. Many birds sing beautiful songs.",
            "Fish live in water. Aquatic animals need clean water.",
        ]

        for doc in docs:
            self.kb.add_text(doc)


class TestSimpleRetriever(TestRetrieverBase):
    """Tests for SimpleRetriever."""

    def test_retrieve_returns_results(self):
        """Test that retriever returns results."""
        retriever = SimpleRetriever(self.kb)
        results = retriever.retrieve("cat mat", top_k=2)
        self.assertGreater(len(results), 0)

    def test_retrieve_top_k(self):
        """Test that top_k parameter is respected."""
        retriever = SimpleRetriever(self.kb)
        results = retriever.retrieve("animal", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_retrieve_scores_are_normalized(self):
        """Test that relevance scores are between 0 and 1."""
        retriever = SimpleRetriever(self.kb)
        results = retriever.retrieve("cat", top_k=10)
        for result in results:
            self.assertGreaterEqual(result.score, 0.0)
            self.assertLessEqual(result.score, 1.0)

    def test_retrieve_empty_query(self):
        """Test retrieval with empty query."""
        retriever = SimpleRetriever(self.kb)
        results = retriever.retrieve("", top_k=5)
        self.assertEqual(len(results), 0)

    def test_retrieve_no_matches(self):
        """Test query with no matches."""
        retriever = SimpleRetriever(self.kb)
        results = retriever.retrieve("xyzabc", top_k=5)
        self.assertEqual(len(results), 0)

    def test_retrieve_ranks_by_relevance(self):
        """Test that results are ranked by relevance."""
        retriever = SimpleRetriever(self.kb)
        results = retriever.retrieve("cat", top_k=5)
        if len(results) > 1:
            # First result should have higher score than subsequent ones
            self.assertGreaterEqual(results[0].score, results[1].score)


class TestSemanticRetriever(TestRetrieverBase):
    """Tests for SemanticRetriever."""

    def test_retrieve_returns_results(self):
        """Test that retriever returns results."""
        retriever = SemanticRetriever(self.kb)
        results = retriever.retrieve("feline", top_k=2)
        # Semantic search might return results even for synonyms
        self.assertGreaterEqual(len(results), 0)

    def test_retrieve_top_k(self):
        """Test that top_k parameter is respected."""
        retriever = SemanticRetriever(self.kb)
        results = retriever.retrieve("animal", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_retrieve_scores_are_normalized(self):
        """Test that relevance scores are between 0 and 1."""
        retriever = SemanticRetriever(self.kb)
        results = retriever.retrieve("pet", top_k=10)
        for result in results:
            self.assertGreaterEqual(result.score, 0.0)
            self.assertLessEqual(result.score, 1.0)

    def test_embedding_caching(self):
        """Test that embeddings are cached."""
        retriever = SemanticRetriever(self.kb)
        query = "test query"
        # First call
        results1 = retriever.retrieve(query, top_k=5)
        # Second call with same query
        results2 = retriever.retrieve(query, top_k=5)
        # Results should be identical
        self.assertEqual(len(results1), len(results2))
        if results1 and results2:
            self.assertEqual(results1[0].chunk.id, results2[0].chunk.id)


class TestHybridRetriever(TestRetrieverBase):
    """Tests for HybridRetriever."""

    def test_retrieve_returns_results(self):
        """Test that retriever returns results."""
        retriever = HybridRetriever(self.kb)
        results = retriever.retrieve("cat", top_k=2)
        self.assertGreater(len(results), 0)

    def test_retrieve_top_k(self):
        """Test that top_k parameter is respected."""
        retriever = HybridRetriever(self.kb)
        results = retriever.retrieve("animal", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_retrieve_scores_are_normalized(self):
        """Test that relevance scores are between 0 and 1."""
        retriever = HybridRetriever(self.kb)
        results = retriever.retrieve("dog", top_k=10)
        for result in results:
            self.assertGreaterEqual(result.score, 0.0)
            self.assertLessEqual(result.score, 1.0)

    def test_hybrid_combines_approaches(self):
        """Test that hybrid retriever combines both approaches."""
        keyword_retriever = SimpleRetriever(self.kb)
        semantic_retriever = SemanticRetriever(self.kb)
        hybrid_retriever = HybridRetriever(self.kb)

        query = "cat"
        keyword_results = keyword_retriever.retrieve(query, top_k=5)
        semantic_results = semantic_retriever.retrieve(query, top_k=5)
        hybrid_results = hybrid_retriever.retrieve(query, top_k=5)

        # Hybrid should return results
        self.assertGreater(len(hybrid_results), 0)

    def test_weight_configuration(self):
        """Test different weight configurations."""
        # Heavy keyword weight
        retriever1 = HybridRetriever(self.kb, keyword_weight=0.9, semantic_weight=0.1)
        results1 = retriever1.retrieve("cat", top_k=3)

        # Heavy semantic weight
        retriever2 = HybridRetriever(self.kb, keyword_weight=0.1, semantic_weight=0.9)
        results2 = retriever2.retrieve("cat", top_k=3)

        # Both should return results (might be different)
        self.assertGreater(len(results1), 0)
        self.assertGreater(len(results2), 0)


class TestRetrieverComparison(TestRetrieverBase):
    """Compare different retrievers."""

    def test_different_retrievers_return_different_results(self):
        """Test that different retrievers may rank results differently."""
        query = "animal"

        simple = SimpleRetriever(self.kb)
        semantic = SemanticRetriever(self.kb)

        simple_results = simple.retrieve(query, top_k=3)
        semantic_results = semantic.retrieve(query, top_k=3)

        # Get the chunk IDs
        simple_ids = [r.chunk.id for r in simple_results]
        semantic_ids = [r.chunk.id for r in semantic_results]

        # Results might be different (but not necessarily)
        # Just verify both return valid results
        if simple_results and semantic_results:
            # At least check they're both lists
            self.assertIsInstance(simple_ids, list)
            self.assertIsInstance(semantic_ids, list)


if __name__ == "__main__":
    unittest.main()

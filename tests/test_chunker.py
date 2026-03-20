"""Tests for document chunking."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import unittest
from agentic_rag import Document, FixedSizeChunker, SentenceChunker, ParagraphChunker
from agentic_rag.chunker import RecursiveChunker


class TestFixedSizeChunker(unittest.TestCase):
    """Tests for FixedSizeChunker."""

    def setUp(self):
        """Set up test fixtures."""
        self.chunker = FixedSizeChunker(by="words")
        self.doc = Document(
            id="test_doc",
            content="This is a test document. It contains multiple sentences. "
            "We want to chunk it into smaller pieces.",
        )

    def test_chunk_by_words(self):
        """Test chunking by words."""
        chunks = self.chunker.chunk(self.doc, chunk_size=5, overlap=1)
        self.assertGreater(len(chunks), 0)
        self.assertTrue(all(chunk.source_doc_id == "test_doc" for chunk in chunks))

    def test_chunk_by_chars(self):
        """Test chunking by characters."""
        char_chunker = FixedSizeChunker(by="chars")
        chunks = char_chunker.chunk(self.doc, chunk_size=50, overlap=10)
        self.assertGreater(len(chunks), 0)

    def test_chunk_ids_unique(self):
        """Test that chunk IDs are unique."""
        chunks = self.chunker.chunk(self.doc, chunk_size=5, overlap=1)
        chunk_ids = [chunk.id for chunk in chunks]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))

    def test_overlap(self):
        """Test that overlap works correctly."""
        chunks_no_overlap = self.chunker.chunk(self.doc, chunk_size=5, overlap=0)
        chunks_overlap = self.chunker.chunk(self.doc, chunk_size=5, overlap=2)
        # With overlap, we expect more chunks
        self.assertGreater(len(chunks_overlap), len(chunks_no_overlap))


class TestSentenceChunker(unittest.TestCase):
    """Tests for SentenceChunker."""

    def setUp(self):
        """Set up test fixtures."""
        self.chunker = SentenceChunker()
        self.doc = Document(
            id="test_doc",
            content="This is the first sentence. This is the second sentence. "
            "This is the third sentence. This is the fourth sentence.",
        )

    def test_chunking_respects_sentences(self):
        """Test that chunks respect sentence boundaries."""
        chunks = self.chunker.chunk(self.doc, chunk_size=2, overlap=0)
        self.assertGreater(len(chunks), 0)
        # Check that chunks contain complete sentences
        for chunk in chunks:
            # Should end with a period or contain multiple sentences
            self.assertIn(".", chunk.content)

    def test_chunk_count(self):
        """Test expected number of chunks."""
        chunks = self.chunker.chunk(self.doc, chunk_size=2, overlap=0)
        # 4 sentences with chunk_size=2 should give us 2 chunks
        self.assertEqual(len(chunks), 2)


class TestParagraphChunker(unittest.TestCase):
    """Tests for ParagraphChunker."""

    def setUp(self):
        """Set up test fixtures."""
        self.chunker = ParagraphChunker()
        self.doc = Document(
            id="test_doc",
            content="First paragraph.\n\nSecond paragraph.\n\nThird paragraph.",
        )

    def test_chunking_respects_paragraphs(self):
        """Test that chunks respect paragraph boundaries."""
        chunks = self.chunker.chunk(self.doc, chunk_size=1, overlap=0)
        self.assertEqual(len(chunks), 3)

    def test_multiple_paragraphs_per_chunk(self):
        """Test combining multiple paragraphs."""
        chunks = self.chunker.chunk(self.doc, chunk_size=2, overlap=0)
        self.assertEqual(len(chunks), 2)


class TestRecursiveChunker(unittest.TestCase):
    """Tests for RecursiveChunker."""

    def setUp(self):
        """Set up test fixtures."""
        self.chunker = RecursiveChunker()
        self.doc = Document(
            id="test_doc",
            content="First paragraph with multiple words.\n\n"
            "Second paragraph with multiple words.\n\n"
            "Third paragraph with multiple words.",
        )

    def test_recursive_chunking(self):
       """Test recursive chunking."""
        chunks = self.chunker.chunk(self.doc, chunk_size=50, overlap=10)
        self.assertGreater(len(chunks), 0)
        # Check that content is preserved
        combined_content = " ".join(chunk.content for chunk in chunks)
       # Most of the original content should be present
        self.assertIn("First paragraph", combined_content)
        self.assertIn("Second paragraph", combined_content)

    def test_metadata_preservation(self):
        """Test that metadata is preserved during chunking."""
        doc = Document(
            id="test_doc",
            content="Test content with metadata.",
            metadata={"$*ource": "test_source", "author": "test_author"},
        )
        chunks = self.chunker.chunk(doc, chunk_size=100, overlap=10)
        self.assertGreater(len(chunks), 0)
        for chunk in chunks:
            self.assertEqual(chunk.metadata["source"], "test_source")
            self.assertEqual(chunk.metadata["author"], "test_author")


class TestChunkerEdgeCases(unittest.TestCase):
    """Test edge cases for chunkers."""

    def test_empty_content_raises(self):
        """Test that empty content raises error."""
        with self.assertRaises(ValueError):
            Document(id="test", content="")

    def test_single_word_document(self):
        """Test chunking a single word document."""
        doc = Document(id="test", content="Hello")
        chunker = FixedSizeChunker(by="words")
        chunks = chunker.chunk(doc, chunk_size=10, overlap=1)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].content, "Hello")

    def test_very_long_document(self):
        """Test chunking a very long document."""
        long_content = " ".join(["word"] * 10000)
        doc = Document(id="test", content=long_content)
        chunker = FixedSizeChunker(by="words")
        chunks = chunker.chunk(doc, chunk_size=100, overlap=10)
        self.assertGreater(len(chunks), 50)  # Should create many chunks


if __name__ == "__main__":
    unittest.main()

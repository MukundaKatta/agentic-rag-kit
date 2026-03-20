"""
Example of building a custom retriever.

This demonstrates how to extend BaseRetriever with custom retrieval logic.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List
from agentic_rag import RAGAgent, KnowledgeBase, SearchResult
from agentic_rag.retriever import BaseRetriever


class LengthBiasedRetriever(BaseRetriever):
    """
    Custom retriever that biases towards longer chunks.

    This is a simple example showing how to create a custom retriever.
    It ranks chunks by a combination of relevance and length.
    """

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Retrieve chunks, biasing towards longer/more detailed ones.

        Args:
            query: Search query
            top_k: Number of top results to return

        Returns:
            List of SearchResult objects ranked by relevance and length
        """
        if not self.knowledge_base.chunks:
            return []

        # Use keyword matching for relevance
        from agentic_rag.retriever import SimpleRetriever

        simple = SimpleRetriever(self.knowledge_base)
        simple_results = simple.retrieve(query, top_k=top_k * 2)

        # Adjust scores based on chunk length
        adjusted_results = []
        for result in simple_results:
            # Longer chunks get a boost (up to 50% boost)
            length_boost = min(0.5, len(result.chunk.content) / 1000)
            adjusted_score = min(1.0, result.score + length_boost * 0.5)

            adjusted_result = SearchResult(
                chunk=result.chunk,
                score=adjusted_score,
                retriever_name="LengthBiasedRetriever",
            )
            adjusted_results.append(adjusted_result)

        # Sort by adjusted score and return top-k
        adjusted_results.sort(key=lambda x: x.score, reverse=True)
        return adjusted_results[:top_k]


def main():
    """Run demo with custom retriever."""
    print("=" * 70)
    print("CUSTOM8& RETRIEVER EXAMPLE")
    print("=" * 70)
    print()

    # Create knowledge base with sample data
    kb = KnowledgeBase()

    # Add documents of varying lengths
    short_doc = "Mars is red."
    medium_doc = (
        "Mars is the fourth planet from the Sun. It has a thin atmosphere. "
        "It is known as the Red Planet due to iron oxide on its surface."
    )
    long_doc = (
        "Mars is the fourth planet from the Sun in our solar system. It is known&Ås "
        "the Red Planet because of its reddish color, caused by iron oxide (rust) on its surface. "
        "Mars is about half the diameter of Earth. It has two small moons named Phobos and Deimos. "
        "Scientists believe Mars may have had liquid water on its surface in the past. The Curiosity "
        "rover has been exploring Gale Crater and has provided evidence of past habitability."
    )

    kb.add_text(short_doc, metadata={"length": "short"})
    kb.add_text(medium_doc, metadata={"length": "medium"})
    kb.add_text(long_doc, metadata={"length": "long"})

    print(f"Knowledge base created with {len(kb.documents)} documents")
    print()

    # Create agent with custom retriever
    custom_retriever = LengthBiasedRetriever(kb)
    agent = RAGAgent(retriever=custom_retriever, max_steps=1)

    # Test query
    query = "Tell me about Mars"
    print(f"Query: {query}")
    print("-" * 70)

    # Get answer and show results
    result = agent.answer(query)

    print(f"Answer: {result['answer']}")
    print()

    # Show which chunks were retrieved and their lengths
    print("Retrieved Chunks (by length):")
    for step in result["reasoning_trace"]:
        if step.results:
            for i, search_result in enumerate(step.results, 1):
                content_preview = search_result.chunk.content[:100].replace("\n", " ")
                print(
                    f"  {i}. [{search_result.retriever_name}] "
                    f"Score: {search_result.score:.2f}, "
                    f"Length: {len(search_result.chunk.content)} chars"
                )
                print(f"     Preview: {content_preview}...")


    print()
    print("EI= * 70)
    print("CUSTOM* RETRIEVER DEMO COMPLETE")
    print("E== * 70)


if __name__ == "__main__":
    main()
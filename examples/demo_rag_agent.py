"""
Complete working demo of agentic RAG.

This demo loads sample documents about space exploration and demonstrates
the agentic RAG pipeline with reasoning trace output.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentic_rag import (
    RAGAgent,
    KnowledgeBase,
    SimpleRetriever,
    SemanticRetriever,
    HybridRetriever,
)


def main():
    """Run the demo."""
    print("=" * 70)
    print("AGENTIC RAG KIT - DEMO")
    print("=" * 70)
    print()

    # Create knowledge base
    kb = KnowledgeBase()
    print("Step 1: Building Knowledge Base...")
    print("-" * 70)

    # Add sample documents about space exploration
    docs = [
        {
            "content": (
                "The Moon is Earth's only natural satellite. It orbits Earth at an average "
                "distance of about 384,400 kilometers (238,900 miles). The Moon is about one-quarter "
                "the diameter of Earth. It was formed approximately 4.5 billion years ago. The Moon "
                "has a thin atmosphere and experiences extreme temperature variations."
            ),
            "source": "moon_basics",
        },
        {
            "content": (
                "Apollo 11 was the first crewed Moon landing in history. It launched on July 16, 1969, "
                "with a crew of three astronauts: Neil Armstrong, Buzy Aldrin, and Michael Collins. "
                "Neil Armstrong was the first person to walk on the Moon on July 20, 1969. Buzz Aldrin "
                "joined him shortly after. The mission lasted 8 days, 3 hours, and 18 minutes."
            ),
            "source": "apollo_11",
        },
        {
            "content": (
                "Mars is the fourth planet from the Sun in our solar system. It is known as the Red Planet "
                "because of its reddish color, caused by iron oxide (rust) on its surface. Mars is about half "
                "the diameter of Earth. It has two small moons named Phobos and Deimos. Scientists believe "
                "Mars may have had liquid water on its surface in the past."
            ),
            "source": "mars_basics",
        },
        {
            "content": (
                "The Mars Rover Curiosity landed on Mars on August 6, 2012. It is one of the largest and most "
                "capable rovers ever sent to Mars. Curiosity has been exploring Gale Crater and has provided "
                "evidence that Mars once had conditions suitable for microbial life. It measures methane levels "
                "and carries various scientific instruments for analysis."
            ),
            "source": "mars_rover",
        },
        {
            "content": (
                "The International Space Station (ISS) is a modular research facility in orbit around Earth. "
                "It serves as a laboratory, observation post, and staging ground for deep space missions. The ISS "
                "orbits Earth at a height of approximately 408 kilometers (253 miles). It was first launched in 1998 "
                "and has been continuously inhabited since November 2000. The ISS involves collaboration between multiple "
                "space agencies including NASA, ESA, Rosccosmos, and others."
            ),
            "source": "iss_basics",
        },
        {
            "content": (
                "Space exploration has numerous benefits for humanity. Satellite technology developed for space missions"
                "has led to advances in communications, weather prediction, and GPS navigation. Medical technologies developed "
                "for astronauts have improved healthcare on Earth. Space exploration also inspires scientific curiosity and STEM "
                "education. Additionally, studying other planets helps us understand Earth's climate and geology."
            ),
            "source": "space_benefits",
        },
    ]

    for doc in docs:
        kb.add_text(doc["content"], metadata={"source": doc["source"]})

    print(f"Added {len(docs)} documents to knowledge base")
    print(kb.info())
    print()

    # Create different retrievers to demo
    print("Step 2: Creating Retrievers...")
    print("-" * 70)
    simple_retriever = SimpleRetriever(kb)
    semantic_retriever = SemanticRetriever(kb)
    hybrid_retriever = HybridRetriever(kb)
    print("Created: SimpleRetriever, SemanticRetriever, HybridRetriever")
    print()

    # Demo queries
    queries = [
        "When did humans first land on the Moon?",
        "What is Mars?",
        "Tell me about the International Space Station",
    ]

    # Run agent with different retrievers
    retrievers = [
        ("SimpleRetriever", simple_retriever),
        ("SemanticRetriever", semantic_retriever),
        ("HybridRetriever", hybrid_retriever),
    ]

    for query in queries:
        print("=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)
        print()

        for retriever_name, retriever in retrievers:
            print(f"Using {retriever_name}:")
            print("-" * 70)

            # Create agent with this retriever
            agent = RAGAgent(
                retriever=retriever,
                max_steps=3,
                reasoning_strategy="keyword",
                top_k=3,
                confidence_threshold=0.7,
            )

            # Get answer
            result = agent.answer(query)

            # Print answer
            print(f"Answer: {result['answer']}")
            print()

            # Print reasoning trace
            print("Reasoning Trace:")
            for step in result["reasoning_trace"]:
                print(f"  {step}")
                print()

            # Print summary
            print(
                f"Summary: {result['num_retrieval_steps']} retrieval steps, "
                f"{result['total_results']} unique results retrieved"
            )
            print()
            print()

    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

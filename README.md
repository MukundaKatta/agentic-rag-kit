# agentic-rag-kit

**Intelligent document retrieval meets agentic reasoning.**

A lightweight Python framework for building agentic RAG (Retrieval-Augmented Generation) pipelines where AI agents reason about what to retrieve, iteratively refine searches, and synthesize answers from multiple sources.

## What is Agentic RAG?

Traditional RAG systems retrieve documents and immediately generate answers. Agentic RAG goes further by enabling an AI agent to:

- **Reason** about what information is needed
- **Iterate** through multi-step retrieval, refining queries based on intermediate results
- **Decide** when it has sufficient information to answer
- **Synthesize** comprehensive answers from diverse sources

This toolkit provides the building blocks to implement agentic RAG pipelines without external LLM dependencies, using rule-based reasoning for demonstrations and easy customization.

## Features

- **Agentic Agent**: Core agent that reasons about retrieval, maintains chain-of-thought, and synthesizes answers
- **Multiple Retrievers**: Keyword-based, semantic (embeddings), and hybrid search
- **Document Chunking**: Fixed-size, sentence-based, paragraph-based, and recursive strategies
- **Knowledge Base**: Easy document management and indexing
- **Zero External Dependencies**: Runs standalone without LLM API calls
- **Extensible**: Build custom retrievers and reasoning strategies
- **Production-Ready**: Clean API, well-structured, tested components

## Installation

### From Source

```bash
git clone https://github.com/yourusername/agentic-rag-kit.git
cd agentic-rag-kit
pip install -e .
```

### From PyPI (coming soon)

```bash
pip install agentic-rag-kit
```

## Quick Start

### Basic Usage

```python
from agentic_rag import RAGAgent, KnowledgeBase, SimpleRetriever
from agentic_rag.chunker import RecursiveChunker

# Create a knowledge base
kb = KnowledgeBase()

# Add documents
kb.add_text(
    "The Moon is Earth's only natural satellite. It is about 384,400 km away.",
    metadata={"source": "astronomy_101"}
)
kb.add_text(
    "Apollo 11 landed on the Moon on July 20, 1969, with Neil Armstrong as the first person to walk on the lunar surface.",
    metadata={"source": "history"}
)

# Create retriever and agent
retriever = SimpleRetriever(kb)
agent = RAGAgent(
    retriever=retriever,
    max_steps=3,
    reasoning_strategy="keyword"
)

# Ask a question
query = "When did humans first land on the Moon?"
result = agent.answer(query)

print(f"Question: {query}")
print(f"Answer: {result['answer']}")
print(f"Reasoning Steps: {len(result['reasoning_trace'])} steps")
```

### Advanced: Custom Retriever

```python
from agentic_rag.retriever import BaseRetriever

class CustomRetriever(BaseRetriever):
    def retrieve(self, query: str, top_k: int = 5):
        # Your custom retrieval logic here
        return search_results

# Use in agent
retriever = CustomRetriever(kb)
agent = RAGAgent(retriever=retriever)
```

### Advanced: Semantic Retrieval

```python
from agentic_rag import SemanticRetriever, RAGAgent

# Semantic retrieval uses simple embeddings (no external APIs)
retriever = SemanticRetriever(kb)
agent = RAGAgent(retriever=retriever)

result = agent.answer("What is the distance to the Moon?")
```

## Architecture

```
agentic-rag-kit/
|
+-- agentic_rag/
|   +-- agent.py          # Core RAG agent with reasoning
|   +-- retriever.py      # Retriever implementations
|   +-- chunker.py        # Document chunking strategies
|   +-- document.py       # Data models
|   +-- knowledge_base.py # Knowledge base management
|   +-- __init__.py       # Package exports
|
+-- examples/
|   +-- demo_rag_agent.py      # Complete working example
|   +-- custom_retriever.py    # Custom retriever example
|
+-- tests/
|   +-- test_chunker.py        # Chunking tests
|   +-- test_retriever.py      # Retriever tests
|
+-- requirements.txt            # Dependencies
+-- setup.py                    # Package configuration
+-- LICENSE                     # MIT License
+-- README.md                   # This file
```

## How It Works

### 1. Document Preparation

Documents are chunked using your chosen strategy:
- **FixedSizeChunker**: Splits by word/character count
- **SentenceChunker**: Respects sentence boundaries
- **ParagraphChunker**: Preserves paragraphs
- **RecursiveChunker**: Hierarchical chunking

### 2. Retrieval

Choose a retrieval strategy:
- **SimpleRetriever**: TF-IDF style keyword matching
- **SemanticRetriever**: Embedding-based similarity
- **HybridRetriever**: Combines both approaches

### 3. Agentic Reasoning

The RAG Agent:

1. Analyzes the user query
2. Decides what to retrieve (query decomposition)
3. Retrieves relevant chunks
4. Evaluates if it has enough information
5. Refines retrieval if needed (iterative)
6. Synthesizes final answer from all sources

Each step maintains a reasoning trace showing the agent's thought process.

## Examples

Run the complete demo:

```bash
python examples/demo_rag_agent.py
```

This loads sample documents about space exploration and demonstrates the agentic RAG pipeline with reasoning trace output.

## Configuration

Configure agent behavior:

```python
agent = RAGAgent(
    retriever=retriever,
    max_steps=5,                    # Max retrieval iterations
    reasoning_strategy="keyword",   # "keyword" or "hybrid"
    top_k=5,                        # Results per retrieval
    confidence_threshold=0.7,       # Stop if confident enough
)
```

## Contributing

Contributions are welcome! Areas for contribution:

- Additional retriever implementations
- Improved reasoning strategies
- Performance optimizations
- Better documentation
- More examples

Please feel free to open issues and pull requests.

## License

MIT License - See LICENSE file for details

Copyright (c) 2026 Mukunda Katta

## Citation

If you use agentic-rag-kit in your research, please cite:

```
@software{agentic_rag_kit,
  title = {agentic-rag-kit: Agentic Retrieval-Augmented Generation Framework},
  author = {Katta, Mukunda},
  year = {2026},
  url = {https://github.com/yourusername/agentic-rag-kit}
}
```

## Roadmap

- [ ] Integration with open-source LLMs for reasoning
- [ ] Support for vector databases (Chroma, Pinecone, Weaviate)
- [ ] Advanced reasoning strategies (COT, tree-of-thought)
- [ ] Multi-modal document support
- [ ] Web UI for pipeline visualization
- [ ] Performance benchmarking suite

---

Built with care for the agentic AI era.

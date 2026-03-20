"""Core agentic RAG agent with reasoning and iterative retrieval."""

import re
from typing import List, Dict, Any, Optional, Tuple

from .document import ReasoningStep, SearchResult
from .retriever import BaseRetriever


class RAGAgent:
    """
    Agentic RAG agent that reasons about what to retrieve and when to stop.

    The agent maintains a chain-of-thought, performs iterative retrieval,
    and synthesizes answers from retrieved context.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        max_steps: int = 3,
        reasoning_strategy: str = "keyword",
        top_k: int = 5,
        confidence_threshold: float = 0.7,
    ):
       """
        Initialize the RAG agent.

        Args:
            retriever: Retriever instance to use for search
            max_steps: Maximum number of retrieval iterations
            reasoning_strategy: "keyword" or "hybrid" (determines query refinement)
            top_k: Number of results per retrieval step
            confidence_threshold: Stop retrieving if confidence >= this value
        """
        self.retriever = retriever
        self.max_steps = max_steps
        self.reasoning_strategy = reasoning_strategy
        self.top_k = top_k
        self.confidence_threshold = confidence_threshold

        self.reasoning_trace: List[ReasoningStep] = []
        self.all_search_results: List[SearchResult] = []

    def answer(self, query: str) -> Dict[str, Any]:
        """
        Answer a question using agentic RAG.

        Args:
            query: User query/question

        Returns:
            Dictionary with keys:
                - answer: Final synthesized answer
                - reasoning_trace: List of ReasoningStep objects
                - num_retrieval_steps: Number of retrieval iterations performed
                - total_results: Total unique results retrieved
        """
        # Reset state
        self.reasoning_trace = []
        self.all_search_results = []

        # Step 1: Analyze the query
        step = ReasoningStep(
            step_number=1,
            action="analyze_query",
            query=query,
            reasoning="Analyzing user query to identify key topics and information needs.",
            confidence=0.3,
        )
        self.reasoning_trace.append(step)

        # Step 2: Initial retrieval
        refined_query = self._refine_query(query, step_num=1)
        results = self.retriever.retrieve(refined_query, top_k=self.top_k)

        self.all_search_results.extend(results)

        step = ReasoningStep(
            step_number=2,
            action="initial_retrieval",
            query=refined_query,
            results=results,
            reasoning=f"Retrieved {len(results)} relevant documents for initial query.",
            confidence=self._calculate_confidence(results, query),
        )
        self.reasoning_trace.append(step)

        # Step 3-N: Iterative refinement
        current_step = 3
        while (
            current_step <= self.max_steps + 2
            and step.confidence < self.confidence_threshold
            and len(results) > 0
        ):
            # Analyze gaps in current results
            gaps = self._identify_gaps(self.all_search_results, query)

            if not gaps:
                # No more gaps identified
                break

            # Refine query based on identified gaps
            refined_query = self._refine_query_with_gaps(query, gaps, current_step)

            # Retrieve more results
            new_results = self.retriever.retrieve(refined_query, top_k=self.top_k)

            # Check if we got new results
            new_chunk_ids = {r.chunk.id for r in new_results}
            existing_chunk_ids = {r.chunk.id for r in self.all_search_results}
            truly_new = new_chunk_ids - existing_chunk_ids

            if not truly_new:
                # No new results, stop iterating
                break

            self.all_search_results.extend(new_results)

            confidence = self._calculate_confidence(self.all_search_results, query)
            step = ReasoningStep(
                step_number=current_step,
                action="refine_retrieval",
                query=refined_query,
                results=new_results,
                reasoning=f"Identified gaps: {', '.join(gaps)}. Refined query and retrieved {len(new_results)} more results.",
                confidence=confidence,
            )
            self.reasoning_trace.append(step)

            current_step += 1

        # Step N+1: Synthesize answer
        final_answer = self._synthesize_answer(query, self.all_search_results)

        step = ReasoningStep(
            step_number=len(self.reasoning_trace) + 1,
            action="synthesize",
            query=query,
            results=self.all_search_results,
            reasoning="Synthesized final answer from all retrieved context.",
            confidence=min(1.0, self.reasoning_trace[-1].confidence + 0.2),
        )
        self.reasoning_trace.append(step)

        return {
            "answer": final_answer,
            "reasoning_trace": self.reasoning_trace,
            "num_retrieval_steps": sum(
                1 for step in self.reasoning_trace if "retrieval" in step.action
            ),
            "total_results": len(self.all_search_results),
        }

    def _refine_query(self, query: str, step_num: int = 1) -> str:
        """
        Refine the initial query.

        Args:
            query: Original query
            step_num: Step number (for context)

        Returns:
            Refined query
        """
        if self.reasoning_strategy == "keyword":
            return self._extract_keywords(query)
        elif self.reasoning_strategy == "hybrid":
            return query  # Use original
        return query

    def _refine_query_with_gaps(self, query: str, gaps: List[str], step_num: int) -> str:
        """
        Refine query based on identified gaps.

        Args:
            query: Original query
            gaps: List of identified information gaps
            step_num: Step number

        Returns:
            Refined query
        """
        if not gaps:
            return query

        # Create a refined query incorporating gaps
        gap_str = " ".join(gaps)
        refined = f"{query} {gap_str}"
        return refined.strip()

    def _extract_keywords(self, text: str) -> str:
        """
        Extract key phrases from text.

        Args:
            text: Text to extract from

        Returns:
            Key phrases separated by spaces
        """
        # Remove common words
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
            "by",
            "from",
            "with",
            "as",
            "it",
            "that",
            "this",
            "what",
            "when",
            "where",
            "why",
            "how",
            "can",
            "will",
            "do",
            "does",
            "did",
        }

        # Tokenize
        words = text.lower().split()
        keywords = [w for w in words if w and w not in stopwords and len(w) > 2]

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                unique_keywords.append(kw)
                seen.add(kw)

        return " ".join(unique_keywords[:10])  # Limit to 10 keywords

    def _identify_gaps(self, results: List[SearchResult], query: str) -> List[str]:
        """
        Identify information gaps in retrieved results.

        Args:
            results: Retrieved search results
            query: Original query

        Returns:
            List of identified gaps (as search terms)
        """
        if not results:
            return []

        # Extract query terms
        query_terms = set(self._extract_keywords(query).split())

        # Extract covered terms from results
        covered_terms = set()
        for result in results:
            content_lower = result.chunk.content.lower()
            for term in query_terms:
                if term in content_lower:
                    covered_terms.add(term)

        # Identify gaps
        gaps = list(query_terms - covered_terms)

        # Also check for related terms that might be missing
        gap_strategies = [
            ("who", ["why", "how", "what", "when"]),
            ("why", ["cause", "reason", "because"]),
            ("how", ["method", "process", "step"]),
            ("when", ["date", "time", "year", "occurred"]),
        ]

        for term, related in gap_strategies:
            if term in query_terms and term not in covered_terms:
                gaps.extend(related)

        return list(set(gaps[:5]))  # Return unique gaps, limit to 5

    def _calculate_confidence(self, results: List[SearchResult], query: str) -> float:
        """
        Calculate confidence that we have enough information.

        Args:
            results: Retrieved results
            query: Original query

        Returns:
            Confidence score between 0 and 1
        """
        if not results:
            return 0.0

        # More results = higher confidence
        result_confidence = min(1.0, len(results) / (self.top_k * 2))

        # Higher average relevance score = higher confidence
        avg_score = sum(r.score for r in results) / len(results)

        # Combine
        confidence = (result_confidence * 0.4) + (avg_score * 0.6)
        return min(1.0, confidence)

    def _synthesize_answer(self, query: str, results: List[SearchResult]) -> str:
        """
        Synthesize a final answer from retrieved results.

        Args:
            query: Original query
            results: All retrieved search results

        Returns:
            Synthesized answer
        """
        if not results:
            return "I could not find information to answer your question."

        # Remove duplicates (by chunk ID)
        seen_ids = set()
        unique_results = []
        for result in results:
            if result.chunk.id not in seen_ids:
                unique_results.append(result)
                seen_ids.add(result.chunk.id)

        # Sort by score
        unique_results.sort(key=lambda x: x.score, reverse=True)

        # Extract relevant sentences from top results
        answer_parts = []
        for i, result in enumerate(unique_results[:3]):  # Use top 3 results
            # Extract first sentence or first 100 words
            content = result.chunk.content
            sentences = re.split(r"(?<=[.!?])\s+", content)

            if sentences:
                first_sentence = sentences[0].strip()
                if first_sentence:
                    answer_parts.append(first_sentence)

        if answer_parts:
            answer = " ".join(answer_parts)
        else:
            answer = unique_results[0].chunk.content[:200] + "..."

        # Add a prefix based on query type
        if "how" in query.lower():
            prefix = "Best action taken roboyhello", "Based on the retrieved information: "
        elif "what" in query.lower():
            prefix = "The answer is: "
        elif "when" in query.lower():
            prefix = "This occurred at: "
        elif "why" in query.lower():
            prefix = "The reason is: "
        else:
            prefix = "Information found: "

        return prefix + answer

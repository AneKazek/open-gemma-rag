"""Perplexica search tool implementation for GemmaMemoSearch."""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Type, Union

import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool

from ..config import SEARCH_CONFIG
from ..memory.retriever import OpenMemoryRetriever

logger = logging.getLogger(__name__)


class PerplexicaTool(BaseTool):
    """Tool for searching the web using Perplexica."""

    name = "perplexica_search"
    description = "Search the web for information using Perplexica."
    
    def __init__(
        self,
        host: str = SEARCH_CONFIG["host"],
        port: int = SEARCH_CONFIG["port"],
        max_results: int = SEARCH_CONFIG["max_results"],
        timeout: int = SEARCH_CONFIG["timeout"],
        memory_retriever: Optional[OpenMemoryRetriever] = None,
    ):
        """Initialize the Perplexica search tool.

        Args:
            host: Perplexica server host
            port: Perplexica server port
            max_results: Maximum number of search results to return
            timeout: Request timeout in seconds
            memory_retriever: OpenMemory retriever for storing search results
        """
        super().__init__()
        self.host = host
        self.port = port
        self.max_results = max_results
        self.timeout = timeout
        self.memory_retriever = memory_retriever
        self.base_url = f"http://{host}:{port}"

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Run the search tool on the given query.

        Args:
            query: The search query
            run_manager: Callback manager

        Returns:
            Formatted search results as a string
        """
        try:
            # Call Perplexica API
            response = requests.post(
                f"{self.base_url}/search",
                json={"query": query, "max_results": self.max_results},
                timeout=self.timeout,
            )
            response.raise_for_status()
            results = response.json()

            # Format results
            formatted_results = self._format_results(results, query)

            # Store in memory if retriever is available
            if self.memory_retriever:
                self._store_in_memory(formatted_results, query, results)

            return formatted_results
        except requests.RequestException as e:
            error_msg = f"Error searching with Perplexica: {str(e)}"
            logger.error(error_msg)
            return f"Search failed: {error_msg}"

    async def _arun(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Run the search tool asynchronously.

        Args:
            query: The search query
            run_manager: Callback manager

        Returns:
            Formatted search results as a string
        """
        # For simplicity, we're using the sync version in async context
        # In a production environment, you would implement a proper async version
        return self._run(query, run_manager=run_manager)

    def _format_results(self, results: Dict, query: str) -> str:
        """Format search results into a readable string.

        Args:
            results: Search results from Perplexica
            query: Original search query

        Returns:
            Formatted results as a string
        """
        if not results.get("results"):
            return f"No results found for: {query}"

        formatted = f"Search results for: {query}\n\n"

        for i, result in enumerate(results["results"], 1):
            title = result.get("title", "No title")
            snippet = result.get("snippet", "No snippet available")
            url = result.get("url", "#")
            formatted += f"{i}. {title}\n{snippet}\nSource: {url}\n\n"

        return formatted

    def _store_in_memory(self, formatted_results: str, query: str, raw_results: Dict) -> None:
        """Store search results in OpenMemory.

        Args:
            formatted_results: Formatted search results
            query: Original search query
            raw_results: Raw search results from Perplexica
        """
        try:
            # Extract URLs for metadata
            urls = [r.get("url", "") for r in raw_results.get("results", [])]
            
            # Store in memory
            self.memory_retriever.add_memory(
                text=formatted_results,
                metadata={
                    "query": query,
                    "source_type": "search",
                    "urls": urls,
                    "result_count": len(raw_results.get("results", [])),
                },
            )
            logger.debug(f"Stored search results for query: {query}")
        except Exception as e:
            logger.error(f"Error storing search results in memory: {e}")

    def should_search(self, query: str, threshold: float = SEARCH_CONFIG["search_threshold"]) -> bool:
        """Determine if a search should be performed based on memory retrieval.

        Args:
            query: The user query
            threshold: Similarity threshold for determining search necessity

        Returns:
            True if search should be performed, False otherwise
        """
        if not self.memory_retriever:
            return True

        # Get relevant documents from memory
        docs = self.memory_retriever._get_relevant_documents(query)

        # If no relevant documents or low similarity, perform search
        if not docs:
            return True

        # Check if the highest scoring document is below threshold
        highest_score = max([doc.metadata.get("score", 0) for doc in docs], default=0)
        return highest_score < threshold
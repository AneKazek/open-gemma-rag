"""RAG chain implementation for GemmaMemoSearch."""

import logging
from typing import Any, Dict, List, Optional, Union

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

from ..llm.ollama import GemmaLLM
from ..memory.retriever import OpenMemoryRetriever
from ..search.tool import PerplexicaTool
from .prompts import COMBINED_PROMPT, RAG_PROMPT, SEARCH_DECISION_PROMPT, SEARCH_SUMMARY_PROMPT

logger = logging.getLogger(__name__)


class GemmaMemoSearchChain:
    """Main RAG chain for GemmaMemoSearch."""

    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        memory_retriever: Optional[OpenMemoryRetriever] = None,
        search_tool: Optional[PerplexicaTool] = None,
    ):
        """Initialize the GemmaMemoSearch chain.

        Args:
            llm: Language model to use
            memory_retriever: OpenMemory retriever
            search_tool: Perplexica search tool
        """
        # Initialize components if not provided
        self.llm = llm or GemmaLLM()
        self.memory_retriever = memory_retriever or OpenMemoryRetriever()
        self.search_tool = search_tool or PerplexicaTool(memory_retriever=self.memory_retriever)
        
        # Initialize chat history
        self.chat_history = []
        
        # Build the chain
        self._build_chain()

    def _build_chain(self) -> None:
        """Build the RAG chain with memory retrieval and search capabilities."""
        # Create the search decision chain
        self.search_decision_chain = SEARCH_DECISION_PROMPT | self.llm | StrOutputParser()
        
        # Create the search summary chain
        self.search_summary_chain = SEARCH_SUMMARY_PROMPT | self.llm | StrOutputParser()
        
        # Create the main RAG chain
        self.chain = self._create_rag_chain()

    def _create_rag_chain(self) -> RunnableSequence:
        """Create the main RAG chain.

        Returns:
            The RAG chain as a RunnableSequence
        """
        # Define the retrieval function
        def retrieve_from_memory(query):
            docs = self.memory_retriever._get_relevant_documents(query)
            return "\n\n".join([doc.page_content for doc in docs]) if docs else "No relevant memory found."

        # Define the search function
        def search_web(query):
            # Check if search is needed
            search_decision = self.search_decision_chain.invoke({"question": query})
            
            if search_decision.startswith("SEARCH:"):
                # Extract search query
                search_query = search_decision.replace("SEARCH:", "").strip()
                logger.info(f"Performing web search for: {search_query}")
                
                # Perform search
                search_results = self.search_tool._run(search_query)
                
                # Summarize results
                summary = self.search_summary_chain.invoke({
                    "query": query,
                    "search_results": search_results
                })
                
                return summary
            
            return "No web search performed."

        # Build the chain
        return RunnableSequence(
            {
                "question": RunnablePassthrough(),
                "chat_history": lambda _: self.chat_history,
                "memory_content": lambda x: retrieve_from_memory(x["question"]),
                "search_content": lambda x: search_web(x["question"]),
            }
            | COMBINED_PROMPT
            | self.llm
            | StrOutputParser()
        )

    def invoke(self, query: str) -> str:
        """Invoke the chain with a query.

        Args:
            query: User query

        Returns:
            Response from the chain
        """
        try:
            # Invoke the chain
            response = self.chain.invoke({"question": query})
            
            # Update chat history
            self.chat_history.append(HumanMessage(content=query))
            self.chat_history.append(AIMessage(content=response))
            
            # Store interaction in memory
            self._store_interaction(query, response)
            
            return response
        except Exception as e:
            error_msg = f"Error invoking GemmaMemoSearch chain: {str(e)}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

    def _store_interaction(self, query: str, response: str) -> None:
        """Store the interaction in memory.

        Args:
            query: User query
            response: System response
        """
        try:
            # Format the interaction
            interaction = f"User: {query}\n\nAssistant: {response}"
            
            # Store in memory
            self.memory_retriever.add_memory(
                text=interaction,
                metadata={
                    "type": "interaction",
                    "query": query,
                },
                source_type="conversation",
            )
            logger.debug(f"Stored interaction in memory for query: {query[:50]}...")
        except Exception as e:
            logger.error(f"Error storing interaction in memory: {e}")

    def reset(self) -> None:
        """Reset the chat history."""
        self.chat_history = []
        logger.info("Chat history reset")
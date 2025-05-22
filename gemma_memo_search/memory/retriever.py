"""OpenMemory retriever implementation for GemmaMemoSearch."""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Union

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

import mem0  # OpenMemory SDK

from ..config import MEMORY_CONFIG

logger = logging.getLogger(__name__)


class OpenMemoryRetriever(BaseRetriever):
    """Retriever that uses OpenMemory for storing and retrieving conversation history."""

    def __init__(
        self,
        host: str = MEMORY_CONFIG["host"],
        port: int = MEMORY_CONFIG["port"],
        collection_name: str = MEMORY_CONFIG["collection_name"],
        top_k: int = MEMORY_CONFIG["top_k"],
        similarity_threshold: float = MEMORY_CONFIG["similarity_threshold"],
        similarity_metric: str = MEMORY_CONFIG["similarity_metric"],
    ):
        """Initialize the OpenMemory retriever.

        Args:
            host: OpenMemory server host
            port: OpenMemory server port
            collection_name: Name of the collection to use
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score for retrieval
            similarity_metric: Similarity metric to use (cosine, dot, euclidean)
        """
        super().__init__()
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.similarity_metric = similarity_metric

        # Initialize OpenMemory client
        self.client = mem0.Client(f"http://{host}:{port}")
        
        # Ensure collection exists
        self._initialize_collection()

    def _initialize_collection(self) -> None:
        """Initialize the OpenMemory collection if it doesn't exist."""
        try:
            collections = self.client.list_collections()
            if self.collection_name not in [c.name for c in collections]:
                logger.info(f"Creating collection: {self.collection_name}")
                self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "GemmaMemoSearch conversation history"}
                )
            logger.info(f"Using collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error initializing OpenMemory collection: {e}")
            raise

    def add_memory(
        self, 
        text: str, 
        metadata: Optional[Dict] = None,
        source_type: str = "conversation",
    ) -> str:
        """Add a memory entry to OpenMemory.

        Args:
            text: The text content to store
            metadata: Additional metadata for the entry
            source_type: Type of memory (conversation, search, etc.)

        Returns:
            ID of the created memory entry
        """
        if metadata is None:
            metadata = {}

        # Add standard metadata
        metadata.update({
            "timestamp": datetime.now().isoformat(),
            "source_type": source_type,
        })

        try:
            # Add memory to OpenMemory
            memory_id = self.client.add_memory(
                collection_name=self.collection_name,
                text=text,
                metadata=metadata,
            )
            logger.debug(f"Added memory: {memory_id}")
            return memory_id
        except Exception as e:
            logger.error(f"Error adding memory: {e}")
            raise

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to the query.

        Args:
            query: Query string
            run_manager: Callback manager

        Returns:
            List of relevant documents
        """
        try:
            # Search OpenMemory for relevant entries
            results = self.client.search(
                collection_name=self.collection_name,
                query=query,
                limit=self.top_k,
                min_score=self.similarity_threshold,
                metric=self.similarity_metric,
            )

            # Convert to LangChain documents
            documents = []
            for result in results:
                doc = Document(
                    page_content=result.text,
                    metadata={
                        "id": result.id,
                        "score": result.score,
                        **result.metadata,
                    },
                )
                documents.append(doc)

            logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
            return documents
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Asynchronously get documents relevant to the query.

        Args:
            query: Query string
            run_manager: Callback manager

        Returns:
            List of relevant documents
        """
        # For simplicity, we're using the sync version in async context
        # In a production environment, you would implement a proper async version
        return self._get_relevant_documents(query, run_manager=run_manager)
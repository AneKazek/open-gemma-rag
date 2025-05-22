#!/usr/bin/env python
"""Basic usage example for GemmaMemoSearch."""

import logging

from gemma_memo_search.llm.ollama import GemmaLLM
from gemma_memo_search.memory.retriever import OpenMemoryRetriever
from gemma_memo_search.rag.chain import GemmaMemoSearchChain
from gemma_memo_search.search.tool import PerplexicaTool

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Initialize components
print("Initializing GemmaMemoSearch components...")

# Initialize memory retriever
memory_retriever = OpenMemoryRetriever(
    top_k=5,
    similarity_threshold=0.7,
)

# Initialize search tool
search_tool = PerplexicaTool(
    memory_retriever=memory_retriever,
)

# Initialize LLM
llm = GemmaLLM(
    model="gemma:3b",  # Use gemma:7b for better performance
    temperature=0.7,
)

# Initialize RAG chain
chain = GemmaMemoSearchChain(
    llm=llm,
    memory_retriever=memory_retriever,
    search_tool=search_tool,
)

print("Initialization complete!")

# Example queries
queries = [
    "What is RAG in the context of AI?",
    "Tell me about Gemma 3 LLM",
    "How does OpenMemory work?",
    "What are the benefits of a self-hosted RAG system?",
]

# Process queries
for i, query in enumerate(queries, 1):
    print(f"\n--- Query {i}: {query} ---")
    response = chain.invoke(query)
    print(f"\nResponse:\n{response}")

print("\nExample complete!")
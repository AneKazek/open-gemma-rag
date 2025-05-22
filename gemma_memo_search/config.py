"""Configuration settings for GemmaMemoSearch."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Create data directory if it doesn't exist
DATA_DIR.mkdir(exist_ok=True)

# OpenMemory configuration
MEMORY_CONFIG = {
    "host": os.getenv("MEMORY_HOST", "localhost"),
    "port": int(os.getenv("MEMORY_PORT", "5000")),
    "collection_name": os.getenv("MEMORY_COLLECTION", "gemma_memo_search"),
    # Retrieval settings
    "top_k": int(os.getenv("MEMORY_TOP_K", "5")),
    "similarity_threshold": float(os.getenv("MEMORY_SIMILARITY_THRESHOLD", "0.7")),
    "similarity_metric": os.getenv("MEMORY_SIMILARITY_METRIC", "cosine"),
}

# Perplexica web search configuration
SEARCH_CONFIG = {
    "host": os.getenv("SEARCH_HOST", "localhost"),
    "port": int(os.getenv("SEARCH_PORT", "5001")),
    "max_results": int(os.getenv("SEARCH_MAX_RESULTS", "5")),
    "search_threshold": float(os.getenv("SEARCH_THRESHOLD", "0.5")),  # When to trigger search
    "timeout": int(os.getenv("SEARCH_TIMEOUT", "10")),  # Seconds
}

# Ollama LLM configuration
LLM_CONFIG = {
    "model": os.getenv("LLM_MODEL", "gemma:3b"),  # Options: gemma:3b, gemma:7b
    "host": os.getenv("LLM_HOST", "localhost"),
    "port": int(os.getenv("LLM_PORT", "11434")),
    "temperature": float(os.getenv("LLM_TEMPERATURE", "0.7")),
    "top_p": float(os.getenv("LLM_TOP_P", "0.9")),
    "max_tokens": int(os.getenv("LLM_MAX_TOKENS", "2048")),
}

# Flask API configuration (optional)
API_CONFIG = {
    "host": os.getenv("API_HOST", "localhost"),
    "port": int(os.getenv("API_PORT", "5002")),
    "debug": os.getenv("API_DEBUG", "False").lower() == "true",
    "token": os.getenv("API_TOKEN", None),  # Optional authentication token
}

# Prompt templates
SYSTEM_PROMPT = """
You are GemmaMemoSearch, a helpful AI assistant with memory and web search capabilities.
You have access to your conversation history and can search the web when needed.
Always provide accurate, helpful, and concise responses based on the available information.

When you don't know something or need more information, you can use your search tool.
After searching, always cite your sources.

Your memory contains previous conversations and search results that you can reference.
"""

# Logging configuration
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": DATA_DIR / "gemma_memo_search.log",
}
"""Flask API for GemmaMemoSearch."""

import json
import logging
from typing import Dict, List, Optional, Union

from flask import Flask, jsonify, request
from flask_cors import CORS

from .config import API_CONFIG
from .llm.ollama import GemmaLLM
from .memory.retriever import OpenMemoryRetriever
from .rag.chain import GemmaMemoSearchChain
from .search.tool import PerplexicaTool

# Set up logging
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize components
memory_retriever = None
search_tool = None
llm = None
chain = None


def initialize_components():
    """Initialize the GemmaMemoSearch components."""
    global memory_retriever, search_tool, llm, chain
    
    try:
        # Initialize memory retriever
        memory_retriever = OpenMemoryRetriever()
        
        # Initialize search tool
        search_tool = PerplexicaTool(memory_retriever=memory_retriever)
        
        # Initialize LLM
        llm = GemmaLLM()
        
        # Initialize RAG chain
        chain = GemmaMemoSearchChain(
            llm=llm,
            memory_retriever=memory_retriever,
            search_tool=search_tool,
        )
        
        logger.info("GemmaMemoSearch components initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing components: {e}")
        return False


@app.before_first_request
def before_first_request():
    """Initialize components before the first request."""
    initialize_components()


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    """Chat endpoint for processing user queries."""
    # Check if components are initialized
    if chain is None:
        if not initialize_components():
            return jsonify({"error": "Failed to initialize components"}), 500
    
    # Get request data
    data = request.json
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400
    
    query = data["query"]
    
    try:
        # Process query
        response = chain.invoke(query)
        
        return jsonify({
            "response": response,
            "success": True,
        })
    except Exception as e:
        logger.exception(f"Error processing query: {query}")
        return jsonify({"error": str(e)}), 500


@app.route("/memory", methods=["GET"])
def get_memory():
    """Get memory entries."""
    # Check if components are initialized
    if memory_retriever is None:
        if not initialize_components():
            return jsonify({"error": "Failed to initialize components"}), 500
    
    try:
        # Get query parameters
        query = request.args.get("query", "")
        limit = int(request.args.get("limit", "10"))
        
        if query:
            # Search memory for relevant entries
            docs = memory_retriever._get_relevant_documents(query)
            entries = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": doc.metadata.get("score", 0),
                }
                for doc in docs[:limit]
            ]
        else:
            # TODO: Implement listing all memory entries
            # This would require additional methods in the OpenMemoryRetriever class
            entries = []
        
        return jsonify({
            "entries": entries,
            "count": len(entries),
        })
    except Exception as e:
        logger.exception("Error retrieving memory entries")
        return jsonify({"error": str(e)}), 500


@app.route("/search", methods=["POST"])
def search():
    """Perform a web search using Perplexica."""
    # Check if components are initialized
    if search_tool is None:
        if not initialize_components():
            return jsonify({"error": "Failed to initialize components"}), 500
    
    # Get request data
    data = request.json
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400
    
    query = data["query"]
    
    try:
        # Perform search
        results = search_tool._run(query)
        
        return jsonify({
            "results": results,
            "success": True,
        })
    except Exception as e:
        logger.exception(f"Error performing search: {query}")
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    """Reset the chat history."""
    # Check if components are initialized
    if chain is None:
        if not initialize_components():
            return jsonify({"error": "Failed to initialize components"}), 500
    
    try:
        # Reset chat history
        chain.reset()
        
        return jsonify({
            "message": "Chat history reset successfully",
            "success": True,
        })
    except Exception as e:
        logger.exception("Error resetting chat history")
        return jsonify({"error": str(e)}), 500


def run_api():
    """Run the Flask API."""
    # Initialize components
    initialize_components()
    
    # Run the app
    app.run(
        host=API_CONFIG["host"],
        port=API_CONFIG["port"],
        debug=API_CONFIG["debug"],
    )


if __name__ == "__main__":
    run_api()
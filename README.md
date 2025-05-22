# GemmaMemoSearch

GemmaMemoSearch is an open-source, fully self-hosted RAG system that integrates the Gemma 3 LLM with OpenMemory (long-term context) and Perplexica (on-demand web search). It autonomously performs live queries and stores both conversation history and search summaries, delivering structured, context-aware responses.

## Features

- **Persistent Memory Storage**: Stores every user-model interaction in OpenMemory with metadata tagging
- **Contextual Retrieval**: Retrieves the top-K most relevant memory entries per query
- **On-Demand Web Search**: Self-hosted Perplexica integration for live web querying
- **Local LLM Inference**: Runs Gemma 3 (3B or 7B) entirely on-premises via Ollama
- **RAG Pipeline Orchestration**: Built on LangChain's ConversationalRetrievalChain or Agent framework
- **Privacy-First, Offline-First**: All components self-hosted with no cloud dependencies

## Requirements

- Python 3.10+
- Ollama (for running Gemma 3B/7B locally)
- OpenMemory MCP Server
- Perplexica (self-hosted)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/open-gemma-rag.git
cd open-gemma-rag

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Setup

1. Install and start Ollama following instructions at [ollama.ai](https://ollama.ai)
2. Pull the Gemma model: `ollama pull gemma:3b` or `ollama pull gemma:7b`
3. Set up OpenMemory MCP Server
4. Set up Perplexica for web search capabilities
5. Configure the application in `config.py`

## Usage

### Command Line Interface

```bash
python -m gemma_memo_search.cli
```

### Optional Flask API

```bash
python -m gemma_memo_search.api
```

## Configuration

Edit `config.py` to customize:

- Memory retrieval parameters
- LLM settings
- Search behavior
- API configuration

## Project Structure

```
gemma_memo_search/
├── __init__.py
├── cli.py              # Command-line interface
├── api.py              # Optional Flask API
├── config.py           # Configuration settings
├── memory/             # OpenMemory integration
│   ├── __init__.py
│   └── retriever.py    # Custom retriever implementation
├── search/             # Perplexica integration
│   ├── __init__.py
│   └── tool.py         # Custom search tool implementation
├── llm/                # LLM integration
│   ├── __init__.py
│   └── ollama.py       # Ollama client for Gemma
└── rag/                # RAG pipeline components
    ├── __init__.py
    ├── chain.py        # ConversationalRetrievalChain implementation
    └── prompts.py      # Prompt templates
```

## License

MIT License - See [LICENSE](LICENSE) for details.
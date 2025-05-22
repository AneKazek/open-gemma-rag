"""Command-line interface for GemmaMemoSearch."""

import logging
import os
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from .config import LOGGING_CONFIG, MEMORY_CONFIG, SEARCH_CONFIG
from .llm.ollama import GemmaLLM
from .memory.retriever import OpenMemoryRetriever
from .rag.chain import GemmaMemoSearchChain
from .search.tool import PerplexicaTool

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("gemma_memo_search.cli")

# Create console for rich output
console = Console()

# Create Typer app
app = typer.Typer(
    name="GemmaMemoSearch",
    help="A self-hosted RAG system with Gemma 3, OpenMemory, and Perplexica.",
    add_completion=False,
)


@app.command()
def chat(
    memory_top_k: int = typer.Option(
        MEMORY_CONFIG["top_k"],
        "--memory-top-k",
        "-k",
        help="Number of memory entries to retrieve",
    ),
    memory_threshold: float = typer.Option(
        MEMORY_CONFIG["similarity_threshold"],
        "--memory-threshold",
        "-t",
        help="Similarity threshold for memory retrieval",
    ),
    search_threshold: float = typer.Option(
        SEARCH_CONFIG["search_threshold"],
        "--search-threshold",
        "-s",
        help="Threshold for triggering web search",
    ),
    reset_chat: bool = typer.Option(
        False,
        "--reset",
        "-r",
        help="Reset chat history",
    ),
):
    """Start an interactive chat session with GemmaMemoSearch."""
    try:
        # Display welcome message
        console.print(
            Panel.fit(
                "[bold green]GemmaMemoSearch[/bold green]\n"
                "A self-hosted RAG system with Gemma 3, OpenMemory, and Perplexica.",
                title="Welcome",
                border_style="green",
            )
        )

        # Initialize components
        console.print("Initializing components...", style="yellow")
        
        # Initialize memory retriever
        memory_retriever = OpenMemoryRetriever(
            top_k=memory_top_k,
            similarity_threshold=memory_threshold,
        )
        
        # Initialize search tool
        search_tool = PerplexicaTool(
            memory_retriever=memory_retriever,
        )
        
        # Initialize LLM
        llm = GemmaLLM()
        
        # Initialize RAG chain
        chain = GemmaMemoSearchChain(
            llm=llm,
            memory_retriever=memory_retriever,
            search_tool=search_tool,
        )
        
        console.print("Initialization complete!", style="green")
        console.print(
            "Type your questions below. Type 'exit', 'quit', or press Ctrl+C to exit.",
            style="blue",
        )

        # Main chat loop
        while True:
            # Get user input
            query = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            
            # Check for exit commands
            if query.lower() in ("exit", "quit", "q", "bye"):
                console.print("Goodbye!", style="green")
                break
                
            # Check for reset command
            if query.lower() in ("reset", "clear"):
                chain.reset()
                console.print("Chat history reset.", style="yellow")
                continue
                
            # Process query
            with console.status("[bold yellow]Thinking...[/bold yellow]"):
                response = chain.invoke(query)
            
            # Display response
            console.print("\n[bold green]GemmaMemoSearch[/bold green]")
            console.print(Markdown(response))

    except KeyboardInterrupt:
        console.print("\nGoodbye!", style="green")
    except Exception as e:
        logger.exception("Error in chat session")
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")


@app.command()
def version():
    """Display version information."""
    from . import __version__
    console.print(f"GemmaMemoSearch v{__version__}")


if __name__ == "__main__":
    app()
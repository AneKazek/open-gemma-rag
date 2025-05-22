#!/usr/bin/env python
"""Main entry point for GemmaMemoSearch."""

import argparse
import logging
import sys

from gemma_memo_search.api import run_api
from gemma_memo_search.cli import app as cli_app
from gemma_memo_search.config import LOGGING_CONFIG

# Set up logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG["level"]),
    format=LOGGING_CONFIG["format"],
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOGGING_CONFIG["file"]),
    ],
)
logger = logging.getLogger("gemma_memo_search")


def main():
    """Main entry point for GemmaMemoSearch."""
    parser = argparse.ArgumentParser(
        description="GemmaMemoSearch - A self-hosted RAG system with Gemma 3, OpenMemory, and Perplexica."
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "api"],
        default="cli",
        help="Run mode: CLI or API (default: cli)",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information",
    )

    args = parser.parse_args()

    # Show version information
    if args.version:
        from gemma_memo_search import __version__
        print(f"GemmaMemoSearch v{__version__}")
        return 0

    # Run in selected mode
    try:
        if args.mode == "cli":
            # Run CLI
            cli_app()
        elif args.mode == "api":
            # Run API
            run_api()
        return 0
    except KeyboardInterrupt:
        print("\nExiting...")
        return 0
    except Exception as e:
        logger.exception(f"Error running GemmaMemoSearch: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
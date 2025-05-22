"""Setup script for GemmaMemoSearch."""

from setuptools import find_packages, setup

# Read version from package __init__.py
with open("gemma_memo_search/__init__.py", "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"')
            break

# Read long description from README.md
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="gemma_memo_search",
    version=version,
    description="A self-hosted RAG system with Gemma 3, OpenMemory, and Perplexica",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AneKazek",
    author_email="example@example.com",
    url="https://github.com/yourusername/open-gemma-rag",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "python-dotenv>=1.0.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.13",
        "langchain-core>=0.1.10",
        "ollama>=0.1.5",
        "mem0>=0.1.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.2",
        "sentence-transformers>=2.2.2",
        "typer>=0.9.0",
        "rich>=13.6.0",
        "flask>=2.3.3",
        "flask-cors>=4.0.0",
        "pydantic>=2.5.0",
        "python-dateutil>=2.8.2",
        "tqdm>=4.66.1",
    ],
    entry_points={
        "console_scripts": [
            "gemma-memo-search=gemma_memo_search.cli:app",
            "gemma-memo-api=gemma_memo_search.api:run_api",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
)
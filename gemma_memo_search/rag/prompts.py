"""Prompt templates for GemmaMemoSearch."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate

from ..config import SYSTEM_PROMPT

# System prompt template
SYSTEM_PROMPT_TEMPLATE = PromptTemplate.from_template(SYSTEM_PROMPT)

# Conversational RAG prompt
RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# Prompt for when search is needed
SEARCH_DECISION_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an AI assistant that determines if a web search is needed to answer a question.
        If the question requires real-time information, current events, specific facts that might not be in your training data,
        or if you're unsure about the answer, respond with 'SEARCH: <search query>'.
        Otherwise, respond with 'NO_SEARCH'."""),
        ("human", "{question}"),
    ]
)

# Prompt for summarizing search results
SEARCH_SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", """You are an AI assistant that summarizes search results.
        Create a concise, informative summary of the search results provided.
        Focus on extracting the key information relevant to the original query.
        Include important facts, figures, and context.
        Cite sources where appropriate."""),
        ("human", "Original query: {query}\n\nSearch results:\n{search_results}\n\nPlease summarize these results."),
    ]
)

# Prompt for combining memory and search results
COMBINED_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("system", """You have access to the following information:
        1. Memory: Previous conversations and information you've stored
        2. Search Results: Information retrieved from the web (if applicable)
        
        Use this information to provide a comprehensive, accurate response.
        Always cite sources when using information from search results."""),
        ("system", "Memory:\n{memory_content}"),
        ("system", "Search Results:\n{search_content}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
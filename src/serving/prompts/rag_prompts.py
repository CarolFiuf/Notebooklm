"""
RAG Prompts using LangChain PromptTemplate for better validation and composability.

This module provides prompts for:
- RAG Q&A (with system and user messages)
- Document summarization
- Timeline extraction
- Document comparison
"""

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from typing import Optional

# ============================================================================
# Legacy string constants (kept for backward compatibility)
# ============================================================================
RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on provided context documents.

Your guidelines:
1. Answer questions using ONLY the information provided in the context
2. If the context doesn't contain enough information, clearly state this
3. Be precise and cite specific parts of the context when relevant
4. If multiple sources contain relevant information, synthesize them clearly
5. Keep responses focused, accurate, and well-structured
6. Always maintain a helpful and professional tone

When you reference information from the context, be specific about which parts support your answer."""

RAG_USER_PROMPT = """Context Information:
{context}

User Question: {question}

Please provide a comprehensive answer based on the context above. If the context doesn't contain sufficient information to fully answer the question, please indicate what information is missing.

Answer:"""

SUMMARY_PROMPT = """Please provide a comprehensive summary of the following document content.

Your summary should:
1. Capture the main topics and key points
2. Highlight important details and findings
3. Maintain the logical structure of the content
4. Be concise but informative
5. Use clear, accessible language

Document Content:
{content}

Summary:"""

TIMELINE_PROMPT = """Based on the following document content, create a chronological timeline of events, developments, or key points mentioned.

Format your timeline as:
- **Date/Period**: Description of what happened
- **Date/Period**: Description of what happened

If specific dates are not mentioned, organize by relative time periods (e.g., "Initially", "Later", "Finally") or logical sequence.

Document Content:
{content}

Timeline:"""

COMPARISON_PROMPT = """Compare the key points, arguments, or information presented in the following documents:

Document 1: {doc1_title}
{doc1_content}

Document 2: {doc2_title}
{doc2_content}

Please provide:
1. **Similarities**: Common points or themes
2. **Differences**: Contrasting viewpoints or information
3. **Key Insights**: Notable observations from the comparison

Comparison Analysis:"""

# ============================================================================
# LangChain PromptTemplates (New - Recommended)
# ============================================================================

# RAG Chat Prompt (with system + user messages)
rag_chat_prompt = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", """Context Information:
{context}

User Question: {question}

Please provide a comprehensive answer based on the context above. If the context doesn't contain sufficient information to fully answer the question, please indicate what information is missing.

Answer:""")
])

# RAG Simple Prompt (without system message, for simpler LLMs)
rag_simple_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_USER_PROMPT
)

# Summary Prompt
summary_prompt = PromptTemplate(
    input_variables=["content"],
    template=SUMMARY_PROMPT
)

# Timeline Prompt
timeline_prompt = PromptTemplate(
    input_variables=["content"],
    template=TIMELINE_PROMPT
)

# Comparison Prompt
comparison_prompt = PromptTemplate(
    input_variables=["doc1_title", "doc1_content", "doc2_title", "doc2_content"],
    template=COMPARISON_PROMPT
)

# ============================================================================
# Helper Functions (Backward Compatible + New LangChain versions)
# ============================================================================

def build_rag_prompt(question: str, context: str, system_prompt: Optional[str] = None) -> str:
    """
    Build complete RAG prompt (Legacy function - backward compatible).

    Args:
        question: User's question
        context: Retrieved context documents
        system_prompt: Optional system prompt (defaults to RAG_SYSTEM_PROMPT)

    Returns:
        Formatted prompt string
    """
    if system_prompt:
        return f"{system_prompt}\n\n{RAG_USER_PROMPT.format(context=context, question=question)}"
    else:
        return RAG_USER_PROMPT.format(context=context, question=question)


def build_rag_prompt_langchain(question: str, context: str, use_system_prompt: bool = True) -> str:
    """
    Build RAG prompt using LangChain templates (New - Recommended).

    Args:
        question: User's question
        context: Retrieved context documents
        use_system_prompt: Whether to include system prompt

    Returns:
        Formatted prompt string with automatic validation
    """
    if use_system_prompt:
        return rag_chat_prompt.format(context=context, question=question)
    else:
        return rag_simple_prompt.format(context=context, question=question)


def build_summary_prompt(content: str) -> str:
    """
    Build summary prompt (Legacy function - backward compatible).

    Args:
        content: Document content to summarize

    Returns:
        Formatted prompt string
    """
    return SUMMARY_PROMPT.format(content=content)


def build_summary_prompt_langchain(content: str) -> str:
    """
    Build summary prompt using LangChain template (New - Recommended).

    Args:
        content: Document content to summarize

    Returns:
        Formatted prompt string with automatic validation
    """
    return summary_prompt.format(content=content)


def build_timeline_prompt(content: str) -> str:
    """
    Build timeline prompt (Legacy function - backward compatible).

    Args:
        content: Document content to extract timeline from

    Returns:
        Formatted prompt string
    """
    return TIMELINE_PROMPT.format(content=content)


def build_timeline_prompt_langchain(content: str) -> str:
    """
    Build timeline prompt using LangChain template (New - Recommended).

    Args:
        content: Document content to extract timeline from

    Returns:
        Formatted prompt string with automatic validation
    """
    return timeline_prompt.format(content=content)


def build_comparison_prompt(doc1_title: str, doc1_content: str,
                           doc2_title: str, doc2_content: str) -> str:
    """
    Build comparison prompt (Legacy function - backward compatible).

    Args:
        doc1_title: Title of first document
        doc1_content: Content of first document
        doc2_title: Title of second document
        doc2_content: Content of second document

    Returns:
        Formatted prompt string
    """
    return COMPARISON_PROMPT.format(
        doc1_title=doc1_title, doc1_content=doc1_content,
        doc2_title=doc2_title, doc2_content=doc2_content
    )


def build_comparison_prompt_langchain(doc1_title: str, doc1_content: str,
                                     doc2_title: str, doc2_content: str) -> str:
    """
    Build comparison prompt using LangChain template (New - Recommended).

    Args:
        doc1_title: Title of first document
        doc1_content: Content of first document
        doc2_title: Title of second document
        doc2_content: Content of second document

    Returns:
        Formatted prompt string with automatic validation
    """
    return comparison_prompt.format(
        doc1_title=doc1_title,
        doc1_content=doc1_content,
        doc2_title=doc2_title,
        doc2_content=doc2_content
    )
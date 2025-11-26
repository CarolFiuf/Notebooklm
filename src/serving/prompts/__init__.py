from .rag_prompts import *
from .legal_prompts import (
    legal_qa_prompt,
    legal_summary_prompt,
    legal_comparison_prompt,
    LEGAL_SYSTEM_PROMPT
)

__all__ = [
    # Standard RAG prompts
    'RAG_SYSTEM_PROMPT',
    'RAG_USER_PROMPT',
    'SUMMARY_PROMPT',
    'build_rag_prompt',
    # Legal prompts
    'legal_qa_prompt',
    'legal_summary_prompt',
    'legal_comparison_prompt',
    'LEGAL_SYSTEM_PROMPT'
]
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

def build_rag_prompt(question: str, context: str, system_prompt: str = None) -> str:
    """Build complete RAG prompt"""
    if system_prompt:
        return f"{system_prompt}\n\n{RAG_USER_PROMPT.format(context=context, question=question)}"
    else:
        return RAG_USER_PROMPT.format(context=context, question=question)

def build_summary_prompt(content: str) -> str:
    """Build summary prompt"""
    return SUMMARY_PROMPT.format(content=content)

def build_timeline_prompt(content: str) -> str:
    """Build timeline prompt"""
    return TIMELINE_PROMPT.format(content=content)

def build_comparison_prompt(doc1_title: str, doc1_content: str, 
                          doc2_title: str, doc2_content: str) -> str:
    """Build comparison prompt"""
    return COMPARISON_PROMPT.format(
        doc1_title=doc1_title, doc1_content=doc1_content,
        doc2_title=doc2_title, doc2_content=doc2_content
    )
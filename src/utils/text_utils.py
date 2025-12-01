import re
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """Clean and normalize text while preserving line structure"""
    if not text:
        return ""

    # FIX: Preserve newlines for legal document structure
    # Normalize whitespace within lines but keep line breaks
    lines = text.split('\n')
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in lines]
    text = '\n'.join(lines)

    # Remove excessive blank lines (3+ newlines → 2 newlines max)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove special characters but keep basic punctuation AND quotation marks (including smart quotes)
    # Keep quotes (both regular and smart quotes) to distinguish quoted Điều from real Điều
    # Smart quotes: " " ' ' (U+201C, U+201D, U+2018, U+2019)
    text = re.sub(r'[^\w\s\.,!?;:()\"\'\u201C\u201D\u2018\u2019\-]', '', text)

    # Strip and return
    return text.strip()

def extract_sentences(text: str) -> List[str]:
    """Extract sentences from text"""
    # Simple sentence splitting (can be improved with spaCy/nltk)
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate text to max_length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def count_tokens_estimate(text: str) -> int:
    """Rough token count estimation (words * 1.3)"""
    words = len(text.split())
    return int(words * 1.3)

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract simple keywords from text"""
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
        'has', 'had', 'will', 'would', 'could', 'should', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
    }
    
    # Extract words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter and count
    word_freq = {}
    for word in words:
        if word not in stop_words and len(word) > 2:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in keywords[:max_keywords]]

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"
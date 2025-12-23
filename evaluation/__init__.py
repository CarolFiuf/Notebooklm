"""
Evaluation package for Vietnamese Legal RAG System

Provides comprehensive evaluation framework combining:
- RAGAS metrics for general RAG evaluation
- Custom legal-specific metrics for article citation accuracy

Usage:
    # Import and use evaluation functions
    from evaluation import run_rag_evaluation, evaluate_with_ragas

    # Or run CLI tool directly
    python -m evaluation.evaluate_rag --help
"""

from .legal_metrics import (
    extract_article_numbers,
    calculate_article_citation_accuracy,
    calculate_article_precision,
    calculate_article_recall,
    evaluate_legal_metrics,
    print_legal_metrics_report
)

from .evaluate_rag import (
    create_evaluation_dataset,
    run_rag_evaluation,
    evaluate_with_ragas,
    print_evaluation_report,
    load_test_dataset,
    validate_test_cases,
    LEGAL_TEST_CASES
)

__all__ = [
    # Legal metrics
    'extract_article_numbers',
    'calculate_article_citation_accuracy',
    'calculate_article_precision',
    'calculate_article_recall',
    'evaluate_legal_metrics',
    'print_legal_metrics_report',

    # RAGAS evaluation
    'create_evaluation_dataset',
    'run_rag_evaluation',
    'evaluate_with_ragas',
    'print_evaluation_report',
    'load_test_dataset',
    'validate_test_cases',
    'LEGAL_TEST_CASES'
]

__version__ = '1.0.0'

"""
Custom Metrics for Vietnamese Legal Document Evaluation

Extends RAGAS with legal-specific metrics:
- Article Citation Accuracy: Checks if retrieved sources contain expected articles
- Article Precision: How many retrieved articles are correct
- Article Recall: How many expected articles are retrieved
"""
import re
import logging
from typing import List, Dict, Any, Optional
from datasets import Dataset

logger = logging.getLogger(__name__)


def extract_article_numbers(text: str) -> List[int]:
    """
    Extract article numbers from Vietnamese legal text

    Args:
        text: Text to extract from (e.g., "Điều 51", "điều 55")

    Returns:
        List of article numbers found
    """
    pattern = re.compile(r'\bđiều\s+(\d+)', re.IGNORECASE)
    matches = pattern.findall(text)
    return [int(num) for num in matches]


def calculate_article_citation_accuracy(
    expected_articles: List[int],
    retrieved_contexts: List[str]
) -> float:
    """
    Calculate article citation accuracy

    Checks if all expected articles are present in retrieved contexts

    Args:
        expected_articles: List of expected article numbers
        retrieved_contexts: List of retrieved text chunks

    Returns:
        Score from 0.0 to 1.0
    """
    if not expected_articles:
        return 1.0

    # Extract all articles from retrieved contexts
    retrieved_articles = set()
    for context in retrieved_contexts:
        articles = extract_article_numbers(context)
        retrieved_articles.update(articles)

    # Calculate how many expected articles are found
    expected_set = set(expected_articles)
    found_articles = expected_set.intersection(retrieved_articles)

    accuracy = len(found_articles) / len(expected_set)

    logger.debug(f"Expected articles: {expected_set}")
    logger.debug(f"Retrieved articles: {retrieved_articles}")
    logger.debug(f"Found: {found_articles}, Accuracy: {accuracy:.2f}")

    return accuracy


def calculate_article_precision(
    expected_articles: List[int],
    retrieved_contexts: List[str]
) -> float:
    """
    Calculate precision of article retrieval

    Precision = (Relevant articles retrieved) / (Total articles retrieved)

    Args:
        expected_articles: List of expected article numbers
        retrieved_contexts: List of retrieved text chunks

    Returns:
        Precision score from 0.0 to 1.0
    """
    # Extract all articles from retrieved contexts
    retrieved_articles = set()
    for context in retrieved_contexts:
        articles = extract_article_numbers(context)
        retrieved_articles.update(articles)

    if not retrieved_articles:
        return 0.0

    expected_set = set(expected_articles)
    relevant_retrieved = expected_set.intersection(retrieved_articles)

    precision = len(relevant_retrieved) / len(retrieved_articles)
    return precision


def calculate_article_recall(
    expected_articles: List[int],
    retrieved_contexts: List[str]
) -> float:
    """
    Calculate recall of article retrieval

    Recall = (Relevant articles retrieved) / (Total expected articles)
    Same as article_citation_accuracy but named for clarity

    Args:
        expected_articles: List of expected article numbers
        retrieved_contexts: List of retrieved text chunks

    Returns:
        Recall score from 0.0 to 1.0
    """
    return calculate_article_citation_accuracy(expected_articles, retrieved_contexts)


def evaluate_legal_metrics(evaluation_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate custom legal metrics across entire dataset

    Args:
        evaluation_data: List of evaluation cases with:
            - expected_articles: List[int]
            - contexts: List[str]

    Returns:
        Dict with average scores for each metric
    """
    citation_accuracies = []
    precisions = []
    recalls = []

    for case in evaluation_data:
        expected_articles = case.get('expected_articles', [])
        contexts = case.get('contexts', [])

        if expected_articles:  # Only evaluate if expected articles are specified
            citation_acc = calculate_article_citation_accuracy(expected_articles, contexts)
            precision = calculate_article_precision(expected_articles, contexts)
            recall = calculate_article_recall(expected_articles, contexts)

            citation_accuracies.append(citation_acc)
            precisions.append(precision)
            recalls.append(recall)

    # Calculate averages
    results = {}
    if citation_accuracies:
        results['article_citation_accuracy'] = sum(citation_accuracies) / len(citation_accuracies)
        results['article_precision'] = sum(precisions) / len(precisions)
        results['article_recall'] = sum(recalls) / len(recalls)

        # F1 score for articles
        if results['article_precision'] + results['article_recall'] > 0:
            results['article_f1'] = 2 * (
                results['article_precision'] * results['article_recall']
            ) / (results['article_precision'] + results['article_recall'])
        else:
            results['article_f1'] = 0.0

    return results


def print_legal_metrics_report(scores: Dict[str, float]):
    """
    Print formatted report for legal-specific metrics

    Args:
        scores: Dict of legal metric scores
    """
    print("\n" + "="*80)
    print("LEGAL-SPECIFIC METRICS")
    print("="*80)

    if not scores:
        print("  No legal metrics available (expected_articles not provided)")
        return

    print("\n📋 Article Citation Metrics:")
    print("-" * 80)

    metrics_info = {
        'article_citation_accuracy': ('Article Recall', 'Expected articles found in retrieval'),
        'article_precision': ('Article Precision', 'Retrieved articles that are relevant'),
        'article_recall': ('Article Recall', 'Same as citation accuracy'),
        'article_f1': ('Article F1 Score', 'Harmonic mean of precision and recall'),
    }

    for metric, score in scores.items():
        emoji = "✅" if score >= 0.8 else "⚠️" if score >= 0.6 else "❌"
        name, description = metrics_info.get(metric, (metric, ''))
        print(f"  {emoji} {name}: {score:.3f}")
        if description:
            print(f"     └─ {description}")

    print("\n" + "="*80 + "\n")

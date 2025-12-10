"""
RAGAS Evaluation for Legal RAG System

Evaluates RAG performance using RAGAS metrics:
- Faithfulness: Answer faithful to retrieved context
- Answer Relevancy: Answer relevant to question
- Context Precision: Retrieved context quality
- Context Recall: Coverage of ground truth
- Context Relevancy: Context relevant to question

For Vietnamese legal documents
"""
import logging
from typing import List, Dict, Any
import pandas as pd
from datasets import Dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_evaluation_dataset(test_cases: List[Dict[str, Any]]) -> Dataset:
    """
    Create RAGAS evaluation dataset

    Args:
        test_cases: List of test cases with:
            - question: User question
            - ground_truth: Expected answer (optional for some metrics)
            - contexts: Retrieved contexts (will be filled by RAG)
            - answer: Generated answer (will be filled by RAG)

    Returns:
        HuggingFace Dataset for RAGAS
    """
    return Dataset.from_pandas(pd.DataFrame(test_cases))


def run_rag_evaluation(
    rag_engine,
    test_questions: List[str],
    ground_truths: List[str] = None,
    document_ids: List[int] = None
) -> Dict[str, Any]:
    """
    Run RAG system and collect results for evaluation

    Args:
        rag_engine: RAGEngine instance
        test_questions: List of test questions
        ground_truths: Optional list of ground truth answers
        document_ids: Optional document IDs to filter

    Returns:
        Dict with evaluation data
    """
    results = []

    for i, question in enumerate(test_questions):
        logger.info(f"Processing question {i+1}/{len(test_questions)}: {question[:50]}...")

        # Query RAG
        response = rag_engine.query(
            question=question,
            document_ids=document_ids,
            top_k=5
        )

        # Extract contexts from sources
        contexts = [
            source.get('content', '')
            for source in response.get('sources', [])
        ]

        result = {
            'question': question,
            'answer': response.get('answer', ''),
            'contexts': contexts,
        }

        # Add ground truth if available
        if ground_truths and i < len(ground_truths):
            result['ground_truth'] = ground_truths[i]

        results.append(result)

    return results


def evaluate_with_ragas(evaluation_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate RAG using RAGAS metrics

    Args:
        evaluation_data: List of dicts with question, answer, contexts, ground_truth

    Returns:
        Dict of metric scores
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            context_relevancy
        )

        # Create dataset
        dataset = create_evaluation_dataset(evaluation_data)

        # Select metrics based on available data
        metrics = [faithfulness, answer_relevancy, context_relevancy]

        # Only add precision/recall if ground_truth is available
        if 'ground_truth' in evaluation_data[0]:
            metrics.extend([context_precision, context_recall])

        logger.info(f"Running RAGAS evaluation with {len(metrics)} metrics...")

        # Run evaluation
        results = evaluate(dataset, metrics=metrics)

        # Convert to dict
        scores = {
            metric: float(results[metric])
            for metric in results.keys()
        }

        return scores

    except ImportError:
        logger.error("RAGAS not installed. Install with: pip install ragas")
        return {}
    except Exception as e:
        logger.error(f"Error during RAGAS evaluation: {e}")
        return {}


def print_evaluation_report(
    scores: Dict[str, float],
    evaluation_data: List[Dict[str, Any]]
):
    """
    Print formatted evaluation report

    Args:
        scores: RAGAS metric scores
        evaluation_data: Evaluation data with questions/answers
    """
    print("\n" + "="*80)
    print("RAGAS EVALUATION REPORT")
    print("="*80)

    # Overall scores
    print("\n📊 Overall Metrics:")
    print("-" * 80)
    for metric, score in scores.items():
        emoji = "✅" if score >= 0.7 else "⚠️" if score >= 0.5 else "❌"
        print(f"  {emoji} {metric.replace('_', ' ').title()}: {score:.3f}")

    # Average
    if scores:
        avg_score = sum(scores.values()) / len(scores)
        print(f"\n  📈 Average Score: {avg_score:.3f}")

    # Sample results
    print("\n📝 Sample Results:")
    print("-" * 80)
    for i, item in enumerate(evaluation_data[:3], 1):
        print(f"\n  Example {i}:")
        print(f"  Question: {item['question'][:100]}...")
        print(f"  Answer: {item['answer'][:150]}...")
        print(f"  Contexts: {len(item['contexts'])} chunks retrieved")
        if 'ground_truth' in item:
            print(f"  Ground Truth: {item['ground_truth'][:100]}...")

    print("\n" + "="*80 + "\n")


# Example test cases for Vietnamese legal documents
LEGAL_TEST_CASES = [
    {
        "question": "Điều 51 quy định gì?",
        "ground_truth": "Điều 51 quy định về [nội dung cụ thể]",
    },
    {
        "question": "Thời gian làm việc tối đa là bao nhiêu giờ?",
        "ground_truth": "Thời gian làm việc tối đa là 8 giờ/ngày và 48 giờ/tuần",
    },
    {
        "question": "Quyền và nghĩa vụ của người lao động là gì?",
        "ground_truth": "Người lao động có quyền được làm việc, hưởng lương, nghỉ ngơi...",
    },
]


def main():
    """Main evaluation workflow"""
    from src.rag.rag_engine import RAGEngine

    # Initialize RAG
    logger.info("Initializing RAG Engine...")
    rag_engine = RAGEngine(enable_reranking=True)

    # Get test questions
    test_questions = [case['question'] for case in LEGAL_TEST_CASES]
    ground_truths = [case['ground_truth'] for case in LEGAL_TEST_CASES]

    # Run RAG and collect results
    logger.info(f"Running RAG on {len(test_questions)} test questions...")
    evaluation_data = run_rag_evaluation(
        rag_engine=rag_engine,
        test_questions=test_questions,
        ground_truths=ground_truths
    )

    # Evaluate with RAGAS
    logger.info("Evaluating with RAGAS metrics...")
    scores = evaluate_with_ragas(evaluation_data)

    # Print report
    print_evaluation_report(scores, evaluation_data)

    # Save results
    import json
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump({
            'scores': scores,
            'test_cases': evaluation_data
        }, f, ensure_ascii=False, indent=2)

    logger.info("Results saved to evaluation_results.json")


if __name__ == "__main__":
    main()

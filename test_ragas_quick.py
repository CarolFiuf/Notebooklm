#!/usr/bin/env python3
"""Quick RAGAS test with single question - matches evaluate_rag.py logic"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("="*80)
print("QUICK RAGAS TEST - Single Question")
print("="*80)

# Initialize RAG
print("\n[1] Initializing RAG Engine...")
from src.rag.rag_engine import RAGEngine
rag_engine = RAGEngine(enable_reranking=True)

# Single test case (same format as LEGAL_TEST_CASES)
test_case = {
    "question": "Luật Đường sắt quy định phạm vi điều chỉnh như thế nào?",
    "ground_truth": "Luật Đường sắt quy định về hoạt động đường sắt; quyền, nghĩa vụ và trách nhiệm của tổ chức, cá nhân liên quan đến hoạt động đường sắt.",
    "expected_articles": [1],
    "category": "railway",
    "difficulty": "easy"
}

# Run RAG query (same as run_rag_evaluation)
print(f"\n[2] Processing question...")
print(f"Question: {test_case['question']}")

response = rag_engine.query(
    question=test_case['question'],
    document_ids=None,
    top_k=5
)

# Extract contexts from sources (same as evaluate_rag.py)
contexts = [
    source.get('content', '')
    for source in response.get('sources', [])
]

evaluation_data = [{
    'question': test_case['question'],
    'answer': response.get('answer', ''),
    'contexts': contexts,
    'ground_truth': test_case['ground_truth']
}]

print(f"\n[3] RAG Response:")
print(f"Answer: {evaluation_data[0]['answer'][:300]}...")
print(f"Contexts: {len(evaluation_data[0]['contexts'])} chunks")

# Test RAGAS evaluation (same as evaluate_with_ragas)
print(f"\n[4] Testing RAGAS evaluation...")
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    )
    from langchain_openai import ChatOpenAI
    from fpt_embeddings import FPTEmbeddings  # Use custom embeddings class
    from datasets import Dataset
    import pandas as pd
    from config.settings import settings

    # Create dataset (same as evaluate_rag.py)
    dataset = Dataset.from_pandas(pd.DataFrame(evaluation_data))

    # Select metrics based on available data (same logic)
    metrics = [faithfulness, answer_relevancy]

    # Only add precision/recall if ground_truth is available
    if 'ground_truth' in evaluation_data[0]:
        metrics.extend([context_precision, context_recall])

    print(f"Metrics to evaluate: {[m.name for m in metrics]}")

    # Configure LLM (same as evaluate_rag.py)
    api_key = settings.EVAL_LLM_API_KEY

    if not api_key:
        print("ERROR: No API key found. Set EVAL_LLM_API_KEY")
        sys.exit(1)

    llm = ChatOpenAI(
        model=settings.EVAL_LLM_MODEL,
        api_key=api_key,
        base_url=settings.EVAL_LLM_BASE_URL,
        temperature=0.1,
        max_retries=settings.EVAL_LLM_MAX_RETRIES,
        request_timeout=settings.EVAL_LLM_TIMEOUT
    )

    # Use custom FPTEmbeddings to avoid tokenization issues
    embeddings = FPTEmbeddings(
        model=settings.EVAL_EMBEDDING_MODEL,
        api_key=settings.EVAL_EMBEDDING_API_KEY,
        base_url=settings.EVAL_EMBEDDING_BASE_URL
    )

    print("\nRunning RAGAS evaluate()...")
    print(f"LLM: {settings.EVAL_LLM_MODEL} @ {settings.EVAL_LLM_BASE_URL}")
    print(f"Embeddings: {settings.EVAL_EMBEDDING_MODEL}")

    # Run evaluation
    result = evaluate(
        dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings
    )

    print("\n" + "="*80)
    print("✅ RAGAS TEST SUCCESSFUL!")
    print("="*80)

    # Display results - iterate through result keys directly
    print(f"\nRAGAS Metrics:")
    for metric in metrics:
        metric_name = metric.name
        try:
            score = result[metric_name]
            # RAGAS returns list of scores (one per test case), get first element or calculate mean
            if isinstance(score, list):
                if len(score) > 0:
                    # For single test case, just take the first element
                    score_val = score[0] if len(score) == 1 else sum(score) / len(score)
                    print(f"  {metric_name}: {score_val:.3f}")
                else:
                    print(f"  {metric_name}: N/A (empty list)")
            else:
                print(f"  {metric_name}: {score:.3f}")
        except (KeyError, TypeError) as e:
            print(f"  {metric_name}: N/A (error: {e})")

    print("="*80)

except Exception as e:
    print("\n" + "="*80)
    print("❌ RAGAS TEST FAILED!")
    print("="*80)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    print("="*80)
    sys.exit(1)

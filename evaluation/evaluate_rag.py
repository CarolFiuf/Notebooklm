"""
RAGAS Evaluation for Legal RAG System

Evaluates RAG performance using RAGAS metrics:
- Faithfulness: Answer faithful to retrieved context
- Answer Relevancy: Answer relevant to question
- Context Precision: Retrieved context quality
- Context Recall: Coverage of ground truth
- Context Relevancy: Context relevant to question

For Vietnamese legal documents

Usage:
    # Run with default test cases (from evaluation/ directory)
    python evaluation/evaluate_rag.py

    # Or from project root
    python -m evaluation.evaluate_rag

    # Run with custom test dataset
    python evaluation/evaluate_rag.py --dataset evaluation/test_dataset.json

    # Run specific document IDs only
    python evaluation/evaluate_rag.py --document-ids 1,2,3

    # Disable reranking
    python evaluation/evaluate_rag.py --no-reranking

    # Save detailed results
    python evaluation/evaluate_rag.py --output results/my_results.json

    # Skip RAGAS evaluation (only legal metrics)
    python evaluation/evaluate_rag.py --llm-provider none
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datasets import Dataset

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import settings after adding project root to path
from config.settings import settings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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


def evaluate_with_ragas(
    evaluation_data: List[Dict[str, Any]],
    llm_provider: str = "fpt"  # "openai", "fpt", or "local"
) -> Dict[str, float]:
    """
    Evaluate RAG using RAGAS metrics

    Uses configuration from config/settings.py for all evaluation parameters.

    Args:
        evaluation_data: List of dicts with question, answer, contexts, ground_truth
        llm_provider: LLM provider ("openai", "fpt", or "local")

    Returns:
        Dict of metric scores
    """
    try:
        import os
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        )
        from config.settings import settings

        # Create dataset
        dataset = create_evaluation_dataset(evaluation_data)

        # Select metrics based on available data
        metrics = [faithfulness, answer_relevancy]

        # Only add precision/recall if ground_truth is available
        if 'ground_truth' in evaluation_data[0]:
            metrics.extend([context_precision, context_recall])

        # Configure LLM based on provider
        llm = None
        embeddings = None

        if llm_provider == "fpt":
            # FPT Cloud configuration from settings
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings

            # Get API key from settings (reads from env: EVAL_LLM_API_KEY -> FPT_API_KEY -> OPENAI_API_KEY)
            api_key = settings.EVAL_LLM_API_KEY

            if not api_key:
                logger.error("No API key found. Set EVAL_LLM_API_KEY, FPT_API_KEY, or OPENAI_API_KEY environment variable")
                return {}

            # LLM configuration
            llm = ChatOpenAI(
                model=settings.EVAL_LLM_MODEL,
                api_key=api_key,
                base_url=settings.EVAL_LLM_BASE_URL,
                temperature=0.1,
                max_retries=settings.EVAL_LLM_MAX_RETRIES,
                request_timeout=settings.EVAL_LLM_TIMEOUT
            )

            # Embeddings configuration (from settings with fallback chain)
            embedding_api_key = settings.EVAL_EMBEDDING_API_KEY
            embedding_base_url = settings.EVAL_EMBEDDING_BASE_URL

            embeddings = OpenAIEmbeddings(
                model=settings.EVAL_EMBEDDING_MODEL,
                api_key=embedding_api_key,
                base_url=embedding_base_url if embedding_base_url else None
            )

            logger.info(f"Using {settings.EVAL_LLM_MODEL} for evaluation")
            logger.info(f"  LLM endpoint: {settings.EVAL_LLM_BASE_URL}")
            if embedding_base_url != settings.EVAL_LLM_BASE_URL:
                logger.info(f"  Embedding endpoint: {embedding_base_url}")

        elif llm_provider == "local":
            # Local llama.cpp model
            from langchain_community.llms import LlamaCpp

            llm = LlamaCpp(
                model_path=str(settings.model_path),
                n_ctx=settings.LLM_CONTEXT_LENGTH,
                n_threads=settings.LLAMACPP_N_THREADS,
                temperature=settings.LLM_TEMPERATURE,
                verbose=False
            )

            logger.info(f"Using local model for evaluation: {settings.LLM_MODEL_NAME}")

        # If llm is configured, wrap it for RAGAS
        ragas_kwargs = {}
        if llm:
            from ragas.llms import LangChainLLMWrapper
            # Use bypass_n for custom endpoints (FPT, local)
            bypass_n = llm_provider in ["fpt", "local"]
            ragas_kwargs['llm'] = LangChainLLMWrapper(llm, bypass_n=bypass_n)

        if embeddings:
            from ragas.embeddings import LangchainEmbeddingsWrapper
            ragas_kwargs['embeddings'] = LangchainEmbeddingsWrapper(embeddings)

        logger.info(f"Running RAGAS evaluation with {len(metrics)} metrics...")

        # Run evaluation
        results = evaluate(dataset, metrics=metrics, **ragas_kwargs)

        # Convert to dict
        scores = {
            metric: float(results[metric])
            for metric in results.keys()
        }

        return scores

    except ImportError as e:
        logger.error(f"RAGAS not installed or missing dependencies: {e}")
        logger.error("Install with: pip install ragas langchain-openai")
        return {}
    except Exception as e:
        logger.error(f"Error during RAGAS evaluation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {}


def print_evaluation_report(
    scores: Dict[str, float],
    evaluation_data: List[Dict[str, Any]],
    legal_scores: Dict[str, float] = None
):
    """
    Print formatted evaluation report

    Args:
        scores: RAGAS metric scores
        evaluation_data: Evaluation data with questions/answers
        legal_scores: Optional legal-specific metric scores
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

    # Legal-specific metrics
    if legal_scores:
        from legal_metrics import print_legal_metrics_report
        print_legal_metrics_report(legal_scores)

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
        if 'expected_articles' in item:
            print(f"  Expected Articles: {item['expected_articles']}")

    print("\n" + "="*80 + "\n")


def load_test_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    Load test cases from JSON file

    Args:
        file_path: Path to JSON file with test cases

    Returns:
        List of test cases
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        test_cases = data.get('test_cases', [])
        logger.info(f"Loaded {len(test_cases)} test cases from {file_path}")

        return test_cases

    except FileNotFoundError:
        logger.error(f"Test dataset file not found: {file_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in test dataset: {e}")
        sys.exit(1)


def validate_test_cases(test_cases: List[Dict[str, Any]]) -> bool:
    """
    Validate test cases have required fields

    Args:
        test_cases: List of test cases to validate

    Returns:
        True if valid, False otherwise
    """
    required_fields = ['question']

    for i, case in enumerate(test_cases, 1):
        missing = [field for field in required_fields if field not in case]
        if missing:
            logger.error(f"Test case {i} missing required fields: {missing}")
            return False

    return True


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Run RAGAS evaluation on Vietnamese Legal RAG system'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to test dataset JSON file'
    )

    parser.add_argument(
        '--document-ids',
        type=str,
        help='Comma-separated document IDs to filter (e.g., "1,2,3")'
    )

    parser.add_argument(
        '--no-reranking',
        action='store_true',
        help='Disable reranking in RAG engine'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top results to retrieve (default: 5)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results.json',
        help='Output file for results (default: evaluation_results.json)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--llm-provider',
        type=str,
        default='fpt',
        choices=['fpt', 'openai', 'local', 'none'],
        help='LLM provider for RAGAS evaluation (default: fpt). Use "none" to skip RAGAS.'
    )

    parser.add_argument(
        '--fpt-api-key',
        type=str,
        help='FPT Cloud API key (default: from FPT_API_KEY env variable)'
    )

    return parser.parse_args()


# Example test cases for Vietnamese legal documents
LEGAL_TEST_CASES = [
    # Luật Đường sắt (95/2025/QH15)
    {
        "question": "Luật Đường sắt quy định phạm vi điều chỉnh như thế nào?",
        "ground_truth": "Luật Đường sắt quy định về hoạt động đường sắt; quyền, nghĩa vụ và trách nhiệm của tổ chức, cá nhân liên quan đến hoạt động đường sắt.",
        "expected_articles": [1],
        "category": "railway",
        "difficulty": "easy"
    },
    {
        "question": "Các hành vi bị nghiêm cấm trong hoạt động đường sắt là gì?",
        "ground_truth": "Các hành vi bị nghiêm cấm bao gồm: phá hoại công trình đường sắt, phương tiện giao thông đường sắt; lấn chiếm hành lang an toàn; làm sai lệch hệ thống báo hiệu; tự ý báo hiệu dừng tàu; để chướng ngại vật, chất dễ cháy nổ trong phạm vi bảo vệ; điều khiển tàu quá tốc độ; nhân viên đường sắt có nồng độ cồn hoặc ma túy trong người.",
        "expected_articles": [6],
        "category": "railway",
        "difficulty": "medium"
    },
    {
        "question": "Đường sắt Việt Nam được phân loại thành những loại nào?",
        "ground_truth": "Hệ thống đường sắt Việt Nam bao gồm: đường sắt quốc gia (phục vụ vận tải chung và liên vận quốc tế), đường sắt địa phương (phục vụ nhu cầu vận tải của địa phương và vùng kinh tế, bao gồm đường sắt đô thị), và đường sắt chuyên dùng (phục vụ nhu cầu riêng của tổ chức, cá nhân).",
        "expected_articles": [7],
        "category": "railway",
        "difficulty": "medium"
    },
    {
        "question": "Khổ đường sắt tiêu chuẩn và khổ đường hẹp có kích thước bao nhiêu?",
        "ground_truth": "Khổ đường sắt tiêu chuẩn là 1435 mm và khổ đường hẹp là 1000 mm. Đường sắt quốc gia và địa phương đầu tư mới phải áp dụng khổ đường tiêu chuẩn.",
        "expected_articles": [8],
        "category": "railway",
        "difficulty": "easy"
    },
    {
        "question": "Hệ thống tín hiệu giao thông đường sắt bao gồm những gì?",
        "ground_truth": "Hệ thống tín hiệu giao thông đường sắt bao gồm: hiệu lệnh của người điều khiển chạy tàu, hệ thống điều khiển chạy tàu, tín hiệu trên tàu, tín hiệu dưới mặt đất, biển báo hiệu, pháo hiệu phòng vệ, đuốc. Hệ thống này phải đầy đủ, chính xác, rõ ràng để bảo đảm an toàn.",
        "expected_articles": [11],
        "category": "railway",
        "difficulty": "hard"
    },

    # Luật sửa đổi về Quân sự (98/2025/QH15)
    {
        "question": "Khu vực phòng thủ được tổ chức như thế nào theo Luật Quốc phòng?",
        "ground_truth": "Khu vực phòng thủ là bộ phận hợp thành phòng thủ quân khu, bao gồm các hoạt động về chính trị, tinh thần, kinh tế, văn hóa, xã hội, khoa học, công nghệ, quân sự, an ninh, đối ngoại; được tổ chức theo địa bàn cấp tỉnh, đơn vị hành chính - kinh tế đặc biệt, lấy xây dựng phòng thủ khu vực, xây dựng cấp xã làm nền tảng.",
        "expected_articles": [1, 9],
        "category": "military",
        "difficulty": "hard"
    },
    {
        "question": "Lệnh thiết quân luật phải xác định những nội dung gì?",
        "ground_truth": "Lệnh thiết quân luật phải xác định cụ thể địa phương cấp tỉnh, cấp xã, đơn vị hành chính - kinh tế đặc biệt thiết quân luật, biện pháp, hiệu lực thi hành; quy định nhiệm vụ, quyền hạn của cơ quan, tổ chức, cá nhân; các quy tắc trật tự xã hội cần thiết và được công bố liên tục trên phương tiện thông tin đại chúng.",
        "expected_articles": [1, 21],
        "category": "military",
        "difficulty": "medium"
    },
    {
        "question": "Công dân nam bao nhiêu tuổi phải đăng ký nghĩa vụ quân sự lần đầu?",
        "ground_truth": "Công dân nam đủ 17 tuổi trong năm phải đăng ký nghĩa vụ quân sự lần đầu. Việc đăng ký được thực hiện vào tháng tư hằng năm, có thể bằng hình thức trực tuyến hoặc trực tiếp tại cơ quan đăng ký nghĩa vụ quân sự.",
        "expected_articles": [4, 16],
        "category": "military",
        "difficulty": "easy"
    },
    {
        "question": "Hành vi trốn tránh nghĩa vụ quân sự được hiểu như thế nào?",
        "ground_truth": "Trốn tránh nghĩa vụ quân sự là hành vi không chấp hành quyết định gọi đăng ký nghĩa vụ quân sự; quyết định gọi khám sức khỏe nghĩa vụ quân sự; quyết định gọi nhập ngũ; quyết định gọi tập trung huấn luyện, diễn tập, kiểm tra sẵn sàng động viên, sẵn sàng chiến đấu.",
        "expected_articles": [4, 3],
        "category": "military",
        "difficulty": "medium"
    },
    {
        "question": "Hội đồng nghĩa vụ quân sự cấp tỉnh có những nhiệm vụ gì?",
        "ground_truth": "Hội đồng nghĩa vụ quân sự cấp tỉnh giúp UBND cấp tỉnh: chỉ đạo đăng ký nghĩa vụ quân sự và quản lý công dân trong độ tuổi; tuyển chọn gọi công dân nhập ngũ; báo cáo quyết định công dân được gọi nhập ngũ, tạm hoãn, miễn gọi; chỉ đạo UBND cấp xã; tổ chức bàn giao công dân cho đơn vị quân đội; kiểm tra thực hiện chính sách hậu phương quân đội; giải quyết khiếu nại, tố cáo.",
        "expected_articles": [4, 37],
        "category": "military",
        "difficulty": "hard"
    },
]


def main():
    """Main evaluation workflow with CLI arguments"""
    import os
    from src.rag.rag_engine import RAGEngine
    from legal_metrics import evaluate_legal_metrics

    # Parse arguments
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load test cases
    if args.dataset:
        test_cases = load_test_dataset(args.dataset)
    else:
        logger.info("Using default test cases from LEGAL_TEST_CASES")
        test_cases = LEGAL_TEST_CASES

    # Validate test cases
    if not validate_test_cases(test_cases):
        logger.error("Test case validation failed")
        sys.exit(1)

    # Parse document IDs
    document_ids = None
    if args.document_ids:
        try:
            document_ids = [int(id.strip()) for id in args.document_ids.split(',')]
            logger.info(f"Filtering to documents: {document_ids}")
        except ValueError:
            logger.error("Invalid document IDs format. Use comma-separated integers.")
            sys.exit(1)

    # Initialize RAG Engine
    logger.info("Initializing RAG Engine...")
    rag_engine = RAGEngine(enable_reranking=not args.no_reranking)

    if args.no_reranking:
        logger.info("Reranking disabled")
    else:
        logger.info("Reranking enabled")

    # Extract questions and ground truths
    test_questions = [case['question'] for case in test_cases]
    ground_truths = [case.get('ground_truth', '') for case in test_cases]

    # Run RAG evaluation
    logger.info(f"Running RAG on {len(test_questions)} test questions...")
    evaluation_data = run_rag_evaluation(
        rag_engine=rag_engine,
        test_questions=test_questions,
        ground_truths=ground_truths,
        document_ids=document_ids
    )

    # Add metadata from test cases
    for i, case in enumerate(test_cases):
        if i < len(evaluation_data):
            # Add expected articles if available
            if 'expected_articles' in case:
                evaluation_data[i]['expected_articles'] = case['expected_articles']

            # Add category and difficulty
            if 'category' in case:
                evaluation_data[i]['category'] = case['category']
            if 'difficulty' in case:
                evaluation_data[i]['difficulty'] = case['difficulty']

    # Evaluate with RAGAS (config from settings.py)
    ragas_scores = {}
    if args.llm_provider != 'none':
        logger.info(f"Evaluating with RAGAS metrics (provider: {args.llm_provider})...")

        # Check if API key is configured in settings (reads from env variables)
        if args.llm_provider == 'fpt' and not settings.EVAL_LLM_API_KEY:
            logger.warning("No API key found. Skipping RAGAS evaluation.")
            logger.info("Set environment variable: export EVAL_LLM_API_KEY=your-api-key")
            logger.info("Or: export FPT_API_KEY=your-api-key")
            logger.info("Or: export OPENAI_API_KEY=your-api-key")
        else:
            ragas_scores = evaluate_with_ragas(
                evaluation_data,
                llm_provider=args.llm_provider  # Uses settings from config/settings.py
            )
    else:
        logger.info("Skipping RAGAS evaluation (--llm-provider=none)")

    # Evaluate legal-specific metrics
    logger.info("Evaluating legal-specific metrics...")
    legal_scores = evaluate_legal_metrics(evaluation_data)

    # Print report
    print_evaluation_report(ragas_scores, evaluation_data, legal_scores)

    # Save results
    output_path = args.output
    logger.info(f"Saving results to {output_path}...")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'num_test_cases': len(test_cases),
                'document_ids': document_ids,
                'reranking_enabled': not args.no_reranking,
                'top_k': args.top_k,
            },
            'ragas_scores': ragas_scores,
            'legal_scores': legal_scores,
            'test_cases': evaluation_data
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Evaluation complete! Results saved to {output_path}")

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Test cases: {len(test_cases)}")
    print(f"Document filter: {document_ids if document_ids else 'All documents'}")
    print(f"Reranking: {'Enabled' if not args.no_reranking else 'Disabled'}")
    print(f"Top-K: {args.top_k}")

    if ragas_scores:
        avg_ragas = sum(ragas_scores.values()) / len(ragas_scores)
        print(f"\n📊 Average RAGAS Score: {avg_ragas:.3f}")

    if legal_scores:
        print(f"📋 Article Citation Accuracy: {legal_scores.get('article_citation_accuracy', 0):.3f}")

    print("="*80 + "\n")


if __name__ == "__main__":
    main()

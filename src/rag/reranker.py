"""
Reranker Service using Cross-Encoder models

Best practices for reranking:
- Use cross-encoder for accurate relevance scoring
- Apply after initial retrieval to refine results
- Supports multilingual (including Vietnamese)
"""
import logging
from typing import List, Dict, Any
import time

logger = logging.getLogger(__name__)


class RerankerService:
    """
    Cross-encoder based reranker for improving retrieval quality

    Uses BAAI/bge-reranker-v2-m3 - multilingual cross-encoder
    that works well with Vietnamese text
    """

    def __init__(self, model_name: str = None):
        """
        Initialize reranker with cross-encoder model

        Args:
            model_name: HuggingFace model name. Defaults to bge-reranker-v2-m3
        """
        from config.settings import settings

        # Model selection - prioritize smaller, faster models for Vietnamese
        self.model_name = model_name or getattr(
            settings, 'RERANKER_MODEL_NAME',
            'BAAI/bge-reranker-v2-m3'  # Multilingual, good for Vietnamese
        )

        self.model = None
        self.device = None
        self._load_model()

        # Metrics
        self.metrics = {
            'total_reranks': 0,
            'avg_rerank_time_ms': 0,
            'total_pairs_scored': 0
        }

        logger.info(f"RerankerService initialized with {self.model_name}")

    def _load_model(self):
        """Load cross-encoder model"""
        try:
            from sentence_transformers import CrossEncoder
            import torch

            # Determine device
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'

            logger.info(f"Loading reranker model {self.model_name} on {self.device}...")

            # Load cross-encoder
            self.model = CrossEncoder(
                self.model_name,
                max_length=512,
                device=self.device
            )

            logger.info(f"Reranker model loaded successfully on {self.device}")

        except Exception as e:
            logger.error(f"Error loading reranker model: {e}")
            raise

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query

        Args:
            query: User query
            documents: List of document dicts with 'content' field
            top_k: Number of top results to return (None = all)
            return_scores: Whether to include rerank scores

        Returns:
            Reranked list of documents
        """
        if not documents:
            return []

        if not self.model:
            logger.warning("Reranker model not loaded, returning original order")
            return documents[:top_k] if top_k else documents

        try:
            start_time = time.time()

            # Prepare query-document pairs
            pairs = []
            for doc in documents:
                content = doc.get('content', '')
                if isinstance(content, str) and content.strip():
                    pairs.append([query, content])
                else:
                    pairs.append([query, ''])

            # Score pairs with cross-encoder
            scores = self.model.predict(pairs, show_progress_bar=False)

            # Combine documents with scores
            scored_docs = []
            for i, (doc, score) in enumerate(zip(documents, scores)):
                doc_copy = doc.copy()
                doc_copy['rerank_score'] = float(score)
                doc_copy['original_rank'] = i
                scored_docs.append(doc_copy)

            # Sort by rerank score (descending)
            reranked = sorted(scored_docs, key=lambda x: x['rerank_score'], reverse=True)

            # Apply top_k limit
            if top_k:
                reranked = reranked[:top_k]

            # Update metrics
            elapsed_ms = (time.time() - start_time) * 1000
            self.metrics['total_reranks'] += 1
            self.metrics['total_pairs_scored'] += len(pairs)
            self.metrics['avg_rerank_time_ms'] = (
                (self.metrics['avg_rerank_time_ms'] * (self.metrics['total_reranks'] - 1) + elapsed_ms)
                / self.metrics['total_reranks']
            )

            logger.debug(f"Reranked {len(documents)} documents in {elapsed_ms:.1f}ms")

            # Remove scores if not needed
            if not return_scores:
                for doc in reranked:
                    doc.pop('rerank_score', None)
                    doc.pop('original_rank', None)

            return reranked

        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fallback to original order
            return documents[:top_k] if top_k else documents

    def rerank_with_fusion(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = None,
        initial_score_weight: float = 0.3,
        rerank_score_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Rerank with score fusion (combine initial retrieval score + rerank score)

        Args:
            query: User query
            documents: List of documents with 'score' field
            top_k: Number of results
            initial_score_weight: Weight for initial retrieval score
            rerank_score_weight: Weight for rerank score

        Returns:
            Reranked documents with fused scores
        """
        if not documents:
            return []

        # First, get rerank scores
        reranked = self.rerank(query, documents, top_k=None, return_scores=True)

        # Normalize scores for fusion
        initial_scores = [doc.get('score', 0) for doc in reranked]
        rerank_scores = [doc.get('rerank_score', 0) for doc in reranked]

        # Min-max normalization
        def normalize(scores):
            if not scores:
                return scores
            min_s, max_s = min(scores), max(scores)
            if max_s == min_s:
                return [0.5] * len(scores)
            return [(s - min_s) / (max_s - min_s) for s in scores]

        norm_initial = normalize(initial_scores)
        norm_rerank = normalize(rerank_scores)

        # Compute fused scores
        for i, doc in enumerate(reranked):
            fused_score = (
                initial_score_weight * norm_initial[i] +
                rerank_score_weight * norm_rerank[i]
            )
            doc['fused_score'] = fused_score

        # Re-sort by fused score
        reranked = sorted(reranked, key=lambda x: x['fused_score'], reverse=True)

        if top_k:
            reranked = reranked[:top_k]

        return reranked

    def get_model_info(self) -> Dict[str, Any]:
        """Get reranker model information"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'is_loaded': self.model is not None,
            'metrics': self.metrics
        }

    def health_check(self) -> bool:
        """Check if reranker is operational"""
        if not self.model:
            return False

        try:
            # Quick test
            test_score = self.model.predict([["test query", "test document"]])
            return True
        except Exception as e:
            logger.error(f"Reranker health check failed: {e}")
            return False


# Singleton instance
_reranker_instance = None

def get_reranker() -> RerankerService:
    """Get or create singleton reranker instance"""
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = RerankerService()
    return _reranker_instance

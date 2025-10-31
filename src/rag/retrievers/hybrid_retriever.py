"""
✅ MIGRATED: LangChain-based Hybrid Retriever
Replaced 584 lines of custom code with ~180 lines using LangChain

Benefits:
- Built-in Ensemble retrieval (semantic + keyword)
- BM25 keyword search
- Automatic score fusion
- Community maintained
"""
import logging
from typing import List, Dict, Any, Optional

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document as LangChainDocument

from src.rag.retrievers.semantic_retriever import SemanticRetriever

logger = logging.getLogger(__name__)

class HybridRetriever:
    """
    ✅ MIGRATED: Hybrid retriever using LangChain

    Combines semantic (vector) and keyword (BM25) search
    """

    def __init__(self, semantic_retriever: SemanticRetriever = None):
        """
        Initialize hybrid retriever

        Args:
            semantic_retriever: SemanticRetriever instance
        """
        self.semantic_retriever = semantic_retriever or SemanticRetriever()

        # Weights for ensemble
        self.semantic_weight = 0.7
        self.keyword_weight = 0.3

        # BM25 retriever (initialized later with documents)
        self.bm25_retriever = None
        self.ensemble_retriever = None

        # Metrics
        self.metrics = {
            'total_searches': 0,
            'semantic_only': 0,
            'hybrid_searches': 0
        }

        logger.info("✅ Hybrid Retriever initialized (LangChain Ensemble)")

    def initialize_bm25(self, documents: List[LangChainDocument]):
        """
        ✅ Initialize BM25 retriever with documents

        Args:
            documents: List of LangChain documents
        """
        try:
            if not documents:
                logger.warning("No documents provided for BM25 initialization")
                return

            logger.info(f"Initializing BM25 with {len(documents)} documents")

            # ✅ Create BM25 retriever
            self.bm25_retriever = BM25Retriever.from_documents(documents)
            self.bm25_retriever.k = 10

            # ✅ Create ensemble retriever
            if self.semantic_retriever.retriever:
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[
                        self.semantic_retriever.retriever,
                        self.bm25_retriever
                    ],
                    weights=[self.semantic_weight, self.keyword_weight]
                )
                logger.info("✅ Ensemble retriever created (70% semantic, 30% keyword)")
            else:
                logger.warning("Semantic retriever not available, using BM25 only")
                self.ensemble_retriever = self.bm25_retriever

        except Exception as e:
            logger.error(f"Error initializing BM25: {e}")
            self.ensemble_retriever = self.semantic_retriever.retriever

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[int]] = None,
        use_hybrid: bool = True
    ) -> List[Dict[str, Any]]:
        """
        ✅ MIGRATED: Hybrid search using LangChain Ensemble

        Args:
            query: Search query
            top_k: Number of results
            document_ids: Filter by document IDs
            use_hybrid: Use hybrid search (True) or semantic only (False)

        Returns:
            List of search results
        """
        try:
            self.metrics['total_searches'] += 1

            # Choose retriever
            if use_hybrid and self.ensemble_retriever:
                retriever = self.ensemble_retriever
                self.metrics['hybrid_searches'] += 1
                logger.debug("Using hybrid search (semantic + BM25)")
            else:
                retriever = self.semantic_retriever.retriever
                self.metrics['semantic_only'] += 1
                logger.debug("Using semantic search only")

            if not retriever:
                logger.error("No retriever available")
                return []

            # Apply document filter if needed
            # 🔧 FIX: Use Qdrant's proper metadata filter format
            if document_ids and hasattr(retriever, 'search_kwargs'):
                if len(document_ids) == 1:
                    # Single document ID - use direct match in metadata
                    retriever.search_kwargs['filter'] = {
                        "must": [
                            {"key": "document_id", "match": {"value": document_ids[0]}}
                        ]
                    }
                else:
                    # Multiple document IDs - use should (OR) condition
                    retriever.search_kwargs['filter'] = {
                        "should": [
                            {"key": "document_id", "match": {"value": doc_id}}
                            for doc_id in document_ids
                        ]
                    }

            # ✅ Retrieve documents
            docs = retriever.get_relevant_documents(query)

            # Convert to expected format (backward compatible)
            results = []
            for doc in docs[:top_k]:
                result = {
                    'content': doc.page_content,
                    'document_id': doc.metadata.get('document_id'),
                    'chunk_index': doc.metadata.get('chunk_index', 0),
                    'metadata': doc.metadata,
                    'score': doc.metadata.get('score', 1.0),
                    'retrieval_method': 'hybrid' if use_hybrid else 'semantic'
                }
                results.append(result)

            logger.debug(f"Retrieved {len(results)} documents")
            return results

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to semantic search
            return self.semantic_retriever.search(query, top_k, document_ids)

    def hybrid_search(
        self,
        query: str,
        document_ids: Optional[List[int]] = None,
        top_k: int = 10,
        semantic_weight: float = None,
        keyword_weight: float = None
    ) -> List[Dict[str, Any]]:
        """
        ✅ MIGRATED: Hybrid search with custom weights

        Args:
            query: Search query
            document_ids: Filter by document IDs
            top_k: Number of results
            semantic_weight: Weight for semantic results (0-1)
            keyword_weight: Weight for keyword results (0-1)

        Returns:
            List of search results
        """
        # Update weights if provided
        if semantic_weight is not None and keyword_weight is not None:
            self.semantic_weight = semantic_weight
            self.keyword_weight = keyword_weight

            # Recreate ensemble with new weights
            if self.semantic_retriever.retriever and self.bm25_retriever:
                self.ensemble_retriever = EnsembleRetriever(
                    retrievers=[
                        self.semantic_retriever.retriever,
                        self.bm25_retriever
                    ],
                    weights=[self.semantic_weight, self.keyword_weight]
                )
                logger.info(f"Updated weights: semantic={semantic_weight}, keyword={keyword_weight}")

        return self.search(query, top_k, document_ids, use_hybrid=True)

    def semantic_search(
        self,
        query: str,
        document_ids: Optional[List[int]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        ✅ Semantic search only (no keyword)

        Args:
            query: Search query
            document_ids: Filter by document IDs
            top_k: Number of results

        Returns:
            List of search results
        """
        return self.semantic_retriever.search(query, top_k, document_ids)

    def keyword_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        ✅ Keyword search only (BM25)

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of search results
        """
        try:
            if not self.bm25_retriever:
                logger.warning("BM25 retriever not initialized")
                return []

            # ✅ Use BM25 retriever
            docs = self.bm25_retriever.get_relevant_documents(query)

            # Convert to expected format
            results = []
            for doc in docs[:top_k]:
                result = {
                    'content': doc.page_content,
                    'document_id': doc.metadata.get('document_id'),
                    'chunk_index': doc.metadata.get('chunk_index', 0),
                    'metadata': doc.metadata,
                    'score': doc.metadata.get('score', 1.0),
                    'retrieval_method': 'keyword'
                }
                results.append(result)

            logger.debug(f"BM25 retrieved {len(results)} documents")
            return results

        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []

    def get_metrics(self) -> Dict[str, Any]:
        """Get retriever metrics"""
        metrics = self.metrics.copy()
        metrics['semantic_retriever_metrics'] = self.semantic_retriever.get_metrics()
        metrics['bm25_initialized'] = self.bm25_retriever is not None
        metrics['ensemble_initialized'] = self.ensemble_retriever is not None
        return metrics

    def get_model_info(self) -> Dict[str, Any]:
        """Get retriever information"""
        return {
            'type': 'hybrid',
            'using_langchain': True,
            'ensemble_enabled': self.ensemble_retriever is not None,
            'semantic_weight': self.semantic_weight,
            'keyword_weight': self.keyword_weight,
            'bm25_initialized': self.bm25_retriever is not None
        }

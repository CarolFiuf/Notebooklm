"""
✅ MIGRATED: LangChain-based Semantic Retriever
Replaced 619 lines of custom code with ~150 lines using LangChain

Benefits:
- Built-in MMR (Max Marginal Relevance) for diversity
- Better relevance scoring
- Community maintained
- Auto-handles edge cases
"""
import logging
from typing import List, Dict, Any, Optional

from langchain.schema import Document as LangChainDocument

logger = logging.getLogger(__name__)

class SemanticRetriever:
    """
    ✅ MIGRATED: Semantic retriever using LangChain

    Uses vector store with MMR for diverse, relevant results
    """

    def __init__(self, embedding_service=None, vector_store=None):
        """
        Initialize semantic retriever

        Args:
            embedding_service: Not used (LangChain handles embeddings)
            vector_store: QdrantVectorStore instance (with LangChain)
        """
        self.embedding_service = embedding_service  # Kept for backward compatibility
        self.vector_store = vector_store

        # Parameters
        self.default_top_k = 10
        self.semantic_threshold = 0.3
        self.diversity_threshold = 0.85

        # Metrics tracking
        self.metrics = {
            'total_searches': 0,
            'cache_hits': 0,
            'avg_response_time_ms': 0,
            'total_results_retrieved': 0
        }

        if self.vector_store and hasattr(self.vector_store, 'vector_store'):
            # ✅ Create LangChain retriever with MMR
            self.retriever = self.vector_store.vector_store.as_retriever(
                search_type="mmr",  # Max Marginal Relevance for diversity
                search_kwargs={
                    "k": 10,
                    "fetch_k": 20,  # Fetch 20, return 10 diverse
                    "lambda_mult": 0.7  # Balance relevance vs diversity
                }
            )
            logger.info("✅ Semantic Retriever initialized with LangChain MMR")
        else:
            self.retriever = None
            logger.warning("Vector store not available, retriever not initialized")

    def semantic_search(
        self,
        query: str,
        document_ids: Optional[List[int]] = None,
        top_k: int = None,
        min_score: float = None,
        enable_reranking: bool = True,
        enable_diversity: bool = True
    ) -> List[Dict[str, Any]]:
        """
        ✅ MIGRATED: Semantic search using LangChain with MMR

        Args:
            query: Search query
            document_ids: Filter by document IDs
            top_k: Number of results
            min_score: Minimum similarity score
            enable_reranking: Enable reranking (MMR always enabled)
            enable_diversity: Enable diversity (MMR handles this)

        Returns:
            List of search results
        """
        return self.search(query, top_k or self.default_top_k, document_ids, min_score or self.semantic_threshold)

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[int]] = None,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        ✅ MIGRATED: Search using LangChain with MMR

        Args:
            query: Search query
            top_k: Number of results
            document_ids: Filter by document IDs
            min_score: Minimum similarity score

        Returns:
            List of search results
        """
        try:
            if not self.retriever:
                logger.error("Retriever not initialized")
                return []

            self.metrics['total_searches'] += 1

            # ✅ Apply document filter if needed
            # 🔧 FIX: Use Qdrant's proper metadata filter format
            search_kwargs = {
                "k": top_k,
                "fetch_k": top_k * 2,
                "lambda_mult": 0.7
            }

            if document_ids:
                if len(document_ids) == 1:
                    # Single document ID - use direct match in metadata
                    search_kwargs['filter'] = {
                        "must": [
                            {"key": "document_id", "match": {"value": document_ids[0]}}
                        ]
                    }
                else:
                    # Multiple document IDs - use should (OR) condition
                    search_kwargs['filter'] = {
                        "should": [
                            {"key": "document_id", "match": {"value": doc_id}}
                            for doc_id in document_ids
                        ]
                    }

            # Update retriever search kwargs
            self.retriever.search_kwargs = search_kwargs

            # ✅ Retrieve documents using LangChain
            docs = self.retriever.get_relevant_documents(query)

            # Convert to expected format (backward compatible)
            results = []
            for doc in docs:
                result = {
                    'content': doc.page_content,
                    'document_id': doc.metadata.get('document_id'),
                    'chunk_index': doc.metadata.get('chunk_index', 0),
                    'metadata': doc.metadata,
                    'score': doc.metadata.get('score', 1.0)
                }

                # Filter by min_score
                if result['score'] >= min_score:
                    results.append(result)

            self.metrics['total_results_retrieved'] += len(results)
            logger.debug(f"Retrieved {len(results)} documents with MMR")

            return results[:top_k]

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def search_similar(
        self,
        query_embedding,
        top_k: int = 5,
        document_ids: Optional[List[int]] = None,
        min_score: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        ✅ Search by embedding vector

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            document_ids: Filter by document IDs
            min_score: Minimum score

        Returns:
            List of search results
        """
        try:
            if not self.vector_store:
                logger.error("Vector store not initialized")
                return []

            # Use vector store's search_similar (already migrated)
            return self.vector_store.search_similar(
                query_embedding=query_embedding,
                top_k=top_k,
                document_ids=document_ids,
                min_score=min_score
            )

        except Exception as e:
            logger.error(f"Error in embedding search: {e}")
            return []

    def get_metrics(self) -> Dict[str, Any]:
        """Get retriever metrics"""
        return self.metrics.copy()

    def get_model_info(self) -> Dict[str, Any]:
        """Get retriever information"""
        return {
            'type': 'semantic',
            'using_langchain': True,
            'search_type': 'mmr',
            'diversity_enabled': True,
            'vector_store_connected': self.vector_store is not None
        }

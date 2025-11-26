"""
✅ MIGRATED: LangChain-based Hybrid Retriever with Reranking

Best retrieval pipeline:
1. Hybrid search (Semantic + BM25 via Ensemble)
2. Cross-encoder reranking for precision
3. Article-aware boosting for legal queries

Benefits:
- Built-in Ensemble retrieval (semantic + keyword)
- BM25 keyword search
- Cross-encoder reranking for better relevance
- Automatic score fusion (RRF)
- Article-aware boosting for Vietnamese legal documents
- Community maintained
"""
import logging
import re
from typing import List, Dict, Any, Optional

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document as LangChainDocument

from src.rag.retrievers.semantic_retriever import SemanticRetriever
from config.settings import settings

logger = logging.getLogger(__name__)


def extract_article_number(query: str) -> Optional[int]:
    """
    ✅ NEW: Extract article number from Vietnamese legal query

    Examples:
        "điều 51" → 51
        "Điều 55 quy định gì" → 55
        "nội dung điều 50" → 50

    Args:
        query: User query string

    Returns:
        Article number if found, else None
    """
    pattern = re.compile(r'\bđiều\s+(\d+)', re.IGNORECASE)
    match = pattern.search(query)

    if match:
        article_num = int(match.group(1))
        logger.info(f"🎯 Detected article query: Điều {article_num}")
        return article_num

    return None

# Lazy import reranker to avoid circular imports
_reranker = None

def get_reranker():
    """Lazy load reranker"""
    global _reranker
    if _reranker is None:
        try:
            from src.rag.reranker import RerankerService
            _reranker = RerankerService()
            logger.info("Reranker loaded for hybrid retriever")
        except Exception as e:
            logger.warning(f"Could not load reranker: {e}")
            _reranker = False  # Mark as failed
    return _reranker if _reranker else None


def build_article_context_filter(article_num: int, context_range: int = 1) -> Dict[str, Any]:
    """
    ✅ NEW: Build Qdrant filter for article with context

    Returns filter for target article + surrounding articles (article±range)
    This provides context while keeping search focused

    Args:
        article_num: Target article number (e.g., 50)
        context_range: Number of articles before/after (default: 1)

    Returns:
        Qdrant filter dict with OR conditions

    Example:
        >>> build_article_context_filter(50, context_range=1)
        # Returns filter for articles 49, 50, 51
        {
            "should": [
                {"key": "metadata.article", "match": {"value": 49}},
                {"key": "metadata.article", "match": {"value": 50}},
                {"key": "metadata.article", "match": {"value": 51}}
            ]
        }
    """
    should_conditions = []

    # Add target article + context range
    for offset in range(-context_range, context_range + 1):
        target_article = article_num + offset
        if target_article > 0:  # Articles must be positive
            should_conditions.append({
                "key": "metadata.article",
                "match": {"value": target_article}
            })

    return {"should": should_conditions}


class HybridRetriever:
    """
    ✅ MIGRATED: Hybrid retriever using LangChain

    Combines semantic (vector) and keyword (BM25) search
    """

    def __init__(self, semantic_retriever: SemanticRetriever = None, enable_reranking: bool = True):
        """
        Initialize hybrid retriever

        Args:
            semantic_retriever: SemanticRetriever instance
            enable_reranking: Enable cross-encoder reranking (default True)
        """
        self.semantic_retriever = semantic_retriever or SemanticRetriever()

        # Weights for ensemble - use settings
        self.semantic_weight = getattr(settings, 'SEMANTIC_WEIGHT', 0.7)
        self.keyword_weight = getattr(settings, 'BM25_WEIGHT', 0.3)

        # BM25 retriever (initialized later with documents)
        self.bm25_retriever = None
        self.ensemble_retriever = None

        # Vector store reference for loading documents
        self._vector_store = None
        if semantic_retriever and hasattr(semantic_retriever, 'vector_store'):
            self._vector_store = semantic_retriever.vector_store

        # Reranking configuration
        self.enable_reranking = enable_reranking
        self._reranker = None
        if enable_reranking:
            self._reranker = get_reranker()

        # Metrics
        self.metrics = {
            'total_searches': 0,
            'semantic_only': 0,
            'hybrid_searches': 0,
            'reranked_searches': 0
        }

        rerank_status = "enabled" if self._reranker else "disabled"
        logger.info(f"✅ Hybrid Retriever initialized (LangChain Ensemble, reranking: {rerank_status})")

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

    def load_bm25_from_qdrant(self, document_ids: Optional[List[int]] = None, limit: int = 10000):
        """
        Load documents from Qdrant and initialize BM25

        This should be called after documents are uploaded to enable hybrid search.

        Args:
            document_ids: Filter to specific document IDs (None = all)
            limit: Maximum number of chunks to load
        """
        try:
            if not self._vector_store:
                logger.warning("Vector store not available, cannot load BM25")
                return

            logger.info(f"Loading documents from Qdrant for BM25 (limit={limit})...")

            # Get all points from Qdrant
            client = self._vector_store.client
            collection_name = self._vector_store.collection_name

            # Scroll through all points
            documents = []
            offset = None

            while True:
                results, offset = client.scroll(
                    collection_name=collection_name,
                    limit=min(100, limit - len(documents)),
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                if not results:
                    break

                for point in results:
                    payload = point.payload or {}
                    doc_id = payload.get('document_id')

                    # Filter by document_ids if specified
                    if document_ids and doc_id not in document_ids:
                        continue

                    content = payload.get('content', '')
                    if content:
                        doc = LangChainDocument(
                            page_content=content,
                            metadata={
                                'document_id': doc_id,
                                'chunk_index': payload.get('chunk_index', 0),
                                **payload.get('metadata', {})
                            }
                        )
                        documents.append(doc)

                if len(documents) >= limit or offset is None:
                    break

            if documents:
                logger.info(f"Loaded {len(documents)} documents from Qdrant")
                self.initialize_bm25(documents)
            else:
                logger.warning("No documents found in Qdrant for BM25")

        except Exception as e:
            logger.error(f"Error loading BM25 from Qdrant: {e}")

    def ensure_bm25_initialized(self, document_ids: Optional[List[int]] = None):
        """
        Ensure BM25 is initialized, loading from Qdrant if needed

        Args:
            document_ids: Filter to specific document IDs
        """
        if self.bm25_retriever is None:
            logger.info("BM25 not initialized, loading from Qdrant...")
            self.load_bm25_from_qdrant(document_ids)

    def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[int]] = None,
        use_hybrid: bool = True,
        use_reranking: bool = None,
        enable_smart_filtering: bool = True  # ✅ NEW: Smart filtering for articles
    ) -> List[Dict[str, Any]]:
        """
        ✅ ENHANCED: Hybrid search with Smart Filtering + Boosting

        Pipeline:
        1. Detect article from query → Build context filter (article ± 1)
        2. Retrieve candidates via Ensemble (semantic + BM25) with filter
        3. Boost main article scores × 100
        4. Rerank with cross-encoder for better relevance
        5. Return top_k results

        Args:
            query: Search query
            top_k: Number of results
            document_ids: Filter by document IDs
            use_hybrid: Use hybrid search (True) or semantic only (False)
            use_reranking: Enable reranking (None = use default)
            enable_smart_filtering: Enable article-aware filtering (default: True)

        Returns:
            List of search results
        """
        try:
            self.metrics['total_searches'] += 1

            # ✅ NEW: Detect article EARLY for smart filtering
            article_num = extract_article_number(query) if enable_smart_filtering else None
            metadata_filter = None

            if article_num:
                logger.info(f"🎯 Detected article query: Điều {article_num} - applying smart filter (±1 context)")
                # Build filter for article ± 1 (context: article-1, article, article+1)
                metadata_filter = build_article_context_filter(article_num, context_range=1)

            # Determine if reranking should be used
            should_rerank = use_reranking if use_reranking is not None else self.enable_reranking
            should_rerank = should_rerank and self._reranker is not None

            # Fetch more candidates if reranking (3x for better rerank quality)
            fetch_k = top_k * 3 if should_rerank else top_k

            # ✅ ENHANCED: Force native search if smart filtering is enabled
            # BM25/LangChain retrievers don't support metadata filtering properly
            force_native = metadata_filter is not None

            # Choose retriever (check force_native FIRST before initializing BM25)
            if force_native:
                # Force native search for metadata filtering
                retriever = None
                logger.debug("🔒 Forcing native search for metadata filtering")
            else:
                # ✅ Ensure BM25 is initialized for hybrid search
                if use_hybrid and self.ensemble_retriever is None:
                    self.ensure_bm25_initialized(document_ids)

                if use_hybrid and self.ensemble_retriever:
                    retriever = self.ensemble_retriever
                    self.metrics['hybrid_searches'] += 1
                    logger.debug("Using hybrid search (semantic + BM25)")
                elif self.semantic_retriever.retriever:
                    retriever = self.semantic_retriever.retriever
                    self.metrics['semantic_only'] += 1
                    logger.debug("Using semantic search only (LangChain)")
                else:
                    retriever = None

            if not retriever:
                # ✅ Fallback: Use native vector store search
                logger.debug("Using native vector store search (no LangChain retriever)")
                self.metrics['semantic_only'] += 1

                # Get embedding service from semantic retriever
                embedding_service = self.semantic_retriever.embedding_service
                if not embedding_service or not self._vector_store:
                    logger.error("No retriever or vector store available")
                    return []

                # ✅ FIX: Lower threshold when using metadata filter
                # Article filters already narrow down results, so we don't need high score threshold
                effective_threshold = 0.0 if metadata_filter else settings.SEMANTIC_THRESHOLD
                if metadata_filter:
                    logger.debug(f"Using min_score=0.0 (no threshold) due to metadata filter")
                else:
                    logger.debug(f"Using min_score={settings.SEMANTIC_THRESHOLD}")

                # Generate embedding and search directly
                query_embedding = embedding_service.encode_single_text(query)
                native_results = self._vector_store.search_similar(
                    query_embedding=query_embedding,
                    top_k=fetch_k,
                    document_ids=document_ids,
                    min_score=effective_threshold,
                    metadata_filter=metadata_filter
                )

                # Apply reranking if enabled
                if should_rerank and native_results:
                    native_results = self._reranker.rerank(
                        query=query,
                        documents=native_results,
                        top_k=top_k,
                        return_scores=True
                    )
                    self.metrics['reranked_searches'] += 1
                    for result in native_results:
                        result['retrieval_method'] = 'semantic+rerank'
                else:
                    native_results = native_results[:top_k]

                return native_results

            if not retriever:
                logger.error("No retriever available")
                return []

            # ✅ ENHANCED: Build combined filter (document_ids + metadata filter)
            if hasattr(retriever, 'search_kwargs'):
                combined_filter = self._build_combined_filter(document_ids, metadata_filter)
                if combined_filter:
                    retriever.search_kwargs['filter'] = combined_filter
                    if metadata_filter:
                        logger.debug(f"Applied smart filter: {combined_filter}")
                    else:
                        logger.debug(f"Applied document filter: document_ids={document_ids}")

            # ✅ Retrieve documents (fetch more for reranking)
            docs = retriever.get_relevant_documents(query)

            # Convert to expected format (backward compatible)
            results = []
            for i, doc in enumerate(docs[:fetch_k]):
                result = {
                    'content': doc.page_content,
                    'document_id': doc.metadata.get('document_id'),
                    'chunk_index': doc.metadata.get('chunk_index', 0),
                    'metadata': doc.metadata,
                    'score': doc.metadata.get('score', 1.0),
                    'retrieval_method': 'hybrid' if use_hybrid else 'semantic',
                    'initial_rank': i + 1
                }
                results.append(result)

            # ✅ ENHANCED: Apply article-aware boosting BEFORE reranking
            # Note: article_num already detected earlier if enable_smart_filtering=True
            if article_num and results:
                filter_msg = "with smart filter" if metadata_filter else "without filter"
                logger.info(f"📊 Applying article boosting for Điều {article_num} ({filter_msg})")

                boosted_count = 0
                for result in results:
                    result_article = result.get('metadata', {}).get('article')

                    if result_article == article_num:
                        # Boost matching articles by 100x
                        original_score = result.get('score', 0)
                        result['score'] = original_score * 100.0
                        result['article_boosted'] = True
                        result['original_score'] = original_score
                        boosted_count += 1
                        logger.debug(f"  ✅ Boosted Article {article_num}: "
                                   f"{original_score:.4f} → {result['score']:.4f}")

                if boosted_count > 0:
                    # Re-sort by boosted scores
                    results.sort(key=lambda x: x.get('score', 0), reverse=True)
                    logger.info(f"  ✅ Boosted {boosted_count} chunks for Điều {article_num}")

                    # Update retrieval method to indicate filtering + boosting
                    for result in results:
                        if result.get('article_boosted'):
                            current_method = result.get('retrieval_method', 'unknown')
                            if metadata_filter:
                                result['retrieval_method'] = f"{current_method}+article_filter+boost"
                            else:
                                result['retrieval_method'] = f"{current_method}+article_boost"
                else:
                    logger.warning(f"  ⚠️  No chunks found with article={article_num} in results!")

            # ✅ Apply reranking if enabled
            if should_rerank and results:
                logger.debug(f"Reranking {len(results)} candidates...")
                results = self._reranker.rerank(
                    query=query,
                    documents=results,
                    top_k=top_k,
                    return_scores=True
                )
                self.metrics['reranked_searches'] += 1

                # Update retrieval method
                for result in results:
                    # Preserve article filter/boost indicators if present
                    current_method = result.get('retrieval_method', '')
                    if 'article_filter+boost' in current_method:
                        # Already has filter+boost, just add rerank
                        result['retrieval_method'] = current_method.replace(
                            '+article_filter+boost', '') + '+article_filter+boost+rerank'
                    elif 'article_boost' in current_method:
                        # Has boost only, add rerank
                        result['retrieval_method'] = current_method.replace(
                            '+article_boost', '') + '+article_boost+rerank'
                    else:
                        # No article processing, standard rerank
                        result['retrieval_method'] = 'hybrid+rerank' if use_hybrid else 'semantic+rerank'

                logger.debug(f"Reranked to {len(results)} results")
            else:
                # Just trim to top_k if no reranking
                results = results[:top_k]

            logger.debug(f"Retrieved {len(results)} documents (reranking: {should_rerank})")
            return results

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to native vector store search
            try:
                if self._vector_store and self.semantic_retriever.embedding_service:
                    query_embedding = self.semantic_retriever.embedding_service.encode_single_text(query)
                    return self._vector_store.search_similar(
                        query_embedding=query_embedding,
                        top_k=top_k,
                        document_ids=document_ids,
                        min_score=settings.SEMANTIC_THRESHOLD
                    )
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
            return []

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

    def _build_combined_filter(
        self,
        document_ids: Optional[List[int]],
        metadata_filter: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        ✅ NEW: Build combined Qdrant filter from document_ids + metadata filter

        Combines:
        - document_ids filter (if provided)
        - metadata filter like article context (if provided)

        Args:
            document_ids: List of document IDs to filter
            metadata_filter: Metadata filter dict (e.g., article context)

        Returns:
            Combined Qdrant filter dict or None

        Example:
            >>> _build_combined_filter([1, 2], {"should": [{"key": "metadata.article", "match": {"value": 50}}]})
            {
                "must": [
                    {"should": [
                        {"key": "document_id", "match": {"value": 1}},
                        {"key": "document_id", "match": {"value": 2}}
                    ]},
                    {"should": [{"key": "metadata.article", "match": {"value": 50}}]}
                ]
            }
        """
        if not document_ids and not metadata_filter:
            return None

        # Build document_ids filter
        doc_filter = None
        if document_ids:
            if len(document_ids) == 1:
                doc_filter = {
                    "must": [
                        {"key": "document_id", "match": {"value": document_ids[0]}}
                    ]
                }
            else:
                doc_filter = {
                    "should": [
                        {"key": "document_id", "match": {"value": doc_id}}
                        for doc_id in document_ids
                    ]
                }

        # Combine filters with AND logic
        if doc_filter and metadata_filter:
            # Both filters: combine with must (AND)
            combined = {"must": []}

            # Add document filter
            if "must" in doc_filter:
                combined["must"].extend(doc_filter["must"])
            else:
                combined["must"].append(doc_filter)

            # Add metadata filter
            if "must" in metadata_filter:
                combined["must"].extend(metadata_filter["must"])
            elif "should" in metadata_filter:
                # Wrap should in a single condition
                combined["must"].append(metadata_filter)
            else:
                combined["must"].append(metadata_filter)

            return combined

        # Return whichever is available
        return metadata_filter if metadata_filter else doc_filter

    def get_metrics(self) -> Dict[str, Any]:
        """Get retriever metrics"""
        metrics = self.metrics.copy()
        metrics['semantic_retriever_metrics'] = self.semantic_retriever.get_metrics()
        metrics['bm25_initialized'] = self.bm25_retriever is not None
        metrics['ensemble_initialized'] = self.ensemble_retriever is not None
        return metrics

    def get_model_info(self) -> Dict[str, Any]:
        """Get retriever information"""
        info = {
            'type': 'hybrid',
            'using_langchain': True,
            'ensemble_enabled': self.ensemble_retriever is not None,
            'semantic_weight': self.semantic_weight,
            'keyword_weight': self.keyword_weight,
            'bm25_initialized': self.bm25_retriever is not None,
            'reranking_enabled': self.enable_reranking and self._reranker is not None
        }

        # Add reranker info if available
        if self._reranker:
            info['reranker'] = self._reranker.get_model_info()

        return info

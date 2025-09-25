import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

from src.rag.embedding_service import EmbeddingService
from src.rag.vector_store import QdrantVectorStore
from src.utils.database import get_db_session, Document, DocumentChunk
from src.utils.exceptions import VectorStoreError

logger = logging.getLogger(__name__)

class SemanticRetriever:
    """Advanced semantic retrieval strategies for Phase 2"""
    
    def __init__(self, embedding_service: EmbeddingService = None, vector_store: QdrantVectorStore = None):
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or QdrantVectorStore()
        
        # Advanced retrieval parameters
        self.default_top_k = 10
        self.semantic_threshold = 0.3
        self.diversity_threshold = 0.85  # For diversity filtering
        self.rerank_top_k = 20  # Retrieve more, then rerank
    
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
        Advanced semantic search with reranking and diversity
        
        Args:
            query: Search query
            document_ids: Optional document filter
            top_k: Number of results to return
            min_score: Minimum similarity score
            enable_reranking: Apply semantic reranking
            enable_diversity: Apply diversity filtering
            
        Returns:
            List of search results with enhanced metadata
        """
        try:
            top_k = top_k or self.default_top_k
            min_score = min_score or self.semantic_threshold
            
            logger.info(f"Semantic search: '{query[:50]}...' (top_k={top_k})")
            
            # Step 1: Generate query embedding
            query_embedding = self.embedding_service.encode_single_text(query)
            
            # Step 2: Initial retrieval (get more for reranking)
            search_top_k = self.rerank_top_k if enable_reranking else top_k
            
            raw_results = self.vector_store.search_similar(
                query_embedding=query_embedding,
                top_k=search_top_k,
                document_ids=document_ids,
                min_score=min_score
            )
            
            if not raw_results:
                return []
            
            # Step 3: Enhance results with metadata
            enhanced_results = self._enhance_search_results(raw_results, query)
            
            # Step 4: Apply reranking if enabled
            if enable_reranking and len(enhanced_results) > top_k:
                enhanced_results = self._rerank_results(enhanced_results, query, query_embedding)
            
            # Step 5: Apply diversity filtering if enabled
            if enable_diversity:
                enhanced_results = self._apply_diversity_filtering(enhanced_results)
            
            # Step 6: Final top-k filtering
            final_results = enhanced_results[:top_k]
            
            logger.info(f"Semantic search completed: {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            raise VectorStoreError(f"Semantic search failed: {e}")
    
    def multi_query_search(
        self,
        queries: List[str],
        document_ids: Optional[List[int]] = None,
        top_k: int = None,
        fusion_method: str = "rrf"  # rrf, weighted, max
    ) -> List[Dict[str, Any]]:
        """
        Multi-query search with result fusion
        
        Args:
            queries: List of search queries
            document_ids: Optional document filter
            top_k: Number of results to return
            fusion_method: Method for fusing results (rrf, weighted, max)
            
        Returns:
            Fused search results
        """
        try:
            top_k = top_k or self.default_top_k
            
            logger.info(f"Multi-query search: {len(queries)} queries")
            
            # Get results for each query
            all_results = []
            for i, query in enumerate(queries):
                results = self.semantic_search(
                    query=query,
                    document_ids=document_ids,
                    top_k=top_k * 2,  # Get more for fusion
                    enable_reranking=False  # Apply reranking after fusion
                )
                
                # Add query index to results
                for result in results:
                    result['query_index'] = i
                    result['query_text'] = query
                
                all_results.extend(results)
            
            # Fuse results
            if fusion_method == "rrf":
                fused_results = self._reciprocal_rank_fusion(all_results, len(queries))
            elif fusion_method == "weighted":
                fused_results = self._weighted_fusion(all_results, queries)
            else:  # max
                fused_results = self._max_score_fusion(all_results)
            
            # Final ranking and top-k
            final_results = fused_results[:top_k]
            
            logger.info(f"Multi-query search completed: {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in multi-query search: {e}")
            raise VectorStoreError(f"Multi-query search failed: {e}")
    
    def contextual_search(
        self,
        query: str,
        conversation_history: List[str] = None,
        document_ids: Optional[List[int]] = None,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Contextual search using conversation history
        
        Args:
            query: Current search query
            conversation_history: Previous queries/responses
            document_ids: Optional document filter
            top_k: Number of results to return
            
        Returns:
            Contextually enhanced search results
        """
        try:
            top_k = top_k or self.default_top_k
            
            # Build contextual query
            if conversation_history:
                # Combine recent conversation context with current query
                context = " ".join(conversation_history[-3:])  # Last 3 exchanges
                contextual_query = f"{context} {query}"
                
                logger.info(f"Contextual search with history: '{query[:30]}...'")
            else:
                contextual_query = query
            
            # Perform enhanced search
            results = self.semantic_search(
                query=contextual_query,
                document_ids=document_ids,
                top_k=top_k,
                enable_reranking=True,
                enable_diversity=True
            )
            
            # Add context relevance scores
            for result in results:
                result['context_enhanced'] = bool(conversation_history)
                result['original_query'] = query
            
            return results
            
        except Exception as e:
            logger.error(f"Error in contextual search: {e}")
            return self.semantic_search(query, document_ids, top_k)
    
    def _enhance_search_results(
        self, 
        raw_results: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """Enhance search results with additional metadata"""
        try:
            # Get document metadata
            document_ids = list(set(result['document_id'] for result in raw_results))
            document_metadata = self._get_document_metadata(document_ids)
            
            enhanced_results = []
            for result in raw_results:
                doc_id = result['document_id']
                doc_meta = document_metadata.get(doc_id, {})
                
                enhanced_result = {
                    **result,
                    'document_filename': doc_meta.get('filename', 'Unknown'),
                    'document_type': doc_meta.get('file_type', 'Unknown'),
                    'document_size': doc_meta.get('file_size', 0),
                    'upload_date': doc_meta.get('upload_date'),
                    'chunk_position': self._calculate_chunk_position(result, doc_meta),
                    'content_preview': self._create_content_preview(result['content'], query),
                    'semantic_score': result['score'],
                    'retrieval_timestamp': datetime.now().isoformat()
                }
                
                enhanced_results.append(enhanced_result)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Error enhancing search results: {e}")
            return raw_results
    
    def _get_document_metadata(self, document_ids: List[int]) -> Dict[int, Dict[str, Any]]:
        """Get metadata for documents"""
        try:
            db = get_db_session()
            documents = db.query(Document).filter(Document.id.in_(document_ids)).all()
            
            metadata = {}
            for doc in documents:
                metadata[doc.id] = {
                    'filename': doc.original_filename,
                    'file_type': doc.file_type,
                    'file_size': doc.file_size,
                    'upload_date': doc.upload_date,
                    'total_chunks': doc.total_chunks,
                    'summary': doc.summary
                }
            
            db.close()
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting document metadata: {e}")
            return {}
    
    def _calculate_chunk_position(
        self, 
        result: Dict[str, Any], 
        doc_meta: Dict[str, Any]
    ) -> float:
        """Calculate relative position of chunk in document (0-1)"""
        try:
            chunk_index = result.get('chunk_index', 0)
            total_chunks = doc_meta.get('total_chunks', 1)
            return chunk_index / max(total_chunks - 1, 1)
        except:
            return 0.0
    
    def _create_content_preview(self, content: str, query: str) -> str:
        """Create content preview highlighting query terms"""
        try:
            # Simple highlighting (can be enhanced with proper NLP)
            query_terms = query.lower().split()
            preview_length = 300
            
            # Find best snippet containing query terms
            content_lower = content.lower()
            best_pos = 0
            max_matches = 0
            
            for i in range(0, len(content) - preview_length, 50):
                snippet = content_lower[i:i + preview_length]
                matches = sum(1 for term in query_terms if term in snippet)
                if matches > max_matches:
                    max_matches = matches
                    best_pos = i
            
            preview = content[best_pos:best_pos + preview_length]
            if best_pos > 0:
                preview = "..." + preview
            if best_pos + preview_length < len(content):
                preview = preview + "..."
            
            return preview
            
        except Exception as e:
            logger.error(f"Error creating content preview: {e}")
            return content[:300] + ("..." if len(content) > 300 else "")
    
    def _rerank_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        query_embedding: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Rerank results using additional signals"""
        try:
            # Simple reranking based on multiple factors
            for result in results:
                semantic_score = result['semantic_score']
                
                # Boost score based on chunk position (early chunks often more important)
                position_boost = 1.0 - (result['chunk_position'] * 0.1)
                
                # Boost score based on content length (not too short, not too long)
                content_len = len(result['content'])
                optimal_length = 800  # Optimal chunk length
                length_score = 1.0 - abs(content_len - optimal_length) / optimal_length * 0.1
                
                # Combined reranking score
                rerank_score = semantic_score * position_boost * length_score
                result['rerank_score'] = rerank_score
            
            # Sort by reranking score
            results.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            return results
            
        except Exception as e:
            logger.error(f"Error reranking results: {e}")
            return results
    
    def _apply_diversity_filtering(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply diversity filtering to avoid redundant results"""
        try:
            if len(results) <= 3:
                return results
            
            diverse_results = [results[0]]  # Always include top result
            
            for result in results[1:]:
                # Check similarity with already selected results
                is_diverse = True
                for selected in diverse_results:
                    if (result['document_id'] == selected['document_id'] and 
                        abs(result['chunk_index'] - selected['chunk_index']) <= 2):
                        # Too similar (same doc, nearby chunks)
                        is_diverse = False
                        break
                
                if is_diverse:
                    diverse_results.append(result)
                
                # Limit diversity filtering to avoid over-filtering
                if len(diverse_results) >= len(results) * 0.8:
                    break
            
            return diverse_results
            
        except Exception as e:
            logger.error(f"Error applying diversity filtering: {e}")
            return results
    
    def _reciprocal_rank_fusion(
        self, 
        all_results: List[Dict[str, Any]], 
        num_queries: int,
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """Apply Reciprocal Rank Fusion (RRF) to combine results"""
        try:
            # Group results by document chunk
            result_groups = {}
            
            for result in all_results:
                key = (result['document_id'], result['chunk_index'])
                if key not in result_groups:
                    result_groups[key] = []
                result_groups[key].append(result)
            
            # Calculate RRF scores
            fused_results = []
            for key, group in result_groups.items():
                # Take best result from group
                best_result = max(group, key=lambda x: x['score'])
                
                # Calculate RRF score
                rrf_score = 0
                for result in group:
                    rank = 1  # Would need actual rank from original results
                    rrf_score += 1.0 / (k + rank)
                
                best_result['rrf_score'] = rrf_score
                best_result['query_coverage'] = len(group) / num_queries
                fused_results.append(best_result)
            
            # Sort by RRF score
            fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)
            
            return fused_results
            
        except Exception as e:
            logger.error(f"Error in RRF fusion: {e}")
            return all_results
    
    def _weighted_fusion(
        self, 
        all_results: List[Dict[str, Any]], 
        queries: List[str]
    ) -> List[Dict[str, Any]]:
        """Apply weighted fusion based on query importance"""
        try:
            # Simple implementation - can be enhanced with query importance scoring
            query_weights = [1.0] * len(queries)  # Equal weights for now
            
            # Similar to RRF but with weights
            result_groups = {}
            
            for result in all_results:
                key = (result['document_id'], result['chunk_index'])
                if key not in result_groups:
                    result_groups[key] = []
                result_groups[key].append(result)
            
            fused_results = []
            for key, group in result_groups.items():
                best_result = max(group, key=lambda x: x['score'])
                
                # Calculate weighted score
                weighted_score = 0
                for result in group:
                    query_idx = result['query_index']
                    weight = query_weights[query_idx]
                    weighted_score += result['score'] * weight
                
                best_result['weighted_score'] = weighted_score / len(group)
                fused_results.append(best_result)
            
            fused_results.sort(key=lambda x: x['weighted_score'], reverse=True)
            return fused_results
            
        except Exception as e:
            logger.error(f"Error in weighted fusion: {e}")
            return all_results
    
    def _max_score_fusion(self, all_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply max score fusion"""
        try:
            result_groups = {}
            
            for result in all_results:
                key = (result['document_id'], result['chunk_index'])
                if key not in result_groups:
                    result_groups[key] = []
                result_groups[key].append(result)
            
            fused_results = []
            for key, group in result_groups.items():
                # Take result with maximum score
                best_result = max(group, key=lambda x: x['score'])
                best_result['max_score'] = best_result['score']
                fused_results.append(best_result)
            
            fused_results.sort(key=lambda x: x['max_score'], reverse=True)
            return fused_results
            
        except Exception as e:
            logger.error(f"Error in max score fusion: {e}")
            return all_results
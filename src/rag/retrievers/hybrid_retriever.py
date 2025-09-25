import logging
from typing import List, Dict, Any, Optional, Set
import re
from collections import Counter
import math

from src.rag.retrievers.semantic_retriever import SemanticRetriever
from src.utils.database import get_db_session, DocumentChunk
from src.utils.exceptions import VectorStoreError

logger = logging.getLogger(__name__)

class HybridRetriever:
    """Hybrid retrieval combining semantic and keyword-based search for Phase 2"""
    
    def __init__(self, semantic_retriever: SemanticRetriever = None):
        self.semantic_retriever = semantic_retriever or SemanticRetriever()
        
        # Hybrid search parameters
        self.semantic_weight = 0.7  # Weight for semantic scores
        self.keyword_weight = 0.3   # Weight for keyword scores
        self.min_keyword_matches = 1
        self.keyword_boost_factor = 1.5
        
        # Stop words for keyword filtering
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'will', 'would', 'could', 'should', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
        }
    
    def hybrid_search(
        self,
        query: str,
        document_ids: Optional[List[int]] = None,
        top_k: int = 10,
        semantic_weight: float = None,
        keyword_weight: float = None,
        enable_query_expansion: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining semantic and keyword-based retrieval
        
        Args:
            query: Search query
            document_ids: Optional document filter
            top_k: Number of results to return
            semantic_weight: Weight for semantic scores (0-1)
            keyword_weight: Weight for keyword scores (0-1)
            enable_query_expansion: Enable automatic query expansion
            
        Returns:
            Hybrid search results with combined scores
        """
        try:
            # Use provided weights or defaults
            sem_weight = semantic_weight or self.semantic_weight
            kw_weight = keyword_weight or self.keyword_weight
            
            # Normalize weights
            total_weight = sem_weight + kw_weight
            sem_weight /= total_weight
            kw_weight /= total_weight
            
            logger.info(f"Hybrid search: '{query[:50]}...' (sem={sem_weight:.2f}, kw={kw_weight:.2f})")
            
            # Step 1: Expand query if enabled
            if enable_query_expansion:
                expanded_queries = self._expand_query(query)
            else:
                expanded_queries = [query]
            
            # Step 2: Semantic search
            semantic_results = []
            for expanded_query in expanded_queries:
                results = self.semantic_retriever.semantic_search(
                    query=expanded_query,
                    document_ids=document_ids,
                    top_k=top_k * 2,  # Get more for hybrid fusion
                    enable_reranking=False  # We'll do our own reranking
                )
                semantic_results.extend(results)
            
            # Step 3: Keyword search
            keyword_results = self._keyword_search(
                query=query,
                document_ids=document_ids,
                top_k=top_k * 2
            )
            
            # Step 4: Combine and score results
            hybrid_results = self._combine_results(
                semantic_results=semantic_results,
                keyword_results=keyword_results,
                semantic_weight=sem_weight,
                keyword_weight=kw_weight,
                original_query=query
            )
            
            # Step 5: Final ranking and filtering
            final_results = hybrid_results[:top_k]
            
            logger.info(f"Hybrid search completed: {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            raise VectorStoreError(f"Hybrid search failed: {e}")
    
    def adaptive_hybrid_search(
        self,
        query: str,
        document_ids: Optional[List[int]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Adaptive hybrid search that adjusts weights based on query characteristics
        
        Args:
            query: Search query
            document_ids: Optional document filter
            top_k: Number of results to return
            
        Returns:
            Adaptively weighted hybrid search results
        """
        try:
            # Analyze query to determine optimal weights
            query_analysis = self._analyze_query(query)
            
            # Adapt weights based on query characteristics
            if query_analysis['is_specific']:
                # Specific queries benefit more from keyword search
                semantic_weight = 0.5
                keyword_weight = 0.5
            elif query_analysis['is_conceptual']:
                # Conceptual queries benefit more from semantic search
                semantic_weight = 0.8
                keyword_weight = 0.2
            else:
                # Balanced approach for general queries
                semantic_weight = 0.7
                keyword_weight = 0.3
            
            logger.info(f"Adaptive weights - semantic: {semantic_weight}, keyword: {keyword_weight}")
            
            return self.hybrid_search(
                query=query,
                document_ids=document_ids,
                top_k=top_k,
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight
            )
            
        except Exception as e:
            logger.error(f"Error in adaptive hybrid search: {e}")
            return self.hybrid_search(query, document_ids, top_k)
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        try:
            expanded_queries = [query]
            
            # Simple query expansion (can be enhanced with word embeddings)
            query_lower = query.lower()
            
            # Add common variations
            expansions = {
                'ai': ['artificial intelligence', 'machine learning', 'AI'],
                'ml': ['machine learning', 'artificial intelligence'],
                'llm': ['large language model', 'language model'],
                'cpu': ['processor', 'central processing unit'],
                'gpu': ['graphics card', 'graphics processing unit'],
                'api': ['application programming interface', 'interface'],
                'db': ['database', 'data storage'],
                'ui': ['user interface', 'interface'],
                'ux': ['user experience', 'experience']
            }
            
            for term, synonyms in expansions.items():
                if term in query_lower:
                    for synonym in synonyms:
                        if synonym not in query_lower:
                            expanded_query = query_lower.replace(term, synonym)
                            expanded_queries.append(expanded_query)
            
            return expanded_queries[:3]  # Limit to 3 queries
            
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return [query]
    
    def _keyword_search(
        self,
        query: str,
        document_ids: Optional[List[int]] = None,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Keyword-based search using TF-IDF scoring
        
        Args:
            query: Search query
            document_ids: Optional document filter
            top_k: Number of results to return
            
        Returns:
            Keyword search results with TF-IDF scores
        """
        try:
            # Extract keywords from query
            keywords = self._extract_keywords(query)
            if not keywords:
                return []
            
            logger.info(f"Keyword search: {keywords}")
            
            # Get chunks from database
            chunks = self._get_chunks_for_keyword_search(document_ids)
            
            if not chunks:
                return []
            
            # Calculate TF-IDF scores for each chunk
            keyword_results = []
            
            for chunk in chunks:
                tf_idf_score = self._calculate_tf_idf(chunk['content'], keywords, chunks)
                
                if tf_idf_score > 0:
                    result = {
                        'id': chunk['id'],
                        'document_id': chunk['document_id'],
                        'chunk_index': chunk['chunk_index'],
                        'content': chunk['content'],
                        'metadata': chunk['metadata'] or {},
                        'score': tf_idf_score,
                        'keyword_matches': self._count_keyword_matches(chunk['content'], keywords),
                        'search_type': 'keyword'
                    }
                    keyword_results.append(result)
            
            # Sort by TF-IDF score
            keyword_results.sort(key=lambda x: x['score'], reverse=True)
            
            return keyword_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in keyword search: {e}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query"""
        try:
            # Basic preprocessing
            query_lower = query.lower()
            
            # Remove punctuation and split
            words = re.findall(r'\b\w+\b', query_lower)
            
            # Filter out stop words and short words
            keywords = [
                word for word in words 
                if word not in self.stop_words and len(word) > 2
            ]
            
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _get_chunks_for_keyword_search(
        self, 
        document_ids: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """Get chunks from database for keyword search"""
        try:
            db = get_db_session()
            
            query = db.query(DocumentChunk)
            
            if document_ids:
                query = query.filter(DocumentChunk.document_id.in_(document_ids))
            
            chunks = query.all()
            
            chunk_list = []
            for chunk in chunks:
                chunk_dict = {
                    'id': str(chunk.id),
                    'document_id': chunk.document_id,
                    'chunk_index': chunk.chunk_index,
                    'content': chunk.content,
                    'metadata': chunk.chunk_metadata
                }
                chunk_list.append(chunk_dict)
            
            db.close()
            return chunk_list
            
        except Exception as e:
            logger.error(f"Error getting chunks for keyword search: {e}")
            return []
    
    def _calculate_tf_idf(
        self, 
        content: str, 
        keywords: List[str], 
        all_chunks: List[Dict[str, Any]]
    ) -> float:
        """Calculate TF-IDF score for content"""
        try:
            content_lower = content.lower()
            total_docs = len(all_chunks)
            
            tf_idf_score = 0.0
            
            for keyword in keywords:
                # Term Frequency (TF)
                tf = content_lower.count(keyword) / len(content_lower.split())
                
                # Document Frequency (DF)
                df = sum(1 for chunk in all_chunks if keyword in chunk['content'].lower())
                
                # Inverse Document Frequency (IDF)
                if df > 0:
                    idf = math.log(total_docs / df)
                    tf_idf_score += tf * idf
            
            return tf_idf_score
            
        except Exception as e:
            logger.error(f"Error calculating TF-IDF: {e}")
            return 0.0
    
    def _count_keyword_matches(self, content: str, keywords: List[str]) -> int:
        """Count keyword matches in content"""
        try:
            content_lower = content.lower()
            return sum(1 for keyword in keywords if keyword in content_lower)
        except:
            return 0
    
    def _analyze_query(self, query: str) -> Dict[str, bool]:
        """Analyze query characteristics to determine search strategy"""
        try:
            query_lower = query.lower()
            
            # Check for specific patterns
            is_specific = any([
                # Specific terms or technical jargon
                bool(re.search(r'\b\w+\.\w+\b', query)),  # e.g., "config.json"
                bool(re.search(r'\b[A-Z]{2,}\b', query)),  # e.g., "API", "CPU"
                any(char.isdigit() for char in query),      # Contains numbers
                '"' in query,                               # Quoted phrases
            ])
            
            is_conceptual = any([
                'what is' in query_lower,
                'how to' in query_lower,
                'explain' in query_lower,
                'concept' in query_lower,
                'theory' in query_lower,
                'approach' in query_lower,
                'method' in query_lower,
            ])
            
            is_question = query.strip().endswith('?')
            
            return {
                'is_specific': is_specific,
                'is_conceptual': is_conceptual,
                'is_question': is_question,
                'length': len(query.split()),
                'has_technical_terms': is_specific
            }
            
        except Exception as e:
            logger.error(f"Error analyzing query: {e}")
            return {
                'is_specific': False,
                'is_conceptual': False,
                'is_question': False,
                'length': 0,
                'has_technical_terms': False
            }
    
    def _combine_results(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        semantic_weight: float,
        keyword_weight: float,
        original_query: str
    ) -> List[Dict[str, Any]]:
        """Combine semantic and keyword results with weighted scoring"""
        try:
            # Create lookup for results by chunk
            result_lookup = {}
            
            # Add semantic results
            for result in semantic_results:
                key = (result['document_id'], result['chunk_index'])
                if key not in result_lookup:
                    result_lookup[key] = result.copy()
                    result_lookup[key]['semantic_score'] = result['score']
                    result_lookup[key]['keyword_score'] = 0.0
                    result_lookup[key]['search_types'] = ['semantic']
                else:
                    # Take best semantic score if multiple
                    if result['score'] > result_lookup[key]['semantic_score']:
                        result_lookup[key]['semantic_score'] = result['score']
            
            # Add keyword results
            for result in keyword_results:
                key = (result['document_id'], result['chunk_index'])
                if key not in result_lookup:
                    # Need to get semantic info for this chunk
                    result_lookup[key] = result.copy()
                    result_lookup[key]['semantic_score'] = 0.0
                    result_lookup[key]['keyword_score'] = result['score']
                    result_lookup[key]['search_types'] = ['keyword']
                else:
                    result_lookup[key]['keyword_score'] = result['score']
                    result_lookup[key]['search_types'].append('keyword')
                    # Update keyword-specific metadata
                    result_lookup[key]['keyword_matches'] = result.get('keyword_matches', 0)
            
            # Calculate combined scores
            combined_results = []
            for key, result in result_lookup.items():
                semantic_score = result['semantic_score']
                keyword_score = result['keyword_score']
                
                # Normalize scores (simple min-max normalization)
                norm_semantic = min(semantic_score, 1.0)  # Semantic scores are already 0-1
                norm_keyword = min(keyword_score / 10.0, 1.0) if keyword_score > 0 else 0.0
                
                # Combined score
                combined_score = (norm_semantic * semantic_weight + 
                                norm_keyword * keyword_weight)
                
                # Boost for results that appear in both searches
                if len(result['search_types']) > 1:
                    combined_score *= self.keyword_boost_factor
                
                result['combined_score'] = combined_score
                result['normalized_semantic_score'] = norm_semantic
                result['normalized_keyword_score'] = norm_keyword
                result['hybrid_search'] = True
                result['query'] = original_query
                
                combined_results.append(result)
            
            # Sort by combined score
            combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            return combined_results
            
        except Exception as e:
            logger.error(f"Error combining results: {e}")
            return semantic_results  # Fallback to semantic results
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """Get statistics about hybrid search performance"""
        try:
            # This would be enhanced with actual usage tracking
            return {
                'semantic_weight': self.semantic_weight,
                'keyword_weight': self.keyword_weight,
                'keyword_boost_factor': self.keyword_boost_factor,
                'stop_words_count': len(self.stop_words),
                'search_strategies': ['semantic', 'keyword', 'hybrid', 'adaptive']
            }
        except Exception as e:
            logger.error(f"Error getting search statistics: {e}")
            return {}
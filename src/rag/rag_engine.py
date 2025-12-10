import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time
from functools import lru_cache
from cachetools import TTLCache
import hashlib
import re

from config.settings import settings
from src.rag.embedding_service import EmbeddingService, LangChainEmbeddingAdapter
from src.rag.vector_store import QdrantVectorStore
from src.rag.retrievers.hybrid_retriever import HybridRetriever
from src.rag.retrievers.semantic_retriever import SemanticRetriever
from src.serving.llm_service import LlamaCppService  # UPDATED import
from src.serving.prompts.rag_prompts import (
    rag_chat_prompt,
    summary_prompt,
    RAG_SYSTEM_PROMPT  # Keep for tokenization
)
from src.serving.prompts.legal_prompts import (
    legal_qa_prompt,
    legal_summary_prompt,
    LEGAL_SYSTEM_PROMPT
)
from src.utils.database import get_db_session, Document, Conversation
from src.utils.models import QueryRequest, QueryResponse, SourceInfo
from src.utils.exceptions import VectorStoreError, LLMServiceError

logger = logging.getLogger(__name__)

class RAGEngine:
    """Main Retrieval-Augmented Generation Engine with llama.cpp - Enhanced for Vietnamese Legal Domain"""

    def __init__(self, use_legal_prompts: bool = True, enable_reranking: bool = True):
        try:
            logger.info("Initializing RAG Engine components...")

            self.embedding_service = EmbeddingService()
            # Use config-only dimension in VectorStore
            self.vector_store = QdrantVectorStore()
            self.llm_service = LlamaCppService()

            # ✅ Setup LangChain wrapper for semantic search
            logger.info("Setting up LangChain integration for hybrid search...")
            embedding_adapter = LangChainEmbeddingAdapter(self.embedding_service)
            self.vector_store.setup_langchain_wrapper(embedding_adapter)

            # ✅ Initialize Hybrid Retriever with reranking
            semantic_retriever = SemanticRetriever(
                embedding_service=self.embedding_service,
                vector_store=self.vector_store
            )
            self.hybrid_retriever = HybridRetriever(
                semantic_retriever=semantic_retriever,
                enable_reranking=enable_reranking
            )

            # RAG parameters
            self.default_top_k = settings.TOP_K_RESULTS
            self.max_context_length = 2500
            self.min_similarity_score = settings.SEMANTIC_THRESHOLD
            self.enable_reranking = enable_reranking

            # Legal domain configuration
            self.use_legal_prompts = use_legal_prompts

            # ✅ FIX: Add caching to prevent redundant operations
            # Query embedding cache (TTL: 1 hour, max 10000 entries)
            self._query_embedding_cache = TTLCache(maxsize=10000, ttl=3600)
            # Document metadata cache (TTL: 5 minutes, max 5000 entries)
            self._doc_metadata_cache = TTLCache(maxsize=5000, ttl=300)

            logger.info("RAG Engine with llama.cpp initialized successfully")
            logger.info(f"Legal prompts: {use_legal_prompts}, Reranking: {enable_reranking}")
            logger.info(f"Caches: query_embedding(10k,1h), doc_metadata(5k,5m)")

        except Exception as e:
            logger.error(f"Error initializing RAG Engine: {e}")
            raise
    
    def query(
        self,
        question: str,
        document_ids: Optional[List[int]] = None,
        top_k: int = 5,
        session_id: Optional[str] = None,
        db: Optional[Any] = None  # ✅ FIX: Accept optional DB session
    ) -> Dict[str, Any]:
        """
        Process user query using RAG pipeline with llama.cpp

        ✅ FIXED: Session reuse to prevent connection pool exhaustion
        """
        # ✅ FIX: Manage DB session properly
        should_close_db = False
        if db is None:
            db = get_db_session()
            should_close_db = True

        try:
            start_time = time.time()
            logger.info(f"Processing RAG query: {question[:100]}...")

            # Step 1 & 2: Retrieve using Hybrid Search + Reranking
            # This combines semantic search, BM25, and cross-encoder reranking
            logger.debug(f"Performing hybrid search with reranking (top_k={top_k})...")
            search_results = self.hybrid_retriever.search(
                query=question,
                top_k=top_k,
                document_ids=document_ids,
                use_hybrid=True,
                use_reranking=self.enable_reranking
            )

            if not search_results:
                return self._create_empty_response(
                    "I couldn't find any relevant information to answer your question.",
                    session_id, time.time() - start_time
                )

            # Step 3: Build context from retrieved chunks
            logger.debug("Building context from retrieved chunks...")
            context, sources = self._build_context_and_sources(search_results)

            # Check context length for llama.cpp
            context_usage = self.llm_service.get_context_window_usage(
                self._build_full_prompt(question, context)
            )

            if context_usage['usage_percentage'] > 90:
                logger.warning(f"Context window usage high: {context_usage['usage_percentage']:.1f}%")
                # Truncate context if needed
                context = context[:self.max_context_length]

            # Step 4: Generate response using llama.cpp
            logger.debug("Generating response with llama.cpp...")
            answer = self._generate_answer(question, context)

            # Step 5: Enhance sources with document metadata - REUSE db session
            enhanced_sources = self._enhance_sources_with_metadata(sources, db=db)

            response_time_ms = int((time.time() - start_time) * 1000)

            result = {
                'answer': answer,
                'sources': enhanced_sources,
                'total_sources': len(search_results),
                'response_time_ms': response_time_ms,
                'session_id': session_id or 'anonymous',
                'context_usage': context_usage
            }

            # Save conversation to database - REUSE db session
            if session_id:
                self._save_conversation(
                    session_id, question, answer,
                    document_ids or [], enhanced_sources, response_time_ms,
                    db=db
                )

            logger.info(f"RAG query completed in {response_time_ms}ms using llama.cpp")
            return result

        except Exception as e:
            logger.error(f"Error processing RAG query: {e}")
            if should_close_db and db:
                db.rollback()
            return self._create_error_response(str(e), session_id)
        finally:
            # ✅ FIX: Only close if we created the session
            if should_close_db and db:
                db.close()
    
    def _get_cached_query_embedding(self, question: str) -> Any:
        """
        ✅ FIX: Get query embedding with caching to prevent redundant encoding
        """
        # Create cache key from question
        cache_key = hashlib.md5(question.encode()).hexdigest()

        # Check cache first
        if cache_key in self._query_embedding_cache:
            logger.debug(f"Query embedding cache HIT for: {question[:50]}...")
            return self._query_embedding_cache[cache_key]

        # Generate embedding
        logger.debug(f"Query embedding cache MISS - generating for: {question[:50]}...")
        embedding = self.embedding_service.encode_single_text(question)

        # Store in cache
        self._query_embedding_cache[cache_key] = embedding

        return embedding

    def _build_full_prompt(self, question: str, context: str, use_legal: bool = None) -> str:
        """Build full prompt for context length estimation using LangChain template"""
        if use_legal is None:
            use_legal = self.use_legal_prompts

        if use_legal:
            return legal_qa_prompt.format(context=context, question=question)
        else:
            return rag_chat_prompt.format(context=context, question=question)
    
    def _build_context_and_sources(self, search_results: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        """Build context string and sources list from Qdrant search results"""
        context_parts = []
        sources = []
        current_length = 0
        
        for i, result in enumerate(search_results):
            # Create source info
            source = {
                'document_id': result['document_id'],
                'chunk_index': result['chunk_index'],
                'content': result['content'],
                'score': result['score'],
                'metadata': result.get('metadata', {})
            }
            sources.append(source)
            
            # Add to context (with length limit)
            chunk_text = result['content']
            source_text = f"[Source {i+1}] {chunk_text}"
            
            if current_length + len(source_text) > self.max_context_length:
                # Truncate if needed
                remaining_length = self.max_context_length - current_length - 20
                if remaining_length > 100:  # Only add if meaningful length remains
                    truncated_text = chunk_text[:remaining_length] + "..."
                    context_parts.append(f"[Source {i+1}] {truncated_text}")
                break
            
            context_parts.append(source_text)
            current_length += len(source_text)
        
        context = "\n\n".join(context_parts)
        return context, sources
    
    def _generate_answer(self, question: str, context: str, use_legal: bool = None) -> str:
        """
        ✅ FIXED: Generate answer with optimized tokenization (tokenize once)
        Enhanced: Supports legal-specific prompts for Vietnamese legal documents
        """
        try:
            if use_legal is None:
                use_legal = self.use_legal_prompts

            max_new_tokens = 1024

            # ✅ FIX: Tokenize context and question ONCE
            context_tokens = self.llm_service.tokenize(context)
            question_tokens = self.llm_service.tokenize(question)

            # Use appropriate system prompt
            system_prompt = LEGAL_SYSTEM_PROMPT if use_legal else RAG_SYSTEM_PROMPT
            system_tokens = self.llm_service.tokenize(system_prompt)

            # Calculate prompt overhead (system prompt + formatting)
            # Approximate: system + "Context:" + "Question:" + "Answer:" + newlines
            formatting_overhead = 50  # Approximate tokens for formatting
            prompt_overhead = len(system_tokens) + len(question_tokens) + formatting_overhead

            # Calculate available space for context
            total_available = self.llm_service.context_length
            safety_margin = 100
            available_context_tokens = total_available - prompt_overhead - max_new_tokens - safety_margin

            # ✅ FIX: Truncate tokens directly if needed (no re-tokenization!)
            if len(context_tokens) > available_context_tokens:
                if available_context_tokens > 200:
                    logger.warning(f"Truncating context: {len(context_tokens)} → {available_context_tokens} tokens")

                    # Smart truncation: keep first 60% and last 40%
                    first_part_tokens = int(available_context_tokens * 0.6)
                    last_part_tokens = int(available_context_tokens * 0.4)

                    # Extract first and last parts directly from tokens
                    first_part = self.llm_service.detokenize(context_tokens[:first_part_tokens])
                    last_part = self.llm_service.detokenize(context_tokens[-last_part_tokens:])

                    # Combine with ellipsis
                    context = f"{first_part}\n\n[... content truncated for length ...]\n\n{last_part}"

                    logger.info(f"Context truncated: {len(context_tokens)} → {first_part_tokens + last_part_tokens} tokens")
                else:
                    # Use minimal context
                    logger.warning(f"Very limited context space: {available_context_tokens} tokens")
                    truncated_tokens = context_tokens[:max(available_context_tokens, 100)]
                    context = self.llm_service.detokenize(truncated_tokens)

            # Build final prompt (only once!) using appropriate LangChain template
            if use_legal:
                prompt = legal_qa_prompt.format(context=context, question=question)
                logger.debug("Using legal-specific prompt for Vietnamese legal documents")
            else:
                prompt = rag_chat_prompt.format(context=context, question=question)
                logger.debug("Using standard RAG prompt")

            # Generate response
            answer = self.llm_service.generate_response(
                prompt=prompt,
                max_tokens=max_new_tokens,
                temperature=0.1
            )

            return answer or "Xin lỗi, tôi không thể tạo câu trả lời." if use_legal else "I apologize, but I couldn't generate a response."

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            error_msg = f"Tôi gặp lỗi khi tạo câu trả lời: {str(e)}" if use_legal else f"I encountered an error while generating the response: {str(e)}"
            return error_msg
    
    def _enhance_sources_with_metadata(
        self,
        sources: List[Dict[str, Any]],
        db: Optional[Any] = None  # ✅ FIX: Accept DB session
    ) -> List[Dict[str, Any]]:
        """
        ✅ FIXED: Add document metadata to sources with session reuse and caching
        """
        if not sources:
            return []

        # ✅ FIX: Manage DB session properly
        should_close_db = False
        if db is None:
            db = get_db_session()
            should_close_db = True

        try:
            # Get unique document IDs
            doc_ids = list(set(source['document_id'] for source in sources))

            # ✅ FIX: Check cache first for document metadata
            doc_lookup = {}
            uncached_ids = []

            for doc_id in doc_ids:
                if doc_id in self._doc_metadata_cache:
                    doc_lookup[doc_id] = self._doc_metadata_cache[doc_id]
                else:
                    uncached_ids.append(doc_id)

            # Fetch uncached documents from DB
            if uncached_ids:
                documents = db.query(Document).filter(Document.id.in_(uncached_ids)).all()
                for doc in documents:
                    # Cache document metadata
                    doc_data = {
                        'id': doc.id,
                        'original_filename': doc.original_filename,
                        'file_type': doc.file_type,
                        'upload_date': doc.upload_date.isoformat() if doc.upload_date else None
                    }
                    self._doc_metadata_cache[doc.id] = doc_data
                    doc_lookup[doc.id] = doc_data

            # Enhance sources
            enhanced_sources = []
            for source in sources:
                doc_id = source['document_id']
                enhanced_source = source.copy()

                if doc_id in doc_lookup:
                    doc_data = doc_lookup[doc_id]
                    enhanced_source.update({
                        'document_filename': doc_data['original_filename'],
                        'document_type': doc_data['file_type'],
                        'upload_date': doc_data['upload_date']
                    })

                # Format content preview (limit to 200 chars)
                content = enhanced_source.get('content', '')
                if len(content) > 200:
                    enhanced_source['content_preview'] = content[:200] + "..."
                else:
                    enhanced_source['content_preview'] = content

                enhanced_sources.append(enhanced_source)

            return enhanced_sources

        except Exception as e:
            logger.error(f"Error enhancing sources: {e}")
            return sources
        finally:
            # ✅ FIX: Only close if we created the session
            if should_close_db and db:
                db.close()
    
    def _save_conversation(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
        context_documents: List[int],
        sources: List[Dict[str, Any]],
        response_time_ms: int,
        db: Optional[Any] = None  # ✅ FIX: Accept DB session
    ):
        """
        ✅ FIXED: Save conversation to database with session reuse
        """
        # ✅ FIX: Manage DB session properly
        should_close_db = False
        if db is None:
            db = get_db_session()
            should_close_db = True

        try:
            conversation = Conversation(
                session_id=session_id,
                user_message=user_message,
                assistant_response=assistant_response,
                context_documents=context_documents,
                sources=sources,
                response_time_ms=response_time_ms
            )

            db.add(conversation)
            db.commit()

        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            db.rollback()
        finally:
            # ✅ FIX: Only close if we created the session
            if should_close_db and db:
                db.close()
    
    def _create_empty_response(self, message: str, session_id: Optional[str], 
                              response_time: float) -> Dict[str, Any]:
        """Create empty response when no relevant documents found"""
        return {
            'answer': message,
            'sources': [],
            'total_sources': 0,
            'response_time_ms': int(response_time * 1000),
            'session_id': session_id or 'anonymous'
        }
    
    def _create_error_response(self, error_message: str, session_id: Optional[str]) -> Dict[str, Any]:
        """Create error response"""
        return {
            'answer': f"I apologize, but I encountered an error: {error_message}",
            'sources': [],
            'total_sources': 0,
            'response_time_ms': 0,
            'session_id': session_id or 'anonymous',
            'error': error_message
        }
    
    def get_document_summary(self, document_id: int, max_chunks: int = 8,
                            use_legal: bool = None) -> str:
        """
        Generate summary for a specific document using llama.cpp
        Enhanced: Supports legal-specific summary for Vietnamese legal documents
        """
        try:
            if use_legal is None:
                use_legal = self.use_legal_prompts

            logger.info(f"Generating {'legal' if use_legal else 'standard'} summary for document {document_id}")

            # Get document chunks with highest scores using a generic query
            summary_query = "main points key information overview summary"
            query_embedding = self.embedding_service.encode_single_text(summary_query)

            search_results = self.vector_store.search_similar(
                query_embedding=query_embedding,
                top_k=max_chunks,
                document_ids=[document_id],
                min_score=0.0
            )

            if not search_results:
                return "Không có nội dung để tóm tắt." if use_legal else "No content available for summary."

            # Combine content from chunks
            content_parts = [result['content'] for result in search_results]
            combined_content = "\n\n".join(content_parts)

            # Limit content length for llama.cpp
            if len(combined_content) > 3000:
                combined_content = combined_content[:3000] + "..."

            # Generate summary using appropriate LangChain template
            if use_legal:
                summary_prompt_text = legal_summary_prompt.format(content=combined_content)
            else:
                summary_prompt_text = summary_prompt.format(content=combined_content)

            # Check context usage
            usage = self.llm_service.get_context_window_usage(summary_prompt_text, max_tokens=512)
            if usage['remaining_tokens'] < 0:
                # Truncate content further
                combined_content = combined_content[:2000] + "..."
                if use_legal:
                    summary_prompt_text = legal_summary_prompt.format(content=combined_content)
                else:
                    summary_prompt_text = summary_prompt.format(content=combined_content)

            summary = self.llm_service.generate_response(
                prompt=summary_prompt_text,
                max_tokens=512,
                temperature=0.1
            )

            return summary or ("Không thể tạo tóm tắt." if use_legal else "Unable to generate summary.")

        except Exception as e:
            logger.error(f"Error generating document summary: {e}")
            error_msg = f"Lỗi khi tạo tóm tắt: {str(e)}" if use_legal else f"Error generating summary: {str(e)}"
            return error_msg
    
    def get_conversation_history(
        self,
        session_id: str,
        limit: int = 10,
        db: Optional[Any] = None  # ✅ FIX: Accept DB session
    ) -> List[Dict[str, Any]]:
        """
        ✅ FIXED: Get conversation history with session reuse
        """
        # ✅ FIX: Manage DB session properly
        should_close_db = False
        if db is None:
            db = get_db_session()
            should_close_db = True

        try:
            conversations = db.query(Conversation).filter(
                Conversation.session_id == session_id
            ).order_by(Conversation.created_at.desc()).limit(limit).all()

            history = []
            for conv in reversed(conversations):  # Reverse to get chronological order
                history.append({
                    'user_message': conv.user_message,
                    'assistant_response': conv.assistant_response,
                    'sources': conv.sources or [],
                    'created_at': conv.created_at.isoformat(),
                    'response_time_ms': conv.response_time_ms
                })

            return history

        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
        finally:
            # ✅ FIX: Only close if we created the session
            if should_close_db and db:
                db.close()
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all RAG components including llama.cpp"""
        return {
            'embedding_service': self.embedding_service.get_model_info()['is_loaded'],
            'vector_store': self.vector_store.health_check(),
            'llm_service': self.llm_service.health_check()  # llama.cpp health check
        }
    
    def get_system_stats(self, db: Optional[Any] = None) -> Dict[str, Any]:
        """
        ✅ FIXED: Get comprehensive system statistics with session reuse
        """
        # ✅ FIX: Manage DB session properly
        should_close_db = False
        if db is None:
            db = get_db_session()
            should_close_db = True

        try:
            # Get Qdrant stats
            qdrant_stats = self.vector_store.get_collection_stats()

            # Get database stats
            from src.utils.database import Document, DocumentChunk, Conversation

            total_docs = db.query(Document).count()
            total_chunks = db.query(DocumentChunk).count()
            total_conversations = db.query(Conversation).count()

            db_stats = {
                'total_documents': total_docs,
                'total_chunks': total_chunks,
                'total_conversations': total_conversations
            }

            # Cache statistics
            cache_stats = {
                'query_embedding_cache_size': len(self._query_embedding_cache),
                'doc_metadata_cache_size': len(self._doc_metadata_cache)
            }

            return {
                'qdrant': qdrant_stats,
                'database': db_stats,
                'embedding_model': self.embedding_service.get_model_info(),
                'llm_model': self.llm_service.get_model_info(),  # llama.cpp model info
                'caches': cache_stats  # ✅ NEW: Cache stats
            }

        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {'error': str(e)}
        finally:
            # ✅ FIX: Only close if we created the session
            if should_close_db and db:
                db.close()

# Initialize RAG engine singleton
def initialize_rag_system() -> RAGEngine:
    """Initialize RAG system components with llama.cpp"""
    try:
        rag_engine = RAGEngine()
        logger.info("RAG system with llama.cpp initialized successfully")
        return rag_engine
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        raise

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

from src.rag.embedding_service import EmbeddingService
from src.rag.vector_store import QdrantVectorStore
from src.serving.llm_service import LlamaCppService  # UPDATED import
from src.serving.prompts.rag_prompts import build_rag_prompt, build_summary_prompt, RAG_SYSTEM_PROMPT
from src.utils.database import get_db_session, Document, Conversation
from src.utils.models import QueryRequest, QueryResponse, SourceInfo
from src.utils.exceptions import VectorStoreError, LLMServiceError

logger = logging.getLogger(__name__)

class RAGEngine:
    """Main Retrieval-Augmented Generation Engine with llama.cpp"""
    
    def __init__(self):
        try:
            logger.info("Initializing RAG Engine components...")
            
            self.embedding_service = EmbeddingService()
            self.vector_store = QdrantVectorStore()
            self.llm_service = LlamaCppService()  # UPDATED to use llama.cpp
            
            # RAG parameters
            self.default_top_k = 5
            self.max_context_length = 2500  # Reduced for llama.cpp efficiency
            self.min_similarity_score = 0.1
            
            logger.info("RAG Engine with llama.cpp initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing RAG Engine: {e}")
            raise
    
    def query(
        self,
        question: str,
        document_ids: Optional[List[int]] = None,
        top_k: int = 5,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process user query using RAG pipeline with llama.cpp
        """
        try:
            start_time = time.time()
            logger.info(f"Processing RAG query: {question[:100]}...")
            
            # Step 1: Generate query embedding
            logger.debug("Generating query embedding...")
            query_embedding = self.embedding_service.encode_single_text(question)
            
            # Step 2: Retrieve relevant chunks from Qdrant
            logger.debug(f"Searching Qdrant for similar chunks (top_k={top_k})...")
            search_results = self.vector_store.search_similar(
                query_embedding=query_embedding,
                top_k=top_k,
                document_ids=document_ids,
                min_score=self.min_similarity_score
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
            
            # Step 5: Enhance sources with document metadata
            enhanced_sources = self._enhance_sources_with_metadata(sources)
            
            response_time_ms = int((time.time() - start_time) * 1000)
            
            result = {
                'answer': answer,
                'sources': enhanced_sources,
                'total_sources': len(search_results),
                'response_time_ms': response_time_ms,
                'session_id': session_id or 'anonymous',
                'context_usage': context_usage
            }
            
            # Save conversation to database
            if session_id:
                self._save_conversation(
                    session_id, question, answer, 
                    document_ids or [], enhanced_sources, response_time_ms
                )
            
            logger.info(f"RAG query completed in {response_time_ms}ms using llama.cpp")
            return result
            
        except Exception as e:
            logger.error(f"Error processing RAG query: {e}")
            return self._create_error_response(str(e), session_id)
    
    def _build_full_prompt(self, question: str, context: str) -> str:
        """Build full prompt for context length estimation"""
        return build_rag_prompt(question, context, RAG_SYSTEM_PROMPT)
    
    def _build_context_and_sources(self, search_results: List[Dict[str, Any]]) -> tuple[str, List[Dict[str, Any]]]:
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
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using llama.cpp with context"""
        try:
            # Build RAG prompt
            prompt = build_rag_prompt(question, context, RAG_SYSTEM_PROMPT)
            
            # Check if we can fit this in context window
            usage = self.llm_service.get_context_window_usage(prompt, max_tokens=1024)
            
            if usage['remaining_tokens'] < 0:
                logger.warning("Prompt too long, truncating context...")
                # Recalculate with shorter context
                shorter_context = context[:len(context)//2]
                prompt = build_rag_prompt(question, shorter_context, RAG_SYSTEM_PROMPT)
            
            # Generate response
            answer = self.llm_service.generate_response(
                prompt=prompt,
                max_tokens=min(1024, usage.get('remaining_tokens', 1024) - 50),
                temperature=0.1
            )
            
            return answer or "I apologize, but I couldn't generate a response."
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"I encountered an error while generating the response: {str(e)}"
    
    def _enhance_sources_with_metadata(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add document metadata to sources"""
        if not sources:
            return []
        
        db = get_db_session()
        try:
            # Get unique document IDs
            doc_ids = list(set(source['document_id'] for source in sources))
            
            # Fetch document metadata
            documents = db.query(Document).filter(Document.id.in_(doc_ids)).all()
            doc_lookup = {doc.id: doc for doc in documents}
            
            # Enhance sources
            enhanced_sources = []
            for source in sources:
                doc_id = source['document_id']
                enhanced_source = source.copy()
                
                if doc_id in doc_lookup:
                    doc = doc_lookup[doc_id]
                    enhanced_source.update({
                        'document_filename': doc.original_filename,
                        'document_type': doc.file_type,
                        'upload_date': doc.upload_date.isoformat() if doc.upload_date else None
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
            db.close()
    
    def _save_conversation(
        self, 
        session_id: str, 
        user_message: str, 
        assistant_response: str,
        context_documents: List[int],
        sources: List[Dict[str, Any]],
        response_time_ms: int
    ):
        """Save conversation to database"""
        db = get_db_session()
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
    
    def get_document_summary(self, document_id: int, max_chunks: int = 8) -> str:
        """Generate summary for a specific document using llama.cpp"""
        try:
            logger.info(f"Generating summary for document {document_id}")
            
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
                return "No content available for summary."
            
            # Combine content from chunks
            content_parts = [result['content'] for result in search_results]
            combined_content = "\n\n".join(content_parts)
            
            # Limit content length for llama.cpp
            if len(combined_content) > 3000:
                combined_content = combined_content[:3000] + "..."
            
            # Generate summary
            summary_prompt = build_summary_prompt(combined_content)
            
            # Check context usage
            usage = self.llm_service.get_context_window_usage(summary_prompt, max_tokens=512)
            if usage['remaining_tokens'] < 0:
                # Truncate content further
                combined_content = combined_content[:2000] + "..."
                summary_prompt = build_summary_prompt(combined_content)
            
            summary = self.llm_service.generate_response(
                prompt=summary_prompt,
                max_tokens=512,
                temperature=0.1
            )
            
            return summary or "Unable to generate summary."
            
        except Exception as e:
            logger.error(f"Error generating document summary: {e}")
            return f"Error generating summary: {str(e)}"
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        db = get_db_session()
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
            db.close()
    
    def health_check(self) -> Dict[str, bool]:
        """Check health of all RAG components including llama.cpp"""
        return {
            'embedding_service': self.embedding_service.get_model_info()['is_loaded'],
            'vector_store': self.vector_store.health_check(),
            'llm_service': self.llm_service.health_check()  # llama.cpp health check
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        try:
            # Get Qdrant stats
            qdrant_stats = self.vector_store.get_collection_stats()
            
            # Get database stats
            db = get_db_session()
            try:
                from src.utils.database import Document, DocumentChunk, Conversation
                
                total_docs = db.query(Document).count()
                total_chunks = db.query(DocumentChunk).count()
                total_conversations = db.query(Conversation).count()
                
                db_stats = {
                    'total_documents': total_docs,
                    'total_chunks': total_chunks,
                    'total_conversations': total_conversations
                }
            finally:
                db.close()
            
            return {
                'qdrant': qdrant_stats,
                'database': db_stats,
                'embedding_model': self.embedding_service.get_model_info(),
                'llm_model': self.llm_service.get_model_info()  # llama.cpp model info
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {'error': str(e)}

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
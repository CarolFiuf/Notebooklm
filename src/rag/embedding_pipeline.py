import logging
from typing import Dict, Any
from datetime import datetime

from .embedding_service import EmbeddingService
from src.rag.vector_store import QdrantVectorStore
from src.utils.database import get_db_session, DocumentChunk 
from src.utils.exceptions import EmbeddingGenerationError, VectorStoreError

logger = logging.getLogger(__name__)

class EmbeddingPipeline:
    """Pipeline for processing document embeddings"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = QdrantVectorStore()
    
    def process_document_embeddings(self, document_id: int) -> Dict[str, Any]:
        """
        Generate and store embeddings for a document's chunks
        
        Args:
            document_id: ID of the document to process
            
        Returns:
            Dictionary with processing results
        """
        try:
            logger.info(f"Processing embeddings for document {document_id}")
            start_time = datetime.now()
            
            # Get document chunks from database
            chunks_data, texts = self._get_document_chunks(document_id)
            
            if not chunks_data:
                raise ValueError(f"No chunks found for document {document_id}")
            
            logger.info(f"Generating embeddings for {len(texts)} chunks")
            
            # Generate embeddings
            embeddings = self.embedding_service.encode_texts(texts)
            
            # Insert into vector store
            embedding_ids = self.vector_store.insert_embeddings(
                document_id, chunks_data, embeddings
            )
            
            # Update database with embedding IDs
            self._update_chunk_embedding_ids(document_id, embedding_ids)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'document_id': document_id,
                'total_chunks': len(chunks_data),
                'embedding_ids': embedding_ids,
                'embedding_dimension': embeddings.shape[1],
                'processing_time_seconds': processing_time,
                'success': True
            }
            
            logger.info(f"Embedding processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing embeddings for document {document_id}: {e}")
            return {
                'document_id': document_id,
                'success': False,
                'error': str(e)
            }
    
    def _get_document_chunks(self, document_id: int) -> tuple[list, list]:
        """Get document chunks and texts from database"""
        db = get_db_session()
        try:
            chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).order_by(DocumentChunk.chunk_index).all()
            
            chunks_data = []
            texts = []
            
            for chunk in chunks:
                chunks_data.append({
                    'chunk_index': chunk.chunk_index,
                    'content': chunk.content,
                    'metadata': chunk.chunk_metadata or {}
                })
                texts.append(chunk.content)
            
            return chunks_data, texts
            
        except Exception as e:
            logger.error(f"Error getting document chunks: {e}")
            raise
        finally:
            db.close()
    
    def _update_chunk_embedding_ids(self, document_id: int, embedding_ids: list):
        """Update database chunks with embedding IDs"""
        db = get_db_session()
        try:
            chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).order_by(DocumentChunk.chunk_index).all()
            
            for chunk, embedding_id in zip(chunks, embedding_ids):
                chunk.embedding_id = embedding_id
            
            db.commit()
            logger.info(f"Updated {len(embedding_ids)} chunks with embedding IDs")
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error updating chunk embedding IDs: {e}")
            raise
        finally:
            db.close()
    
    def delete_document_embeddings(self, document_id: int) -> Dict[str, Any]:
        """Delete all embeddings for a document"""
        try:
            # Delete from vector store
            deleted_count = self.vector_store.delete_by_document_id(document_id)
            
            # Clear embedding IDs from database
            db = get_db_session()
            try:
                chunks = db.query(DocumentChunk).filter(
                    DocumentChunk.document_id == document_id
                ).all()
                
                for chunk in chunks:
                    chunk.embedding_id = None
                
                db.commit()
                db_updated_count = len(chunks)
                
            except Exception as e:
                db.rollback()
                raise e
            finally:
                db.close()
            
            return {
                'document_id': document_id,
                'deleted_from_vector_store': deleted_count,
                'updated_in_database': db_updated_count,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Error deleting embeddings for document {document_id}: {e}")
            return {
                'document_id': document_id,
                'success': False,
                'error': str(e)
            }

# ===================================
# Convenience function for backward compatibility
# ===================================
def process_document_embeddings(document_id: int) -> Dict[str, Any]:
    """
    Generate and store embeddings for a document's chunks using Qdrant
    
    Args:
        document_id: ID of the document to process
        
    Returns:
        Dictionary with processing results
    """
    pipeline = EmbeddingPipeline()
    return pipeline.process_document_embeddings(document_id)
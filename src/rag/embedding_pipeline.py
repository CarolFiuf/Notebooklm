import logging
from typing import Dict, Any, Tuple, List
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
        # Use config-only dimension in VectorStore
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
            
            # Get document chunks from database (filter out empty content)
            chunks_data, texts = self._get_document_chunks(document_id)
            
            if not chunks_data:
                raise ValueError(f"No valid chunks to embed for document {document_id}")
            
            logger.info(f"Generating embeddings for {len(texts)} chunks")
            
            # Generate embeddings
            embeddings = self.embedding_service.encode_texts(texts)
            
            # Insert into vector store
            embedding_ids = self.vector_store.insert_embeddings(
                document_id, chunks_data, embeddings
            )
            
            # Update database with embedding IDs (align by chunk_index of filtered chunks)
            self._update_chunk_embedding_ids(document_id, embedding_ids, chunks_data)
            
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
    
    def _get_document_chunks(self, document_id: int) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Get document chunks and texts from database, filtering out empty content"""
        db = get_db_session()
        try:
            chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).order_by(DocumentChunk.chunk_index).all()
            
            chunks_data = []
            texts = []
            skipped = 0
            
            for chunk in chunks:
                content = (chunk.content or "").strip()
                if not content:
                    skipped += 1
                    continue
                chunks_data.append({
                    'chunk_index': chunk.chunk_index,
                    'content': content,
                    'metadata': chunk.chunk_metadata or {}
                })
                texts.append(content)
            
            if skipped:
                logger.warning(
                    f"Filtered out {skipped} empty chunk(s) for document {document_id}"
                )
            
            return chunks_data, texts
            
        except Exception as e:
            logger.error(f"Error getting document chunks: {e}")
            raise
        finally:
            db.close()
    
    def _update_chunk_embedding_ids(self, document_id: int, embedding_ids: list, used_chunks: list):
        """Update database chunks with embedding IDs for the filtered set only"""
        db = get_db_session()
        try:
            # Build mapping: chunk_index -> embedding_id
            used_indices = [c.get('chunk_index') for c in used_chunks]
            index_to_embedding = {idx: emb for idx, emb in zip(used_indices, embedding_ids)}

            # Update only matched chunk indices
            chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id,
                DocumentChunk.chunk_index.in_(used_indices)
            ).all()

            updated = 0
            for chunk in chunks:
                emb_id = index_to_embedding.get(chunk.chunk_index)
                if emb_id is not None:
                    chunk.embedding_id = emb_id
                    updated += 1
            
            db.commit()
            logger.info(f"Updated {updated} chunks with embedding IDs (filtered)")
            
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

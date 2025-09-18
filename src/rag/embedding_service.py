import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime

from config.config import settings
from src.utils.exceptions import EmbeddingGenerationError
from src.rag.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings using BGE-M3 model"""
    
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.embedding_dim = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            start_time = datetime.now()
            
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True
            )
            
            # Get actual embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            load_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Embedding model loaded in {load_time:.2f}s. Dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise EmbeddingGenerationError(f"Failed to load model: {e}")
    
    def encode_texts(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        try:
            if not texts:
                return np.array([])
            
            # Filter out empty texts
            non_empty_texts = [text.strip() for text in texts if text and text.strip()]
            
            if not non_empty_texts:
                logger.warning("All texts are empty")
                return np.array([])
            
            logger.info(f"Encoding {len(non_empty_texts)} texts with batch_size={batch_size}")
            start_time = datetime.now()
            
            embeddings = self.model.encode(
                non_empty_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
            
            encoding_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Encoding completed in {encoding_time:.2f}s")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise EmbeddingGenerationError(f"Embedding generation failed: {e}")
    
    def encode_single_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """Generate embedding for a single text"""
        if not text or not text.strip():
            raise EmbeddingGenerationError("Empty text provided")
        
        embeddings = self.encode_texts([text.strip()], batch_size=1, 
                                     show_progress=False, normalize=normalize)
        return embeddings[0]
    
    def get_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Ensure embeddings are normalized
            embedding1 = embedding1 / np.linalg.norm(embedding1)
            embedding2 = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(embedding1, embedding2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'embedding_dimension': self.embedding_dim,
            'max_seq_length': getattr(self.model, 'max_seq_length', 'unknown'),
            'is_loaded': self.model is not None
        }
        
def process_document_embeddings(document_id: int) -> Dict[str, Any]:
    """
    Generate and store embeddings for a document's chunks using Qdrant
    
    Args:
        document_id: ID of the document to process
        
    Returns:
        Dictionary with processing results
    """
    try:
        logger.info(f"Processing embeddings for document {document_id}")
        start_time = datetime.now()
        
        # Initialize services
        from src.rag.embedding_service import EmbeddingService
        embedding_service = EmbeddingService()
        vector_store = QdrantVectorStore()  # Using Qdrant now
        
        # Get document chunks from database
        db = get_db_session()
        try:
            chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).order_by(DocumentChunk.chunk_index).all()
            
            if not chunks:
                raise ValueError(f"No chunks found for document {document_id}")
            
            # Prepare chunk data and texts
            chunks_data = []
            texts = []
            
            for chunk in chunks:
                chunks_data.append({
                    'chunk_index': chunk.chunk_index,
                    'content': chunk.content,
                    'metadata': chunk.chunk_metadata or {}
                })
                texts.append(chunk.content)
            
            logger.info(f"Generating embeddings for {len(texts)} chunks")
            
            # Generate embeddings
            embeddings = embedding_service.encode_texts(texts)
            
            # Insert into Qdrant vector store
            embedding_ids = vector_store.insert_embeddings(
                document_id, chunks_data, embeddings
            )
            
            # Update database with embedding IDs
            for chunk, embedding_id in zip(chunks, embedding_ids):
                chunk.embedding_id = embedding_id
            
            db.commit()
            
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
            db.rollback()
            raise e
        finally:
            db.close()
            
    except Exception as e:
        logger.error(f"Error processing embeddings for document {document_id}: {e}")
        return {
            'document_id': document_id,
            'success': False,
            'error': str(e)
        }
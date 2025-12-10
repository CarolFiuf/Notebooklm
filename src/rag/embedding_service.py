import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from datetime import datetime
import threading
from collections import deque
import time
import os
import re
import unicodedata

# LangChain compatibility
from langchain_core.embeddings import Embeddings

# from llama_cpp import Llama

from config.settings import settings
from src.utils.exceptions import EmbeddingGenerationError

logger = logging.getLogger(__name__)


def normalize_query(query: str) -> str:
    """
    ✅ NEW: Normalize query for better matching

    Especially important for Vietnamese legal queries with:
    - Case variations (điều vs Điều)
    - Extra whitespace
    - Vietnamese diacritics normalization

    Args:
        query: Raw query string

    Returns:
        Normalized query string
    """
    if not query:
        return query

    # Normalize Unicode characters (NFC form for Vietnamese)
    query = unicodedata.normalize('NFC', query)

    # Lowercase for consistent matching
    query = query.lower()

    # Normalize whitespace (multiple spaces → single space)
    query = re.sub(r'\s+', ' ', query)

    # Remove leading/trailing whitespace
    query = query.strip()

    # Remove trailing punctuation (but keep internal punctuation)
    query = query.rstrip('.,;!?:')

    return query

class EmbeddingService:
    """
    FIXED: Service for generating embeddings with rate limiting and concurrency control

    Features:
    - Thread-safe embedding generation
    - Semaphore-based concurrency control (max 3 concurrent requests)
    - Request queue monitoring
    """

    def __init__(self, max_concurrent_requests: int = None):
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.embedding_dim = None
        cpu_count = os.cpu_count()  # 🔧 FIX: Call the method
        max_concurrent_requests = max(cpu_count//2, 4)
        self.max_concurrent = max_concurrent_requests

        # 🔧 FIX: Use self.max_concurrent_requests instead of parameter
        self._semaphore = threading.Semaphore(self.max_concurrent)
        self._lock = threading.Lock()

        # 🔧 FIX: Request tracking for monitoring
        self._active_requests = 0
        self._total_requests = 0
        self._queue_wait_times = deque(maxlen=100)  # Track last 100 wait times

        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
            start_time = datetime.now()
            
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
                cache_folder=None,  # Use default cache
                use_auth_token=None,  # No auth token needed
                revision=None,  # Use main branch
                local_files_only=False,  # Allow downloads if needed
                model_kwargs={
                    'torch_dtype': torch.float32 if self.device == 'cpu' else torch.float16,
                    'attn_implementation': 'eager',  # Use eager attention
                    'low_cpu_mem_usage': False,  # Prevent meta device initialization
                }
            )
            
            # Get actual embedding dimension
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            load_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Embedding model loaded in {load_time:.2f}s. Dimension: {self.embedding_dim}")

            # Validate against config-only dimension to prevent drift
            try:
                expected_dim = int(settings.EMBEDDING_DIMENSION)
            except Exception:
                expected_dim = None
            if expected_dim and self.embedding_dim != expected_dim:
                msg = (
                    f"Embedding dimension mismatch: model={self.embedding_dim}, "
                    f"config={expected_dim}. Please update config or choose a matching model."
                )
                logger.error(msg)
                raise EmbeddingGenerationError(msg)
            
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
        """
        🔧 FIXED: Generate embeddings with concurrency control

        Uses semaphore to limit concurrent requests and prevent GPU/CPU overload.
        """
        wait_start = time.time()

        # 🔧 FIX: Acquire semaphore (blocks if max concurrent requests reached)
        acquired = self._semaphore.acquire(blocking=True, timeout=300)  # 5 min timeout

        if not acquired:
            raise EmbeddingGenerationError(
                f"Failed to acquire embedding slot within 5 minutes. "
                f"Active requests: {self._active_requests}"
            )

        wait_time = time.time() - wait_start
        request_tracked = False  # Track if we incremented _active_requests

        try:
            # Track request - increment counter
            with self._lock:
                self._active_requests += 1
                self._total_requests += 1
                self._queue_wait_times.append(wait_time)
                request_tracked = True  # Mark that we successfully tracked the request

            if wait_time > 1.0:
                logger.warning(f"Waited {wait_time:.2f}s for embedding slot (active: {self._active_requests})")

            if not texts:
                return np.array([])

            # Filter out empty texts
            non_empty_texts = [text.strip() for text in texts if text and text.strip()]

            if not non_empty_texts:
                logger.warning("All texts are empty")
                return np.array([])

            logger.info(
                f"Encoding {len(non_empty_texts)} texts "
                f"(active: {self._active_requests}/{self.max_concurrent}, "
                f"total: {self._total_requests})"
            )
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
        finally:
            # 🔧 CRITICAL FIX: Release semaphore FIRST, then decrement counter
            # This prevents deadlock if exception occurs during lock acquisition
            self._semaphore.release()

            # Only decrement if we successfully incremented
            if request_tracked:
                with self._lock:
                    self._active_requests -= 1
    
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
        """
        🔧 FIXED: Get model information with concurrency stats
        """
        avg_wait_time = (
            sum(self._queue_wait_times) / len(self._queue_wait_times)
            if self._queue_wait_times else 0.0
        )

        return {
            'model_name': self.model_name,
            'device': self.device,
            'embedding_dimension': self.embedding_dim,
            'max_seq_length': getattr(self.model, 'max_seq_length', 'unknown'),
            'is_loaded': self.model is not None,
            # 🔧 NEW: Concurrency stats
            'concurrency': {
                'max_concurrent': self.max_concurrent,
                'active_requests': self._active_requests,
                'total_requests': self._total_requests,
                'avg_wait_time_s': round(avg_wait_time, 3)
            }
        }


class LangChainEmbeddingAdapter(Embeddings):
    """
    LangChain-compatible adapter for EmbeddingService

    Wraps EmbeddingService to provide LangChain's Embeddings interface
    """

    def __init__(self, embedding_service: EmbeddingService):
        """
        Initialize adapter with existing EmbeddingService

        Args:
            embedding_service: EmbeddingService instance to wrap
        """
        self.embedding_service = embedding_service
        logger.info("✅ LangChain embedding adapter initialized")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents (LangChain interface)

        Args:
            texts: List of text documents to embed

        Returns:
            List of embedding vectors as lists of floats
        """
        try:
            # Use EmbeddingService to encode texts
            embeddings = self.embedding_service.encode_texts(texts)

            # Convert numpy arrays to lists of floats
            return [embedding.tolist() for embedding in embeddings]

        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query (LangChain interface)

        Args:
            text: Query text to embed

        Returns:
            Embedding vector as list of floats
        """
        try:
            # Use EmbeddingService to encode single text
            embedding = self.embedding_service.encode_single_text(text)

            # Convert numpy array to list of floats
            return embedding.tolist()

        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise


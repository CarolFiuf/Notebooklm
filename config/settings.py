"""
Unified Settings for Vietnamese Legal RAG System
Consolidated from config.py + vietnam_legal_config.py (removed unused configs)
"""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    DOCUMENTS_DIR: Path = DATA_DIR / "documents"
    EMBEDDINGS_DIR: Path = DATA_DIR / "embeddings"
    MODELS_DIR: Path = DATA_DIR / "models"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"

    # Database Configuration
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "notebooklm"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "123456"

    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: str = ""

    # Qdrant Configuration
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "document_embeddings"
    QDRANT_USE_HTTPS: bool = False

    # LLM Configuration - llama.cpp
    LLM_MODEL_PATH: str = ""  
    # LLM_MODEL_NAME: str = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    LLM_MODEL_NAME: str = "Qwen3-8B-Q4_K_M.gguf"
    # LLM_MODEL_URL: str = "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.1
    LLM_TOP_P: float = 0.9
    LLM_CONTEXT_LENGTH: int = 4096  # Increased to 4096 to handle larger context windows

    # llama.cpp specific settings
    LLAMACPP_N_GPU_LAYERS: int = 0
    LLAMACPP_N_BATCH: int = 64  # Increased for better performance
    LLAMACPP_N_THREADS: int = 4
    LLAMACPP_VERBOSE: bool = False

    # Embedding Configuration
    EMBEDDING_MODEL_NAME: str = "Qwen/Qwen3-Embedding-0.6B"
    EMBEDDING_DIMENSION: int = 1024

    # Retrieval Configuration
    SEMANTIC_THRESHOLD: float = 0.3  # No threshold - get all results for debugging
    TOP_K_RESULTS: int = 10  # Increase to see more candidates

    # Reranker Configuration
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"  # Multilingual, good for Vietnamese
    ENABLE_RERANKING: bool = True

    # Hybrid Search Weights
    SEMANTIC_WEIGHT: float = 0.7  # Weight for semantic/dense search (lower for exact queries)
    BM25_WEIGHT: float = 0.3  # Weight for BM25/keyword search (higher for exact match)

    # Document Processing & Chunking
    CHUNK_SIZE: int = 20000  # Increased from 1200 to reduce broken chunks
    CHUNK_OVERLAP: int = 200  # Increased from 150 for better context

    # File Processing
    MAX_FILE_SIZE_MB: int = 100
    SUPPORTED_FORMATS: List[str] = [".pdf", ".txt", ".md", ".docx", ".doc"]

    # Streamlit Configuration
    STREAMLIT_SERVER_PORT: int = 8501
    STREAMLIT_SERVER_ADDRESS: str = "0.0.0.0"

    # ========================================================================
    # Evaluation Configuration (RAGAS)
    # ========================================================================

    # LLM for RAGAS Evaluation
    EVAL_LLM_MODEL: str = "gpt-oss-120b"  # FPT Cloud model
    EVAL_LLM_API_KEY: str = os.getenv("EVAL_LLM_API_KEY", os.getenv("FPT_API_KEY", os.getenv("OPENAI_API_KEY", "")))
    EVAL_LLM_BASE_URL: str = os.getenv("EVAL_LLM_BASE_URL", "https://mkp-api.fptcloud.com/v1")
    EVAL_LLM_MAX_RETRIES: int = 5
    EVAL_LLM_TIMEOUT: int = 180  # seconds

    # Embeddings for RAGAS Evaluation (can be different from LLM)
    EVAL_EMBEDDING_MODEL: str = "Vietnamese_Embedding"
    EVAL_EMBEDDING_API_KEY: str = os.getenv("EVAL_EMBEDDING_API_KEY", os.getenv("EVAL_LLM_API_KEY", os.getenv("FPT_API_KEY", os.getenv("OPENAI_API_KEY", ""))))
    EVAL_EMBEDDING_BASE_URL: str = os.getenv("EVAL_EMBEDDING_BASE_URL", os.getenv("EVAL_LLM_BASE_URL", "https://mkp-api.fptcloud.com/v1"))

    # Evaluation Performance & Rate Limiting
    EVAL_MAX_CONCURRENT: int = 2  # Number of concurrent test cases (lower = safer)
    EVAL_QUERY_TOP_K: int = 10  # Number of chunks to retrieve per query

    # Results Directory
    EVAL_RESULTS_DIR: Path = PROJECT_ROOT / "evaluation" / "results"

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def qdrant_url(self) -> str:
        protocol = "https" if self.QDRANT_USE_HTTPS else "http"
        return f"{protocol}://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

    @property
    def redis_url(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}"
        else:
            return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}"

    @property
    def model_path(self) -> Path:
        if self.LLM_MODEL_PATH:
            return Path(self.LLM_MODEL_PATH)
        else:
            return self.MODELS_DIR / self.LLM_MODEL_NAME

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


# Create global settings instance
settings = Settings()

# Ensure directories exist
for directory in [settings.DATA_DIR, settings.DOCUMENTS_DIR,
                  settings.EMBEDDINGS_DIR, settings.MODELS_DIR, settings.LOGS_DIR,
                  settings.EVAL_RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# Backward compatibility: Export LEGAL_CHUNKING_CONFIG for document_processor.py
LEGAL_CHUNKING_CONFIG = {
    "chunk_size": settings.CHUNK_SIZE,
    "chunk_overlap": settings.CHUNK_OVERLAP,
}

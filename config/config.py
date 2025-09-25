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
    
    # Redis Configuration (ADDED - was missing)
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
    
    # LLM Configuration - UPDATED for llama.cpp
    LLM_MODEL_PATH: str = ""  # Path to GGUF model file
    # LLM_MODEL_NAME: str = "Qwen2.5-7B-Instruct-Q6_K_L.gguf"  # GGUF model filename
    LLM_MODEL_NAME: str = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"  # GGUF model filename
    # LLM_MODEL_URL: str = "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q6_K_L.gguf"
    LLM_MODEL_URL: str = "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    LLM_MAX_TOKENS: int = 2048
    LLM_TEMPERATURE: float = 0.1
    LLM_TOP_P: float = 0.9
    LLM_CONTEXT_LENGTH: int = 4096
    
    # llama.cpp specific settings - safer defaults to prevent GGML errors
    LLAMACPP_N_GPU_LAYERS: int = 0  # Number of layers to offload to GPU (0 = CPU only)
    LLAMACPP_N_BATCH: int = 64 # Reduced batch size to prevent memory issues
    LLAMACPP_N_THREADS: int = 4  # Reduced CPU threads for stability
    LLAMACPP_VERBOSE: bool = False
    
    # Embedding Configuration
    # EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
    # EMBEDDING_MODEL_NAME: str = "Qwen/Qwen3-Embedding-0.6B"
    EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-large-instruct"
    EMBEDDING_DIMENSION: int = 1024
    CHUNK_SIZE: int = 800   
    CHUNK_OVERLAP: int = 100
    
    # File Processing
    MAX_FILE_SIZE_MB: int = 100
    SUPPORTED_FORMATS: List[str] = [".pdf", ".txt", ".md"]
    
    # Streamlit Configuration
    STREAMLIT_SERVER_PORT: int = 8501
    STREAMLIT_SERVER_ADDRESS: str = "0.0.0.0"
    
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def qdrant_url(self) -> str:
        protocol = "https" if self.QDRANT_USE_HTTPS else "http"
        return f"{protocol}://{self.QDRANT_HOST}:{self.QDRANT_PORT}"
    
    @property
    def redis_url(self) -> str:
        """Get Redis connection URL"""
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}"
        else:
            return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}"
    
    @property
    def model_path(self) -> Path:
        """Get full path to model file"""
        if self.LLM_MODEL_PATH:
            return Path(self.LLM_MODEL_PATH)
        else:
            return self.MODELS_DIR / self.LLM_MODEL_NAME
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        # FIXED: Allow extra fields to prevent validation errors
        extra = "ignore"  # This will ignore extra fields instead of raising errors

# Create global settings instance
settings = Settings()

# Ensure directories exist
for directory in [settings.DATA_DIR, settings.DOCUMENTS_DIR, 
                  settings.EMBEDDINGS_DIR, settings.MODELS_DIR, settings.LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
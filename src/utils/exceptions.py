class NotebookLMError(Exception):
    """Base exception class for NotebookLM application"""
    pass

class ConfigurationError(NotebookLMError):
    """Configuration related errors"""
    pass

class DocumentProcessingError(NotebookLMError):
    """Document processing related errors"""
    pass

class EmbeddingGenerationError(NotebookLMError):
    """Embedding generation related errors"""
    pass

class VectorStoreError(NotebookLMError):
    """Vector database related errors"""
    pass

class QdrantVectorStoreError(VectorStoreError):
    """Qdrant vector store specific errors"""
    pass

class LLMServiceError(NotebookLMError):
    """LLM service related errors"""
    pass

class DatabaseError(NotebookLMError):
    """Database operation related errors"""
    pass

class FileProcessingError(DocumentProcessingError):
    """File processing specific errors"""
    pass

class UnsupportedFileTypeError(FileProcessingError):
    """Unsupported file type error"""
    pass
from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
import json

class DocumentBase(BaseModel):
    """Base document model"""
    original_filename: str = Field(..., min_length=1, max_length=255)
    file_type: str = Field(..., pattern=r'^\.(pdf|txt|md)$')
    
    @validator('original_filename')
    def validate_filename(cls, v):
        if not v or v.isspace():
            raise ValueError('Filename cannot be empty')
        return v.strip()

class DocumentCreate(DocumentBase):
    """Document creation model"""
    filename: str = Field(..., max_length=255)
    file_path: str = Field(..., min_length=1)
    file_size: int = Field(..., gt=0)
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('file_size')
    def validate_file_size(cls, v):
        max_size = 100 * 1024 * 1024  # 100MB in bytes
        if v > max_size:
            raise ValueError(f'File size must be less than {max_size} bytes')
        return v

class DocumentResponse(BaseModel):
    """Document response model"""
    id: int
    filename: str
    original_filename: str
    file_size: int
    file_type: str
    upload_date: datetime
    processing_status: str
    total_chunks: int
    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True
        
    @property
    def file_size_mb(self) -> float:
        """File size in MB"""
        return round(self.file_size / (1024 * 1024), 2)

class DocumentUpdate(BaseModel):
    """Document update model"""
    processing_status: Optional[str] = None
    total_chunks: Optional[int] = None
    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class ChunkBase(BaseModel):
    """Base chunk model"""
    chunk_index: int = Field(..., ge=0)
    content: str = Field(..., min_length=1)
    chunk_metadata: Optional[Dict[str, Any]] = None

class ChunkCreate(ChunkBase):
    """Chunk creation model"""
    document_id: int = Field(..., gt=0)
    embedding_id: Optional[str] = None

class ChunkResponse(ChunkBase):
    """Chunk response model"""
    id: int
    document_id: int
    embedding_id: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class ConversationBase(BaseModel):
    """Base conversation model"""
    user_message: str = Field(..., min_length=1, max_length=10000)
    assistant_response: str = Field(..., min_length=1, max_length=20000)
    
    @validator('user_message', 'assistant_response')
    def validate_message(cls, v):
        if not v or v.isspace():
            raise ValueError('Message cannot be empty')
        return v.strip()

class ConversationCreate(ConversationBase):
    """Conversation creation model"""
    session_id: str = Field(..., min_length=1, max_length=100)
    context_documents: Optional[List[int]] = None
    sources: Optional[Dict[str, Any]] = None
    response_time_ms: Optional[int] = None

class ConversationResponse(ConversationBase):
    """Conversation response model"""
    id: int
    session_id: str
    context_documents: Optional[List[int]] = None
    sources: Optional[Dict[str, Any]] = None
    response_time_ms: Optional[int] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class QueryRequest(BaseModel):
    """Query request model"""
    question: str = Field(..., min_length=1, max_length=1000)
    document_ids: Optional[List[int]] = None
    top_k: int = Field(default=5, ge=1, le=20)
    session_id: Optional[str] = None
    
    @validator('question')
    def validate_question(cls, v):
        return v.strip()

class QueryResponse(BaseModel):
    """Query response model"""
    answer: str
    sources: List[Dict[str, Any]]
    total_sources: int
    response_time_ms: int
    session_id: str

class SourceInfo(BaseModel):
    """Source information model"""
    document_id: int
    document_filename: str
    chunk_index: int
    content: str
    score: float = Field(..., ge=0, le=1)
    metadata: Optional[Dict[str, Any]] = None

class ProcessingStatus(BaseModel):
    """Processing status model"""
    document_id: int
    status: str
    progress: float = Field(..., ge=0, le=100)
    message: str
    created_at: datetime

class SystemStats(BaseModel):
    """System statistics model"""
    total_documents: int
    total_chunks: int
    total_conversations: int
    storage_used_mb: float
    active_sessions: int

class HealthCheck(BaseModel):
    """Health check model"""
    status: str
    database: bool
    vector_store: bool
    llm_service: bool
    timestamp: datetime

# Utility functions for model conversion
def document_db_to_response(document: 'Document') -> DocumentResponse:
    """Convert database document to response model"""
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        original_filename=document.original_filename,
        file_size=document.file_size,
        file_type=document.file_type,
        upload_date=document.upload_date,
        processing_status=document.processing_status,
        total_chunks=document.total_chunks,
        summary=document.summary,
        metadata=document.metadata,
        created_at=document.created_at
    )

def chunk_db_to_response(chunk: 'DocumentChunk') -> ChunkResponse:
    """Convert database chunk to response model"""
    return ChunkResponse(
        id=chunk.id,
        document_id=chunk.document_id,
        chunk_index=chunk.chunk_index,
        content=chunk.content,
        chunk_metadata=chunk.chunk_metadata,
        embedding_id=chunk.embedding_id,
        created_at=chunk.created_at
    )

def conversation_db_to_response(conversation: 'Conversation') -> ConversationResponse:
    """Convert database conversation to response model"""
    return ConversationResponse(
        id=conversation.id,
        session_id=conversation.session_id,
        user_message=conversation.user_message,
        assistant_response=conversation.assistant_response,
        context_documents=conversation.context_documents,
        sources=conversation.sources,
        response_time_ms=conversation.response_time_ms,
        created_at=conversation.created_at
    )
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, JSON, ARRAY, Float, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from datetime import datetime
from typing import Generator
import logging
import os

from config.config import settings

logger = logging.getLogger(__name__)

cpu_count = os.cpu_count()
pool_size = cpu_count*2
max_overflow = pool_size*2
# Database setup
engine = create_engine(
    settings.database_url,
    pool_size=pool_size,
    max_overflow=max_overflow,
    pool_pre_ping=True,
    echo=settings.DEBUG
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy Models
class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False, unique=True, index=True)  
    original_filename = Column(String(255), nullable=False, index=True)      
    file_path = Column(Text, nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String(50), nullable=False, index=True)              
    upload_date = Column(DateTime, default=datetime.utcnow, index=True)     
    processing_status = Column(String(50), default="pending", index=True)   
    total_chunks = Column(Integer, default=0)
    summary = Column(Text, nullable=True)
    document_metadata = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)      
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # ✅ ADDED: Relationship to chunks
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.original_filename}')>"

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey('documents.id', ondelete='CASCADE'), 
                        nullable=False, index=True)  # ✅ ADDED ForeignKey
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    chunk_metadata = Column(JSON, nullable=True)
    embedding_id = Column(String(100), nullable=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # ✅ ADDED: Relationship to document
    document = relationship("Document", back_populates="chunks")
    
    # ✅ ADDED: Composite index for better query performance
    __table_args__ = (
        Index('idx_doc_chunk', 'document_id', 'chunk_index'),
    )
    
    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, doc_id={self.document_id}, chunk={self.chunk_index})>"

class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), nullable=False, index=True)
    user_message = Column(Text, nullable=False)
    assistant_response = Column(Text, nullable=False)
    context_documents = Column(ARRAY(Integer), nullable=True)
    sources = Column(JSON, nullable=True)
    response_time_ms = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # ✅ ADDED: Composite index for session queries
    __table_args__ = (
        Index('idx_session_created', 'session_id', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Conversation(id={self.id}, session='{self.session_id}')>"

# ✅ ADDED: Import Index for composite indexes

# Database connection helpers
def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()

def get_db_session() -> Session:
    """Get database session directly"""
    return SessionLocal()

def init_database():
    """Initialize database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise

def check_database_connection() -> bool:
    """Check database connection"""
    try:
        db = get_db_session()
        db.execute("SELECT 1")
        db.close()
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False

# ✅ ADDED: Database maintenance functions
def get_database_stats() -> dict:
    """Get database statistics"""
    try:
        db = get_db_session()
        stats = {
            'total_documents': db.query(Document).count(),
            'total_chunks': db.query(DocumentChunk).count(),
            'total_conversations': db.query(Conversation).count(),
            'completed_documents': db.query(Document).filter(
                Document.processing_status == 'completed'
            ).count()
        }
        db.close()
        return stats
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        return {}

def cleanup_orphaned_chunks() -> int:
    """Clean up chunks without documents"""
    try:
        db = get_db_session()
        
        # Find orphaned chunks
        orphaned = db.query(DocumentChunk).filter(
            ~DocumentChunk.document_id.in_(
                db.query(Document.id)
            )
        )
        
        count = orphaned.count()
        if count > 0:
            orphaned.delete(synchronize_session=False)
            db.commit()
            logger.info(f"Cleaned up {count} orphaned chunks")
        
        db.close()
        return count
        
    except Exception as e:
        logger.error(f"Error cleaning orphaned chunks: {e}")
        return 0
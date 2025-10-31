import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy.exc import IntegrityError

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangChainDocument

from config.config import settings
from src.utils.exceptions import DocumentProcessingError, UnsupportedFileTypeError
from src.utils.database import get_db_session, Document, DocumentChunk
from src.utils.models import DocumentCreate
from src.utils.file_utils import generate_file_hash, generate_content_hash, get_file_info
from .extractors import PDFExtractor, TextExtractor, WordExtractor

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Main document processing pipeline"""
    
    def __init__(self):
        # Initialize extractors
        self.pdf_extractor = PDFExtractor()
        self.text_extractor = TextExtractor()
        self.word_extractor = WordExtractor()

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            length_function=len
        )

        # Supported file types mapping
        self.extractors = {
            '.pdf': self.pdf_extractor,
            '.txt': self.text_extractor,
            '.md': self.text_extractor,
            '.docx': self.word_extractor,
            '.doc': self.word_extractor
        }
    
    def process_document(self, file_path: str, original_filename: str) -> Dict[str, Any]:
        """
        Process document through the complete pipeline
        
        Args:
            file_path: Path to the uploaded file
            original_filename: Original filename from user
            
        Returns:
            Dict containing processing results
        """
        try:
            start_time = datetime.now()
            logger.info(f"Starting document processing: {original_filename}")
            
            # Step 1: Validate file
            file_info = self._validate_file(file_path, original_filename)
            
            # Step 2: Check if document already exists
            existing_doc = self._check_existing_document(file_path, original_filename)
            if existing_doc:
                logger.info(f"Document already exists with ID: {existing_doc['id']}")
                return existing_doc
            
            # Step 3: Extract text
            text_content, extraction_metadata = self._extract_text(
                file_path, file_info['file_type']
            )
            
            # Step 4: Create chunks
            chunks = self._create_chunks(text_content, extraction_metadata)
            
            # Step 5: Generate unique filename with timestamp
            file_hash = generate_file_hash(file_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_filename = f"{file_hash}_{timestamp}_{original_filename}"
            
            # Step 6: Prepare result
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'filename': unique_filename,
                'original_filename': original_filename,
                'file_path': file_path,
                'file_size': file_info['size'],
                'file_type': file_info['file_type'],
                'text_content': text_content,
                'chunks': chunks,
                'metadata': {
                    **extraction_metadata,
                    **file_info,
                    'processing_time_seconds': processing_time,
                    'processed_at': datetime.now().isoformat()
                },
                'total_chunks': len(chunks),
                'processing_status': 'completed'
            }
            
            logger.info(f"Document processing completed in {processing_time:.2f}s: "
                       f"{len(chunks)} chunks created")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document {original_filename}: {e}")
            raise DocumentProcessingError(f"Processing failed: {e}")
    
    def _check_existing_document(self, file_path: str, original_filename: str) -> Optional[Dict[str, Any]]:
        """Check if document with same content already exists"""
        try:
            file_hash = generate_file_hash(file_path)
            
            db = get_db_session()
            try:
                # Look for existing document with same file hash pattern
                existing_docs = db.query(Document).filter(
                    Document.filename.like(f"{file_hash}_%"),
                    Document.original_filename == original_filename
                ).first()
                
                if existing_docs:
                    logger.info(f"Found existing document: {existing_docs.id}")
                    return {
                        'id': existing_docs.id,
                        'filename': existing_docs.filename,
                        'original_filename': existing_docs.original_filename,
                        'file_path': existing_docs.file_path,
                        'file_size': existing_docs.file_size,
                        'file_type': existing_docs.file_type,
                        'total_chunks': existing_docs.total_chunks,
                        'processing_status': existing_docs.processing_status,
                        'metadata': existing_docs.document_metadata,
                        'already_exists': True
                    }
                
                return None
                
            finally:
                db.close()
                
        except Exception as e:
            logger.error(f"Error checking existing document: {e}")
            return None
    
    def _validate_file(self, file_path: str, original_filename: str) -> Dict[str, Any]:
        """Validate file and extract basic information"""
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            raise DocumentProcessingError(f"File not found: {file_path}")
        
        # Check file type
        file_extension = path.suffix.lower()
        if file_extension not in settings.SUPPORTED_FORMATS:
            raise UnsupportedFileTypeError(
                f"Unsupported file type: {file_extension}. "
                f"Supported: {', '.join(settings.SUPPORTED_FORMATS)}"
            )
        
        # Get file info
        file_info = get_file_info(file_path)
        file_info['file_type'] = file_extension
        
        # Check file size
        max_size_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        if file_info['size'] > max_size_bytes:
            raise DocumentProcessingError(
                f"File too large: {file_info['size_mb']}MB > {settings.MAX_FILE_SIZE_MB}MB"
            )
        
        # Check if file is readable
        if not file_info['is_readable']:
            raise DocumentProcessingError(f"File is not readable: {file_path}")
        
        return file_info
    
    def _extract_text(self, file_path: str, file_type: str) -> tuple[str, Dict[str, Any]]:
        """Extract text using appropriate extractor"""
        extractor = self.extractors.get(file_type)
        
        if not extractor:
            raise UnsupportedFileTypeError(f"No extractor available for {file_type}")
        
        try:
            return extractor.extract_text(file_path)
        except Exception as e:
            logger.error(f"Text extraction failed for {file_path}: {e}")
            raise DocumentProcessingError(f"Text extraction failed: {e}")
    
    def _create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create text chunks using LangChain text splitter"""
        try:
            if not text.strip():
                logger.warning("Empty text provided for chunking")
                return []
            
            # Create LangChain document
            langchain_doc = LangChainDocument(page_content=text, metadata=metadata)
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([langchain_doc])
            
            # Convert to our format
            chunk_list = []
            for idx, chunk in enumerate(chunks):
                chunk_content = chunk.page_content.strip()
                
                chunk_data = {
                    'chunk_index': idx,
                    'content': chunk_content,
                    'metadata': {
                        **chunk.metadata,
                        'chunk_size': len(chunk_content),
                        'chunk_id': f"chunk_{idx}",
                        'chunk_hash': generate_content_hash(chunk_content)
                    }
                }
                
                # Only add non-empty chunks
                if chunk_data['content']:
                    chunk_list.append(chunk_data)
            
            logger.info(f"Created {len(chunk_list)} chunks from {len(text)} characters")
            return chunk_list
            
        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            raise DocumentProcessingError(f"Chunk creation failed: {e}")
    
    def save_to_database(self, processed_doc: Dict[str, Any]) -> int:
        """
        🔧 FIXED: Save processed document and chunks with optimized bulk insert

        Optimizations:
        - Use bulk_insert_mappings for large chunk sets (>100 chunks)
        - Batch commits to reduce transaction overhead
        - Better error handling and rollback
        """

        # If document already exists, return its ID
        if processed_doc.get('already_exists'):
            return processed_doc['id']

        db = get_db_session()
        try:
            logger.info(f"Saving document to database: {processed_doc['original_filename']}")

            # Create document record
            doc_record = Document(
                filename=processed_doc['filename'],
                original_filename=processed_doc['original_filename'],
                file_path=processed_doc['file_path'],
                file_size=processed_doc['file_size'],
                file_type=processed_doc['file_type'],
                processing_status="processing",
                total_chunks=processed_doc['total_chunks'],
                document_metadata=processed_doc['metadata']
            )

            db.add(doc_record)

            try:
                db.flush()  # Get the ID without committing
            except IntegrityError as e:
                if "duplicate key value violates unique constraint" in str(e):
                    db.rollback()
                    logger.warning(f"Document filename already exists, attempting to find existing record")

                    # Try to find the existing document
                    existing_doc = db.query(Document).filter(
                        Document.filename == processed_doc['filename']
                    ).first()

                    if existing_doc:
                        logger.info(f"Found existing document with ID: {existing_doc.id}")
                        return existing_doc.id
                    else:
                        # Generate a new unique filename with milliseconds
                        import time
                        timestamp_ms = int(time.time() * 1000)
                        file_hash = processed_doc['filename'].split('_')[0]
                        new_filename = f"{file_hash}_{timestamp_ms}_{processed_doc['original_filename']}"

                        doc_record.filename = new_filename
                        processed_doc['filename'] = new_filename

                        db.add(doc_record)
                        db.flush()
                        logger.info(f"Created new document with filename: {new_filename}")
                else:
                    raise e

            # 🔧 FIX: Use bulk insert for better performance
            chunks_count = len(processed_doc['chunks'])

            if chunks_count > 100:
                # For large documents, use bulk_insert_mappings (faster)
                logger.info(f"Using bulk insert for {chunks_count} chunks")

                chunk_mappings = []
                for chunk in processed_doc['chunks']:
                    chunk_mappings.append({
                        'document_id': doc_record.id,
                        'chunk_index': chunk['chunk_index'],
                        'content': chunk['content'],
                        'chunk_metadata': chunk['metadata']
                    })

                # Bulk insert in batches of 500
                batch_size = 500
                for i in range(0, len(chunk_mappings), batch_size):
                    batch = chunk_mappings[i:i + batch_size]
                    db.bulk_insert_mappings(DocumentChunk, batch)
                    logger.debug(f"Inserted batch {i//batch_size + 1}/{(len(chunk_mappings)-1)//batch_size + 1}")
            else:
                # For small documents, use add_all (better for relationships)
                logger.info(f"Using add_all for {chunks_count} chunks")
                chunk_records = []
                for chunk in processed_doc['chunks']:
                    chunk_record = DocumentChunk(
                        document_id=doc_record.id,
                        chunk_index=chunk['chunk_index'],
                        content=chunk['content'],
                        chunk_metadata=chunk['metadata']
                    )
                    chunk_records.append(chunk_record)

                db.add_all(chunk_records)

            # Update processing status
            doc_record.processing_status = "completed"

            # Commit all changes
            db.commit()
            db.refresh(doc_record)

            logger.info(f"Document saved successfully with ID: {doc_record.id} ({chunks_count} chunks)")
            return doc_record.id

        except Exception as e:
            db.rollback()
            logger.error(f"Error saving document to database: {e}")
            raise DocumentProcessingError(f"Database save failed: {e}")
        finally:
            db.close()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics"""
        db = get_db_session()
        try:
            total_docs = db.query(Document).count()
            completed_docs = db.query(Document).filter(
                Document.processing_status == "completed"
            ).count()
            total_chunks = db.query(DocumentChunk).count()
            
            return {
                'total_documents': total_docs,
                'completed_documents': completed_docs,
                'pending_documents': total_docs - completed_docs,
                'total_chunks': total_chunks,
                'average_chunks_per_doc': total_chunks / max(completed_docs, 1)
            }
        except Exception as e:
            logger.error(f"Error getting processing stats: {e}")
            return {}
        finally:
            db.close()

# Convenience function
def process_and_save_document(file_path: str, original_filename: str) -> int:
    """Process document and save to database"""
    processor = DocumentProcessor()
    processed_doc = processor.process_document(file_path, original_filename)
    document_id = processor.save_to_database(processed_doc)
    return document_id
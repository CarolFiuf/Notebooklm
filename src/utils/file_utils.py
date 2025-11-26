import os
import hashlib
import shutil
from pathlib import Path
from typing import Optional, Tuple, Union
import logging

from config.settings import settings
from .exceptions import FileProcessingError, UnsupportedFileTypeError

logger = logging.getLogger(__name__)

def validate_file_type(filename: str) -> bool:
    """Check if file type is supported"""
    file_extension = Path(filename).suffix.lower()
    return file_extension in settings.SUPPORTED_FORMATS

def validate_file_size(file_path: str) -> bool:
    """Check if file size is within limits"""
    file_size = os.path.getsize(file_path)
    max_size_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    return file_size <= max_size_bytes

def generate_file_hash(file_path: str) -> str:
    """Generate SHA-256 hash for file from file path"""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()[:16]  # Use first 16 chars
    except Exception as e:
        logger.error(f"Error generating hash for {file_path}: {e}")
        raise FileProcessingError(f"Failed to generate hash: {e}")

def generate_content_hash(content: Union[str, bytes]) -> str:
    """
    Generate SHA-256 hash for text or byte content
    
    Args:
        content: String or bytes content to hash
        
    Returns:
        First 8 characters of SHA-256 hash
    """
    try:
        hash_sha256 = hashlib.sha256()
        
        if isinstance(content, str):
            hash_sha256.update(content.encode('utf-8'))
        elif isinstance(content, bytes):
            hash_sha256.update(content)
        else:
            raise ValueError(f"Content must be str or bytes, got {type(content)}")
        
        return hash_sha256.hexdigest()[:8]  # Use first 8 chars for chunk hashes
        
    except Exception as e:
        logger.error(f"Error generating content hash: {e}")
        raise FileProcessingError(f"Failed to generate content hash: {e}")

def save_uploaded_file(uploaded_file, filename_prefix: Optional[str] = None) -> Tuple[str, str]:
    """
    Save uploaded file and return (file_path, unique_filename)
    """
    try:
        # Validate file type
        if not validate_file_type(uploaded_file.name):
            raise UnsupportedFileTypeError(
                f"File type not supported: {Path(uploaded_file.name).suffix}"
            )
        
        # Generate unique filename
        file_extension = Path(uploaded_file.name).suffix
        prefix = filename_prefix or "doc"
        import uuid
        unique_filename = f"{prefix}_{uuid.uuid4().hex[:8]}_{uploaded_file.name}"
        
        # Create file path
        file_path = settings.DOCUMENTS_DIR / unique_filename
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Validate file size after saving
        if not validate_file_size(str(file_path)):
            os.remove(file_path)  # Clean up
            raise FileProcessingError(
                f"File size exceeds limit of {settings.MAX_FILE_SIZE_MB}MB"
            )
        
        logger.info(f"File saved: {unique_filename}")
        return str(file_path), unique_filename
        
    except Exception as e:
        logger.error(f"Error saving file {uploaded_file.name}: {e}")
        raise FileProcessingError(f"Failed to save file: {e}")

def cleanup_old_files(days: int = 30) -> int:
    """Clean up files older than specified days"""
    import time
    
    count = 0
    cutoff_time = time.time() - (days * 24 * 60 * 60)
    
    try:
        for file_path in settings.DOCUMENTS_DIR.glob("*"):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                os.remove(file_path)
                count += 1
                logger.info(f"Cleaned up old file: {file_path.name}")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
    
    return count

def get_file_info(file_path: str) -> dict:
    """Get file information"""
    try:
        path = Path(file_path)
        stat = path.stat()
        
        return {
            'filename': path.name,
            'size': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'extension': path.suffix.lower(),
            'created': stat.st_ctime,
            'modified': stat.st_mtime,
            'exists': path.exists(),
            'is_readable': os.access(file_path, os.R_OK)
        }
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {e}")
        return {}
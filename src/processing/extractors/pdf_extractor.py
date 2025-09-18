import logging
from typing import Tuple, Dict, Any, List
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
from PIL import Image

from src.utils.exceptions import FileProcessingError
from src.utils.text_utils import clean_text

logger = logging.getLogger(__name__)

class PDFExtractor:
    """PDF text extraction using PyMuPDF and pdfplumber"""
    
    def __init__(self):
        self.min_text_length = 50  # Minimum text length to consider extraction successful
    
    def extract_text(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text from PDF file
        
        Returns:
            Tuple[str, Dict[str, Any]]: (extracted_text, metadata)
        """
        try:
            logger.info(f"Extracting text from PDF: {Path(file_path).name}")
            
            # Try PyMuPDF first (faster)
            text_content, metadata = self._extract_with_pymupdf(file_path)
            
            # If extraction is poor, try pdfplumber
            if len(text_content.strip()) < self.min_text_length:
                logger.info("PyMuPDF extraction insufficient, trying pdfplumber...")
                text_content, plumber_metadata = self._extract_with_pdfplumber(file_path)
                metadata.update(plumber_metadata)
            
            # Final validation
            if len(text_content.strip()) < self.min_text_length:
                logger.warning(f"Very little text extracted from {file_path}")
                
            # Clean extracted text
            text_content = clean_text(text_content)
            
            # Add extraction metadata
            metadata.update({
                'extraction_method': 'pymupdf_pdfplumber',
                'text_length': len(text_content),
                'word_count': len(text_content.split()),
                'extraction_success': len(text_content) >= self.min_text_length
            })
            
            logger.info(f"PDF extraction completed: {len(text_content)} characters")
            return text_content, metadata
            
        except Exception as e:
            logger.error(f"Error extracting PDF text from {file_path}: {e}")
            raise FileProcessingError(f"PDF extraction failed: {e}")
    
    def _extract_with_pymupdf(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text using PyMuPDF (fitz)"""
        try:
            doc = fitz.open(file_path)
            
            # Extract metadata
            metadata = {
                'total_pages': doc.page_count,
                'title': doc.metadata.get('title', '').strip(),
                'author': doc.metadata.get('author', '').strip(),
                'subject': doc.metadata.get('subject', '').strip(),
                'creator': doc.metadata.get('creator', '').strip(),
                'producer': doc.metadata.get('producer', '').strip(),
                'creation_date': doc.metadata.get('creationDate', ''),
                'modification_date': doc.metadata.get('modDate', ''),
                'pages_with_text': 0,
                'pages_with_images': 0
            }
            
            # Extract text page by page
            pages_text = []
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                page_text = page.get_text().strip()
                
                if page_text:
                    pages_text.append(f"--- Page {page_num + 1} ---\n{page_text}")
                    metadata['pages_with_text'] += 1
                
                # Check for images
                if page.get_images():
                    metadata['pages_with_images'] += 1
            
            doc.close()
            
            # Combine all page text
            text_content = "\n\n".join(pages_text)
            
            return text_content, metadata
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            return "", {}
    
    def _extract_with_pdfplumber(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text using pdfplumber (better for complex layouts)"""
        try:
            pages_text = []
            metadata = {'pdfplumber_pages': 0, 'tables_found': 0}
            
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    page_text = page.extract_text()
                    
                    if page_text:
                        pages_text.append(f"--- Page {page_num + 1} ---\n{page_text}")
                        metadata['pdfplumber_pages'] += 1
                    
                    # Extract tables if any
                    tables = page.extract_tables()
                    if tables:
                        metadata['tables_found'] += len(tables)
                        for table in tables:
                            # Convert table to text
                            table_text = self._table_to_text(table)
                            if table_text:
                                pages_text.append(f"--- Table on Page {page_num + 1} ---\n{table_text}")
            
            text_content = "\n\n".join(pages_text)
            return text_content, metadata
            
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            return "", {}
    
    def _table_to_text(self, table: List[List[str]]) -> str:
        """Convert table data to readable text"""
        try:
            if not table:
                return ""
            
            text_rows = []
            for row in table:
                if row and any(cell for cell in row if cell):
                    # Clean and join cells
                    cleaned_row = [str(cell).strip() if cell else "" for cell in row]
                    text_rows.append(" | ".join(cleaned_row))
            
            return "\n".join(text_rows)
            
        except Exception as e:
            logger.warning(f"Error converting table to text: {e}")
            return ""
import logging
from typing import Tuple, Dict, Any
from pathlib import Path
import chardet

from src.utils.exceptions import FileProcessingError
from src.utils.text_utils import clean_text

logger = logging.getLogger(__name__)

class TextExtractor:
    """Plain text file extraction (.txt, .md)"""
    
    def __init__(self):
        self.encodings_to_try = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'iso-8859-1']
    
    def extract_text(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text from plain text files
        
        Returns:
            Tuple[str, Dict[str, Any]]: (extracted_text, metadata)
        """
        try:
            file_path_obj = Path(file_path)
            logger.info(f"Extracting text from: {file_path_obj.name}")
            
            # Detect encoding
            encoding = self._detect_encoding(file_path)
            logger.info(f"Detected encoding: {encoding}")
            
            # Read file with detected encoding
            text_content = self._read_file_with_encoding(file_path, encoding)
            
            # Clean text
            text_content = clean_text(text_content)
            
            # Create metadata
            metadata = {
                'file_encoding': encoding,
                'file_extension': file_path_obj.suffix.lower(),
                'text_length': len(text_content),
                'line_count': len(text_content.split('\n')),
                'word_count': len(text_content.split()),
                'character_count': len(text_content),
                'extraction_method': 'direct_read'
            }
            
            # Special handling for Markdown files
            if file_path_obj.suffix.lower() == '.md':
                metadata.update(self._analyze_markdown(text_content))
            
            logger.info(f"Text extraction completed: {len(text_content)} characters")
            return text_content, metadata
            
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise FileProcessingError(f"Text extraction failed: {e}")
    
    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding"""
        try:
            # Use chardet for encoding detection
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                
                if result and result['encoding'] and result['confidence'] > 0.7:
                    return result['encoding']
                
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}")
        
        # Fallback to UTF-8
        return 'utf-8'
    
    def _read_file_with_encoding(self, file_path: str, preferred_encoding: str) -> str:
        """Read file with specified encoding, with fallbacks"""
        
        # Try preferred encoding first
        encodings = [preferred_encoding] + [enc for enc in self.encodings_to_try 
                                          if enc != preferred_encoding]
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
                    
            except UnicodeDecodeError:
                logger.warning(f"Failed to read with encoding {encoding}")
                continue
            except Exception as e:
                logger.error(f"Error reading file with {encoding}: {e}")
                continue
        
        # If all encodings fail, try with error handling
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                logger.warning("Reading file with error handling (some characters may be lost)")
                return f.read()
        except Exception as e:
            raise FileProcessingError(f"Could not read file with any encoding: {e}")
    
    def _analyze_markdown(self, text: str) -> Dict[str, Any]:
        """Analyze markdown content"""
        import re
        
        analysis = {
            'is_markdown': True,
            'headers': [],
            'links': [],
            'images': [],
            'code_blocks': 0,
            'lists': 0
        }
        
        try:
            # Find headers
            header_pattern = r'^(#{1,6})\s+(.+)$'
            headers = re.findall(header_pattern, text, re.MULTILINE)
            analysis['headers'] = [{'level': len(h[0]), 'text': h[1]} for h in headers]
            
            # Find links
            link_pattern = r'\[([^\]]+)\]\(([^\)]+)\)'
            links = re.findall(link_pattern, text)
            analysis['links'] = [{'text': l[0], 'url': l[1]} for l in links]
            
            # Find images
            image_pattern = r'!\[([^\]]*)\]\(([^\)]+)\)'
            images = re.findall(image_pattern, text)
            analysis['images'] = [{'alt': i[0], 'url': i[1]} for i in images]
            
            # Count code blocks
            analysis['code_blocks'] = len(re.findall(r'```', text)) // 2
            
            # Count lists
            list_pattern = r'^[\s]*[-\*\+]\s'
            analysis['lists'] = len(re.findall(list_pattern, text, re.MULTILINE))
            
        except Exception as e:
            logger.warning(f"Markdown analysis failed: {e}")
        
        return analysis
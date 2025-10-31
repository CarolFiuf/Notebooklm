"""
✅ MIGRATED: LangChain-based PDF Extractor
Replaced custom PyMuPDF implementation with LangChain loaders

Benefits:
- Built-in PDF parsing
- Better metadata extraction
- Support for multiple PDF libraries
- Community maintained
"""
import logging
from typing import Tuple, Dict, Any, List
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader, PDFPlumberLoader

from src.utils.exceptions import FileProcessingError
from src.utils.text_utils import clean_text

logger = logging.getLogger(__name__)

class PDFExtractor:
    """
    ✅ MIGRATED: PDF extraction using LangChain loaders

    Uses PyMuPDFLoader as primary, PDFPlumberLoader as fallback
    """

    def __init__(self):
        self.min_text_length = 50
        logger.info("✅ PDF Extractor initialized with LangChain loaders")

    def extract_text(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        ✅ MIGRATED: Extract text using LangChain loaders

        Returns:
            Tuple[str, Dict[str, Any]]: (extracted_text, metadata)
        """
        try:
            logger.info(f"Extracting PDF with LangChain: {Path(file_path).name}")

            # ✅ Try PyMuPDFLoader first (faster)
            extraction_method = None
            pymupdf_error = None

            try:
                loader = PyMuPDFLoader(file_path)
                documents = loader.load()

                # Combine all pages
                text_content = "\n\n".join([doc.page_content for doc in documents])

                # Extract metadata from first page
                metadata = documents[0].metadata if documents else {}
                metadata['total_pages'] = len(documents)
                extraction_method = 'pymupdf_langchain'

            except Exception as e:
                pymupdf_error = e
                logger.warning(f"PyMuPDF failed: {e}, trying PDFPlumber...")

                # ✅ Fallback to PDFPlumberLoader
                try:
                    loader = PDFPlumberLoader(file_path)
                    documents = loader.load()

                    text_content = "\n\n".join([doc.page_content for doc in documents])

                    metadata = documents[0].metadata if documents else {}
                    metadata['total_pages'] = len(documents)
                    extraction_method = 'pdfplumber_langchain'

                except Exception as e2:
                    logger.error(
                        f"Both PDF extractors failed for {file_path}:\n"
                        f"  - PyMuPDF error: {pymupdf_error}\n"
                        f"  - PDFPlumber error: {e2}"
                    )
                    raise FileProcessingError(
                        f"PDF extraction failed with all methods. "
                        f"PyMuPDF: {pymupdf_error}, PDFPlumber: {e2}"
                    )

            # Set extraction method in metadata
            if extraction_method:
                metadata['extraction_method'] = extraction_method

            # Validate extraction
            if len(text_content.strip()) < self.min_text_length:
                logger.warning(f"Very little text extracted from {file_path}")

            # Clean extracted text
            text_content = clean_text(text_content)

            # Add extraction stats
            metadata.update({
                'text_length': len(text_content),
                'word_count': len(text_content.split()),
                'using_langchain': True
            })

            logger.info(f"✅ Extracted {len(text_content)} chars from {metadata.get('total_pages', 0)} pages")

            return text_content, metadata

        except Exception as e:
            logger.error(f"Error extracting PDF: {e}")
            raise FileProcessingError(f"PDF extraction failed: {e}")

    def extract_pages(self, file_path: str) -> List[Dict[str, Any]]:
        """
        ✅ Extract text page by page using LangChain

        Returns:
            List of dicts with page content and metadata
        """
        try:
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()

            pages = []
            for i, doc in enumerate(documents):
                page_data = {
                    'page_number': i + 1,
                    'content': clean_text(doc.page_content),
                    'metadata': doc.metadata
                }
                pages.append(page_data)

            logger.info(f"✅ Extracted {len(pages)} pages separately")
            return pages

        except Exception as e:
            logger.error(f"Error extracting pages: {e}")
            raise FileProcessingError(f"Page extraction failed: {e}")

"""
Legal Document Information Extractor for Vietnamese Legal System
Trích xuất thông tin từ văn bản pháp luật Việt Nam
"""

import re
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class VietnameseLegalExtractor:
    """Trích xuất thông tin từ văn bản pháp luật Việt Nam"""

    def __init__(self):
        # Regex patterns cho cấu trúc văn bản
        self.patterns = {
            'article': re.compile(r'Điều\s+(\d+)', re.IGNORECASE),
            'clause': re.compile(r'Khoản\s+(\d+)', re.IGNORECASE),
            'point': re.compile(r'[Đ|đ]iểm\s+([a-z])', re.IGNORECASE),
            'chapter': re.compile(r'Chương\s+([IVXLCDM]+)', re.IGNORECASE),
        }

        # Patterns cho metadata
        self.metadata_patterns = {
            'document_number': re.compile(r'(\d+/\d+/[A-Z\-]+)', re.IGNORECASE),
            'date': re.compile(
                r'ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})',
                re.IGNORECASE
            ),
        }

        # Phân loại văn bản theo thứ bậc hiệu lực
        self.document_hierarchy = [
            'Hiến pháp', 'Luật', 'Bộ luật', 'Pháp lệnh',
            'Nghị quyết', 'Nghị định', 'Quyết định', 'Thông tư'
        ]

    def extract_structure(self, text: str) -> Dict[str, List[Dict]]:
        """Trích xuất cấu trúc văn bản (Điều, Khoản, Điểm...)"""
        structure = {'articles': [], 'clauses': [], 'points': [], 'chapters': []}

        for match in self.patterns['article'].finditer(text):
            structure['articles'].append({
                'number': int(match.group(1)),
                'position': match.start()
            })

        for match in self.patterns['chapter'].finditer(text):
            structure['chapters'].append({
                'number': match.group(1),
                'position': match.start()
            })

        return structure

    def extract_metadata(self, text: str) -> Dict:
        """Trích xuất metadata từ văn bản pháp luật"""
        metadata = {
            'document_number': None,
            'issue_date': None,
            'document_type': None
        }

        # Trích xuất số hiệu
        doc_num_match = self.metadata_patterns['document_number'].search(text[:1000])
        if doc_num_match:
            metadata['document_number'] = doc_num_match.group(1)

        # Trích xuất ngày ban hành
        date_match = self.metadata_patterns['date'].search(text[:1000])
        if date_match:
            day, month, year = date_match.groups()
            metadata['issue_date'] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"

        # Xác định loại văn bản
        metadata['document_type'] = self.classify_document_type(text)

        return metadata

    def classify_document_type(self, text: str) -> Optional[str]:
        """Phân loại loại văn bản"""
        text_sample = text[:500].upper()
        for doc_type in self.document_hierarchy:
            if doc_type.upper() in text_sample:
                return doc_type
        return None

    def split_by_legal_structure(self, text: str, max_chunk_size: int = 1200,
                                 overlap: int = 150) -> List[Dict]:
        """Chia văn bản theo cấu trúc pháp lý"""
        structure = self.extract_structure(text)
        chunks = []

        if not structure['articles']:
            return self._split_generic(text, max_chunk_size, overlap)

        # Chia theo từng Điều
        for i, article in enumerate(structure['articles']):
            start = article['position']
            end = structure['articles'][i + 1]['position'] if i + 1 < len(structure['articles']) else len(text)

            article_content = text[start:end].strip()

            if len(article_content) <= max_chunk_size:
                chunks.append({
                    'content': article_content,
                    'article': article['number'],
                    'type': 'article'
                })
            else:
                # Chia nhỏ hơn nếu quá dài
                sub_chunks = self._split_generic(article_content, max_chunk_size, overlap)
                for chunk in sub_chunks:
                    chunk['article'] = article['number']
                chunks.extend(sub_chunks)

        return chunks

    def _split_generic(self, text: str, max_size: int, overlap: int) -> List[Dict]:
        """Chia văn bản theo cách thông thường"""
        chunks = []
        start = 0

        while start < len(text):
            end = start + max_size
            chunks.append({
                'content': text[start:end],
                'type': 'generic'
            })
            start += (max_size - overlap)

        return chunks

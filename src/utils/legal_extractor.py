"""
Legal Document Information Extractor for Vietnamese Legal System
Trích xuất thông tin từ văn bản pháp luật Việt Nam

ENHANCED FOR VIETNAMESE LEGAL DOMAIN:
- Cải thiện metadata extraction (số hiệu, cơ quan, người ký, ngày hiệu lực)
- Nhận diện văn bản liên quan (thay thế, sửa đổi, bổ sung)
- Trích xuất cấu trúc phân cấp (Phần > Chương > Mục > Điều > Khoản > Điểm)
- Legal entity recognition
- Citation format chuẩn
"""

import re
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class VietnameseLegalExtractor:
    """Trích xuất thông tin từ văn bản pháp luật Việt Nam - Enhanced version"""

    def __init__(self):
        # Regex patterns cho cấu trúc văn bản phân cấp
        self.patterns = {
            'part': re.compile(r'Phần\s+(?:thứ\s+)?([IVXLCDM]+|[A-Z]+)', re.IGNORECASE),
            'chapter': re.compile(r'Chương\s+([IVXLCDM]+)', re.IGNORECASE),
            'section': re.compile(r'Mục\s+(\d+)', re.IGNORECASE),
            # ✅ FIXED: Điều pattern không yêu cầu dấu chấm
            # Match: "Điều X." hoặc "Điều X " hoặc "Điều X\n"
            'article': re.compile(r'Điều\s+(\d+)', re.IGNORECASE),
            # 🔧 FIX: Pattern cho "Khoản X" và số đầu dòng "1. ", "2. "
            'clause': re.compile(r'(?:Khoản\s+(\d+)|^(\d+)\.\s+[A-ZÀÁẢÃẠ])', re.IGNORECASE | re.MULTILINE),
            'numbered_item': re.compile(r'^(\d+)\.\s+', re.MULTILINE),  # "1. Hoạt động...", "2. Chạy tàu..."
            'point': re.compile(r'[Đ|đ]iểm\s+([a-z])', re.IGNORECASE),
        }

        # 🔧 SIMPLIFIED: Chỉ giữ patterns thiết yếu
        self.metadata_patterns = {
            # Số hiệu văn bản (hỗ trợ format "Luật số: 95/2025/QH15")
            'document_number': re.compile(
                r'(?:Luật|Nghị định|Quyết định|Thông tư|Nghị quyết|Pháp lệnh|Chỉ thị)\s*(?:số[:\s]*)?(\d+[/-]\d{4}[/-][A-Z\d\-]+)',
                re.IGNORECASE
            ),

            # Cơ quan ban hành
            'issuing_authority': re.compile(
                r'(QUỐC\s+HỘI|CHÍNH\s+PHỦ|THỦ\s+TƯỚNG\s+CHÍNH\s+PHỦ|'
                r'CHỦ\s+TỊCH\s+NƯỚC|BỘ\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ\s\-]+)',
                re.IGNORECASE
            ),
        }

        # Phân loại văn bản theo thứ bậc hiệu lực
        self.document_hierarchy = {
            'Hiến pháp': 1,
            'Luật': 2,
            'Bộ luật': 2,
            'Pháp lệnh': 3,
            'Nghị quyết': 4,
            'Nghị định': 5,
            'Quyết định': 6,
            'Thông tư': 7,
            'Chỉ thị': 8,
            'Quy định': 9,
            'Quy chế': 10
        }

        # Legal entities patterns
        self.legal_entities = {
            'organizations': re.compile(
                r'\b(Quốc\s+hội|Chính\s+phủ|Bộ\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ\s\-]+|'
                r'Ủy\s+ban\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ\s]+|'
                r'Tòa\s+án\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ\s]+)\b',
                re.IGNORECASE
            ),
            'legal_terms': re.compile(
                r'\b(vi\s+phạm|xử\s+phạt|trách\s+nhiệm|quyền|nghĩa\s+vụ|'
                r'khiếu\s+nại|tố\s+cáo|tranh\s+chấp|thi\s+hành|áp\s+dụng)\b',
                re.IGNORECASE
            )
        }

    def preprocess_text(self, text: str) -> str:
        """
        🔧 NEW: Làm sạch văn bản trước khi xử lý
        - Loại bỏ gạch dưới liên tiếp (________)
        - Chuẩn hóa khoảng trắng
        - Loại bỏ ký tự đặc biệt không cần thiết
        """
        # Loại bỏ gạch dưới liên tiếp (3 trở lên)
        text = re.sub(r'_{3,}', '', text)

        # Chuẩn hóa nhiều khoảng trắng thành 1
        text = re.sub(r'[ \t]+', ' ', text)

        # Chuẩn hóa nhiều xuống dòng thành tối đa 2
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def extract_structure(self, text: str) -> Dict[str, List[Dict]]:
        """
        Trích xuất cấu trúc phân cấp đầy đủ của văn bản (Phần > Chương > Mục > Điều > Khoản > Điểm)
        🔧 IMPROVED: Nhận diện cả "Khoản X" và số thứ tự "1. ", "2. "
        """
        structure = {
            'parts': [],
            'chapters': [],
            'sections': [],
            'articles': [],
            'clauses': [],
            'points': []
        }

        # Trích xuất Phần
        for match in self.patterns['part'].finditer(text):
            structure['parts'].append({
                'number': match.group(1),
                'position': match.start(),
                'text': match.group(0)
            })

        # Trích xuất Chương
        for match in self.patterns['chapter'].finditer(text):
            structure['chapters'].append({
                'number': match.group(1),
                'position': match.start(),
                'text': match.group(0)
            })

        # Trích xuất Mục
        for match in self.patterns['section'].finditer(text):
            structure['sections'].append({
                'number': int(match.group(1)),
                'position': match.start(),
                'text': match.group(0)
            })

        # Trích xuất Điều
        for match in self.patterns['article'].finditer(text):
            structure['articles'].append({
                'number': int(match.group(1)),
                'position': match.start(),
                'text': match.group(0)
            })

        # 🔧 FIX: Trích xuất cả số thứ tự đầu dòng (1., 2., ...) trong context của Điều
        for match in self.patterns['numbered_item'].finditer(text):
            # Chỉ lấy nếu nằm trong một Điều (không phải ở header)
            pos = match.start()
            # Kiểm tra có thuộc một Điều nào không
            in_article = False
            for article in structure.get('articles', []):
                if article['position'] < pos:
                    in_article = True
                    break

            if in_article:
                structure['clauses'].append({
                    'number': int(match.group(1)),
                    'position': match.start(),
                    'text': match.group(0)
                })

        # Trích xuất Điểm
        for match in self.patterns['point'].finditer(text):
            structure['points'].append({
                'letter': match.group(1),
                'position': match.start(),
                'text': match.group(0)
            })

        return structure

    def extract_metadata(self, text: str) -> Dict:
        """
        🔧 SIMPLIFIED: Trích xuất metadata thiết yếu từ văn bản pháp luật Việt Nam
        Chỉ giữ lại 4 fields quan trọng nhất
        """
        # Làm sạch text trước
        text = self.preprocess_text(text)

        # ✅ CHỈ GIỮ 4 METADATA THIẾT YẾU
        metadata = {
            'document_type': None,        # Loại văn bản (Luật, Nghị định, Thông tư...)
            'document_number': None,      # Số hiệu (VD: 95/2025/QH15)
            'issuing_authority': None,    # Cơ quan ban hành
            'hierarchy_level': None       # Cấp độ văn bản (1-10)
        }

        # Tìm kiếm trong phần đầu văn bản
        header_text = text[:2000]

        # 1. Xác định loại văn bản và cấp độ
        doc_type = self.classify_document_type(text)
        metadata['document_type'] = doc_type
        if doc_type:
            metadata['hierarchy_level'] = self.document_hierarchy.get(doc_type)

        # 2. Trích xuất số hiệu
        doc_num_match = self.metadata_patterns['document_number'].search(header_text)
        if doc_num_match:
            metadata['document_number'] = doc_num_match.group(1).strip()

        # 3. Trích xuất cơ quan ban hành
        authority_match = self.metadata_patterns['issuing_authority'].search(header_text)
        if authority_match:
            metadata['issuing_authority'] = authority_match.group(1).strip()

        return metadata

    def classify_document_type(self, text: str) -> Optional[str]:
        """
        🔧 IMPROVED: Phân loại dựa trên TITLE LINE, không phải toàn bộ header
        Tránh nhầm lẫn khi có "Căn cứ Hiến pháp..." trong header
        """
        # 🔧 FIX: Tìm title line, xử lý cả trường hợp dính nhau "LUẬTĐƯỜNG SẮT"
        title_pattern = re.compile(
            r'^(HIẾN\s*PHÁP|BỘ\s*LUẬT|LUẬT|NGHỊ\s*ĐỊNH|QUYẾT\s*ĐỊNH|THÔNG\s*TƯ|NGHỊ\s*QUYẾT|PHÁP\s*LỆNH|CHỈ\s*THỊ|QUY\s*ĐỊNH|QUY\s*CHẾ)(?:[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴĐ\s]*)?$',
            re.IGNORECASE | re.MULTILINE
        )

        # Tìm trong 1500 ký tự đầu
        for line in text[:1500].split('\n'):
            line_clean = line.strip()
            match = title_pattern.match(line_clean)
            if match:
                doc_type_upper = match.group(1).upper()
                # Map về format chuẩn
                for doc_type, _ in self.document_hierarchy.items():
                    if doc_type.upper() == doc_type_upper:
                        logger.info(f"Detected document type: {doc_type} from title line: {line_clean[:50]}")
                        return doc_type

        # Fallback: tìm theo cách cũ nhưng ưu tiên thấp hơn
        text_sample = text[:1000].upper()
        sorted_types = sorted(
            self.document_hierarchy.items(),
            key=lambda x: x[1],
            reverse=True  # Ưu tiên từ thấp đến cao để tránh "Hiến pháp" trong "Căn cứ"
        )

        for doc_type, _ in sorted_types:
            if doc_type.upper() in text_sample:
                logger.warning(f"Fallback detection for document type: {doc_type}")
                return doc_type

        return None

    def extract_legal_entities(self, text: str) -> Dict[str, List[str]]:
        """Trích xuất các thực thể pháp lý (cơ quan, thuật ngữ pháp lý)"""
        entities = {
            'organizations': [],
            'legal_terms': []
        }

        # Trích xuất tên tổ chức/cơ quan
        org_matches = self.legal_entities['organizations'].finditer(text)
        seen_orgs = set()
        for match in org_matches:
            org = match.group(1).strip()
            if org not in seen_orgs:
                entities['organizations'].append(org)
                seen_orgs.add(org)

        # Trích xuất thuật ngữ pháp lý
        term_matches = self.legal_entities['legal_terms'].finditer(text)
        seen_terms = set()
        for match in term_matches:
            term = match.group(1).strip()
            if term not in seen_terms:
                entities['legal_terms'].append(term)
                seen_terms.add(term)

        return entities

    def create_citation(self, metadata: Dict, article: Optional[int] = None,
                       clause: Optional[int] = None, point: Optional[str] = None) -> str:
        """
        Tạo citation chuẩn cho văn bản pháp luật Việt Nam

        Format: Theo [Tên văn bản] [Số hiệu], Điều X, Khoản Y, Điểm z
        VD: Theo Bộ luật Dân sự số 91/2015/QH13, Điều 1, Khoản 1
        """
        parts = []

        # Loại văn bản
        if metadata.get('document_type'):
            parts.append(metadata['document_type'])

        # Số hiệu
        if metadata.get('document_number'):
            parts.append(f"số {metadata['document_number']}")

        citation = "Theo " + " ".join(parts) if parts else "Theo văn bản"

        # Thêm Điều, Khoản, Điểm nếu có
        if article:
            citation += f", Điều {article}"
        if clause:
            citation += f", Khoản {clause}"
        if point:
            citation += f", Điểm {point}"

        return citation

    def split_by_legal_structure(self, text: str, max_chunk_size: int = 1200,
                                 overlap: int = 150) -> List[Dict]:
        """
        Chia văn bản theo cấu trúc pháp lý phân cấp
        Ưu tiên: Điều > Khoản > Phân đoạn tự nhiên
        """
        structure = self.extract_structure(text)
        chunks = []

        if not structure['articles']:
            # Không tìm thấy Điều, chia theo cách thông thường
            return self._split_generic(text, max_chunk_size, overlap)

        # Chia theo từng Điều
        articles = structure['articles']
        for i, article in enumerate(articles):
            start = article['position']
            end = articles[i + 1]['position'] if i + 1 < len(articles) else len(text)

            article_content = text[start:end].strip()

            # Tìm chapter/section chứa điều này (nếu có)
            parent_chapter = self._find_parent_structure(
                article['position'],
                structure['chapters']
            )
            parent_section = self._find_parent_structure(
                article['position'],
                structure['sections']
            )

            chunk_metadata = {
                'article': article['number'],
                'chapter': parent_chapter['number'] if parent_chapter else None,
                'section': parent_section['number'] if parent_section else None,
                'type': 'article'
            }

            if len(article_content) <= max_chunk_size:
                # Điều vừa đủ, lưu nguyên
                # ✅ FIX: Only add if content is meaningful (>= 100 chars)
                if len(article_content) >= 100:
                    chunks.append({
                        'content': article_content,
                        **chunk_metadata
                    })
                else:
                    logger.warning(f"Skipping article {chunk_metadata.get('article')} - too small ({len(article_content)} bytes)")
            else:
                # Điều quá dài, cần chia nhỏ theo Khoản hoặc generic
                sub_chunks = self._split_long_article(
                    article_content,
                    max_chunk_size,
                    overlap,
                    start_position=start
                )
                for chunk in sub_chunks:
                    chunk.update(chunk_metadata)
                chunks.extend(sub_chunks)

        return chunks

    def _find_parent_structure(self, position: int, parent_list: List[Dict]) -> Optional[Dict]:
        """Tìm cấu trúc cha (chapter/section) chứa vị trí này"""
        for i, parent in enumerate(parent_list):
            next_pos = parent_list[i + 1]['position'] if i + 1 < len(parent_list) else float('inf')
            if parent['position'] <= position < next_pos:
                return parent
        return None

    def _split_long_article(self, content: str, max_size: int, overlap: int,
                           start_position: int) -> List[Dict]:
        """
        🔧 IMPROVED: Chia Điều dài thành các chunks nhỏ hơn
        Ưu tiên: số thứ tự "1. ", "2. " > "Khoản X" > generic split
        """
        chunks = []

        # 🔧 FIX: Tìm các khoản theo số thứ tự đầu dòng "1. ", "2. ", ...
        numbered_pattern = re.compile(r'^(\d+)\.\s+', re.MULTILINE)
        numbered_items = list(numbered_pattern.finditer(content))

        # Fallback: tìm "Khoản X"
        clause_pattern = re.compile(r'Khoản\s+\d+', re.IGNORECASE)
        clauses = list(clause_pattern.finditer(content))

        # Chọn pattern phù hợp nhất (ưu tiên numbered_items nếu nhiều hơn)
        if len(numbered_items) > len(clauses):
            items_to_split = numbered_items
            split_type = 'numbered_clause'
        else:
            items_to_split = clauses
            split_type = 'clause'

        if not items_to_split or len(content) < max_size * 1.5:
            # Không có khoản hoặc không quá dài, chia generic
            return self._split_generic(content, max_size, overlap)

        # Chia theo các khoản tìm được
        for i, item_match in enumerate(items_to_split):
            item_start = item_match.start()
            item_end = items_to_split[i + 1].start() if i + 1 < len(items_to_split) else len(content)

            item_content = content[item_start:item_end].strip()

            if len(item_content) <= max_size:
                # ✅ FIX: Only add if content is meaningful (>= 100 chars)
                if len(item_content) >= 100:
                    chunks.append({
                        'content': item_content,
                        'type': split_type
                    })
                else:
                    logger.debug(f"Skipping {split_type} - too small ({len(item_content)} bytes)")
            else:
                # Khoản vẫn quá dài, chia generic
                sub_chunks = self._split_generic(item_content, max_size, overlap)
                chunks.extend(sub_chunks)

        return chunks

    def _split_generic(self, text: str, max_size: int, overlap: int) -> List[Dict]:
        """
        🔧 FIXED: Chia văn bản theo cách thông thường với overlap
        Fix vòng lặp vô hạn khi overlap > 0
        """
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + max_size, text_len)

            # Tìm điểm ngắt tự nhiên gần end
            if end < text_len:
                # Tìm dấu xuống dòng, chấm, hoặc dấu phẩy gần nhất
                natural_breaks = [
                    text.rfind('\n\n', start, end),
                    text.rfind('\n', start, end),
                    text.rfind('. ', start, end),
                    text.rfind('; ', start, end),
                ]
                best_break = max(b for b in natural_breaks if b > start)
                if best_break > start + max_size * 0.5:  # Giảm threshold từ 0.7 -> 0.5
                    end = best_break + 1

            chunk_content = text[start:end].strip()

            # ✅ FIX: Skip chunks that are too small (likely incomplete references like "khoản 3 Điều 148")
            # Minimum 100 characters to ensure meaningful content
            if len(chunk_content) < 100:
                logger.warning(f"Skipping chunk too small ({len(chunk_content)} bytes): {chunk_content[:50]}...")
            elif chunk_content:
                chunks.append({
                    'content': chunk_content,
                    'type': 'generic'
                })

            # 🔧 FIX: Đảm bảo start luôn tiến về phía trước
            if end >= text_len:
                # Đã đến cuối văn bản
                break
            else:
                # Di chuyển start với overlap, nhưng đảm bảo luôn tiến
                new_start = end - overlap
                if new_start <= start:
                    # Nếu overlap quá lớn khiến không tiến, force di chuyển
                    new_start = start + max(1, max_size // 2)
                start = new_start

        return chunks

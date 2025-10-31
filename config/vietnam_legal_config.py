"""
Cấu hình đặc thù cho hệ thống tư vấn pháp luật Việt Nam
Vietnamese Legal Domain Configuration

File này chứa các cấu hình, văn bản giao diện và tham số đặc thù
cho việc áp dụng hệ thống vào lĩnh vực luật pháp Việt Nam.
"""

# ============================================================================
# CẤU HÌNH DOMAIN
# ============================================================================

LEGAL_DOMAIN = {
    "name": "Pháp luật Việt Nam",
    "description": "Hệ thống tư vấn và tra cứu văn bản pháp luật Việt Nam",
    "version": "1.0",
    "language": "vi"
}

# ============================================================================
# CẤU HÌNH CHUNKING CHO VĂN BẢN PHÁP LUẬT
# ============================================================================

LEGAL_CHUNKING_CONFIG = {
    # Văn bản pháp luật thường có cấu trúc rõ ràng: Điều > Khoản > Điểm
    # Nên chunk size lớn hơn để giữ nguyên맥 văn bản
    "chunk_size": 1200,  # Tăng từ 800 lên 1200 để giữ nguyên cấu trúc điều, khoản
    "chunk_overlap": 150,  # Tăng overlap để không mất맥 giữa các điều khoản

    # Các từ khóa để phát hiện ranh giới tự nhiên trong văn bản luật
    "legal_boundaries": [
        "Điều ",
        "Khoản ",
        "Điểm ",
        "Chương ",
        "Mục ",
        "Phần "
    ],

    # Ưu tiên tách theo cấu trúc pháp lý
    "split_by_structure": True
}

# ============================================================================
# TỪ KHÓA VÀ THUẬT NGỮ PHÁP LÝ
# ============================================================================

LEGAL_KEYWORDS = {
    # Loại văn bản
    "document_types": [
        "Luật", "Bộ luật", "Pháp lệnh", "Nghị định", "Nghị quyết",
        "Thông tư", "Quyết định", "Chỉ thị", "Quy định"
    ],

    # Cấu trúc văn bản
    "structure_keywords": [
        "Điều", "Khoản", "Điểm", "Chương", "Mục", "Phần"
    ],

    # Thuật ngữ pháp lý thường gặp
    "common_legal_terms": [
        "hiệu lực", "áp dụng", "quy định", "trách nhiệm", "quyền",
        "nghĩa vụ", "vi phạm", "xử phạt", "thẩm quyền", "thủ tục",
        "hồ sơ", "đơn", "tố cáo", "khiếu nại", "tranh chấp"
    ]
}

# ============================================================================
# BẢN ĐỊA HÓA GIAO DIỆN
# ============================================================================

UI_TEXTS = {
    # Header và tiêu đề
    "app_title": "⚖️ Hệ thống Tư vấn Pháp luật Việt Nam",
    "app_subtitle": "Tra cứu và tư vấn văn bản pháp luật bằng AI",
    "app_description": "Hệ thống AI giúp bạn tìm kiếm, phân tích và hiểu rõ các văn bản pháp luật Việt Nam",

    # Sidebar
    "sidebar_upload": "📤 Tải văn bản pháp luật",
    "sidebar_library": "📚 Thư viện văn bản",
    "sidebar_select": "Chọn văn bản để tra cứu",
    "sidebar_status": "🔧 Trạng thái hệ thống",

    # Upload
    "upload_instruction": "Chọn file văn bản pháp luật (PDF, TXT, MD)",
    "upload_button": "📄 Xử lý văn bản: {}",
    "upload_success": "✅ Đã xử lý thành công văn bản: {}",
    "upload_error": "❌ Lỗi khi xử lý văn bản: {}",
    "upload_duplicate": "⚠️ Văn bản này đã tồn tại trong hệ thống",

    # Document info
    "doc_type": "Loại văn bản",
    "doc_size": "Kích thước",
    "doc_chunks": "Số đoạn",
    "doc_status": "Trạng thái",
    "doc_uploaded": "Ngày tải lên",
    "doc_summary_btn": "Tạo tóm tắt",

    # Status
    "status_completed": "Hoàn thành",
    "status_processing": "Đang xử lý",
    "status_pending": "Chờ xử lý",
    "status_failed": "Thất bại",
    "status_duplicate": "Trùng lặp",

    # Chat
    "chat_header": "💬 Tư vấn pháp luật",
    "chat_placeholder": "Hỏi về văn bản pháp luật... (VD: 'Điều 10 quy định gì về quyền sở hữu?')",
    "chat_no_docs": "👈 Vui lòng chọn văn bản pháp luật từ thanh bên để bắt đầu",
    "chat_thinking": "🤔 Đang phân tích văn bản pháp luật...",
    "chat_no_history": "💭 Chưa có cuộc hội thoại. Hãy đặt câu hỏi về văn bản pháp luật!",

    # Response info
    "response_time": "⚡ Thời gian phản hồi",
    "sources_found": "📊 Nguồn tham chiếu",
    "documents_used": "📄 Văn bản sử dụng",
    "relevance": "Độ liên quan",

    # Actions
    "clear_history": "🗑️ Xóa lịch sử",
    "refresh_docs": "🔄 Làm mới danh sách",
    "generate_summary": "Tạo tóm tắt",

    # Getting started
    "getting_started_title": "🚀 Hướng dẫn sử dụng",
    "getting_started_steps": """
        1. **Tải văn bản**: Sử dụng thanh bên để tải file PDF, TXT hoặc Markdown chứa văn bản pháp luật
        2. **Chờ xử lý**: Hệ thống sẽ tự động phân tích và lưu trữ văn bản (trích xuất text + tạo embedding)
        3. **Chọn văn bản**: Chọn các văn bản pháp luật bạn muốn tra cứu
        4. **Bắt đầu hỏi**: Đặt câu hỏi về nội dung văn bản pháp luật!
    """,

    "tips_title": "💡 Mẹo sử dụng",
    "tips_content": """
        - Bạn có thể tra cứu nhiều văn bản pháp luật cùng lúc
        - Đặt câu hỏi cụ thể để có kết quả tốt nhất (VD: "Điều 5 quy định gì?")
        - AI sẽ trích dẫn nguồn từ các văn bản pháp luật của bạn
        - Hỏi về mối quan hệ giữa các văn bản (thay thế, sửa đổi, bổ sung)
    """,

    # System messages
    "system_initializing": "🔄 Đang khởi tạo hệ thống...",
    "system_init_success": "✅ Hệ thống đã sẵn sàng!",
    "system_init_error": "❌ Lỗi khởi tạo hệ thống",
    "system_not_initialized": "❌ Hệ thống chưa khởi tạo. Vui lòng kiểm tra logs và khởi động lại.",
}

# ============================================================================
# CÂU HỎI GỢI Ý THEO DOMAIN PHÁP LUẬT
# ============================================================================

SUGGESTED_QUESTIONS = {
    "general": [
        "Văn bản này quy định về vấn đề gì?",
        "Phạm vi điều chỉnh và đối tượng áp dụng là ai?",
        "Văn bản này có hiệu lực từ khi nào?",
        "Những quy định chính trong văn bản là gì?",
    ],

    "specific": [
        "Điều [số] quy định về vấn đề gì?",
        "Quyền và nghĩa vụ của [đối tượng] là gì?",
        "Trường hợp nào bị xử phạt theo văn bản này?",
        "Thủ tục để thực hiện [hành vi pháp lý] là gì?",
    ],

    "comparison": [
        "So sánh quy định của các văn bản về [vấn đề]",
        "Văn bản nào có hiệu lực pháp lý cao hơn?",
        "Văn bản mới thay đổi gì so với văn bản cũ?",
        "Có mâu thuẫn nào giữa các quy định không?",
    ],

    "practical": [
        "Tôi có quyền gì theo quy định này?",
        "Nghĩa vụ của tôi là gì?",
        "Làm thế nào để khiếu nại/khởi kiện?",
        "Hồ sơ cần chuẩn bị những gì?",
    ]
}

# ============================================================================
# CẤU HÌNH TRÍCH XUẤT THÔNG TIN PHÁP LÝ
# ============================================================================

LEGAL_EXTRACTION_CONFIG = {
    # Regex patterns để nhận diện cấu trúc văn bản luật
    "article_pattern": r"Điều\s+\d+",  # Điều 1, Điều 2...
    "clause_pattern": r"Khoản\s+\d+",  # Khoản 1, Khoản 2...
    "point_pattern": r"[Đ|đ]iểm\s+[a-z]",  # Điểm a, điểm b...
    "chapter_pattern": r"Chương\s+[IVXLCDM]+",  # Chương I, II, III...

    # Metadata cần trích xuất
    "metadata_fields": [
        "số_hiệu",  # VD: 68/2006/QH11
        "loại_văn_bản",  # Luật, Nghị định, Thông tư...
        "cơ_quan_ban_hành",
        "ngày_ban_hành",
        "ngày_hiệu_lực",
        "người_ký",
        "văn_bản_thay_thế",
        "văn_bản_được_sửa_đổi_bởi"
    ]
}

# ============================================================================
# CẤU HÌNH EMBEDDING VÀ RETRIEVAL
# ============================================================================

LEGAL_RETRIEVAL_CONFIG = {
    # Tăng top_k vì văn bản luật thường cần nhiều맥 văn bản hơn
    "top_k": 7,  # Tăng từ 5 lên 7

    # Threshold cho semantic search - có thể giảm để không bỏ sót quy định quan trọng
    "min_score": 0.25,  # Giảm từ 0.3 xuống 0.25

    # Hybrid search cho kết quả tốt hơn với thuật ngữ pháp lý
    "use_hybrid_search": True,
    "semantic_weight": 0.7,
    "keyword_weight": 0.3,

    # Reranking dựa trên cấu trúc pháp lý
    "boost_structured_content": True,  # Ưu tiên các đoạn có cấu trúc Điều, Khoản
    "boost_multiplier": 1.2
}

# ============================================================================
# PROMPT TEMPLATES ĐẶC THÙ CHO DOMAIN PHÁP LUẬT
# ============================================================================

LEGAL_PROMPT_TEMPLATES = {
    # Template cho việc trích dẫn chính xác
    "citation_format": """
Khi trích dẫn, vui lòng sử dụng định dạng:
- Theo [Tên văn bản], [Điều X], [Khoản Y]: "[Nội dung]"
- VD: Theo Bộ luật Dân sự 2015, Điều 1, Khoản 1: "Bộ luật này quy định..."
""",

    # Template cho câu trả lời có cấu trúc
    "structured_answer": """
Vui lòng trả lời theo cấu trúc:

1. **Quy định pháp luật**: Trích dẫn chính xác điều, khoản liên quan
2. **Giải thích**: Phân tích ý nghĩa và phạm vi áp dụng
3. **Lưu ý**: Các điều kiện, ngoại lệ hoặc quy định liên quan khác
""",

    # Template cho so sánh văn bản
    "comparison_template": """
Khi so sánh văn bản pháp luật, hãy phân tích:

1. **Hiệu lực pháp lý**: Văn bản nào có hiệu lực cao hơn (Luật > Nghị định > Thông tư)
2. **Thời gian**: Văn bản nào mới hơn, có thay thế/sửa đổi không
3. **Nội dung**: Điểm giống và khác về quy định
4. **Áp dụng**: Trong tình huống cụ thể nên áp dụng văn bản nào
"""
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_ui_text(key: str, default: str = "") -> str:
    """Lấy văn bản giao diện theo key"""
    return UI_TEXTS.get(key, default)

def get_suggested_questions(category: str = "general") -> list:
    """Lấy danh sách câu hỏi gợi ý theo category"""
    return SUGGESTED_QUESTIONS.get(category, [])

def get_legal_config() -> dict:
    """Lấy toàn bộ cấu hình pháp luật"""
    return {
        "domain": LEGAL_DOMAIN,
        "chunking": LEGAL_CHUNKING_CONFIG,
        "keywords": LEGAL_KEYWORDS,
        "extraction": LEGAL_EXTRACTION_CONFIG,
        "retrieval": LEGAL_RETRIEVAL_CONFIG,
        "prompts": LEGAL_PROMPT_TEMPLATES
    }

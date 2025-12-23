
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

LEGAL_SYSTEM_PROMPT = """
Bạn là trợ lý AI về pháp luật Việt Nam.

- Chỉ dùng nội dung trong phần "Thông tin từ văn bản pháp luật" (context).
- Không bịa hoặc suy đoán Điều, Khoản, Điểm, tên/số/năm văn bản nếu context không có.
- Khi trích dẫn, nêu rõ tên văn bản, số/ký hiệu, Điều/Khoản/Điểm (nếu có).
- Nếu có luật gốc và luật sửa đổi, hiểu nội dung hiện hành là sau sửa đổi và nêu rõ luật gốc + luật sửa đổi khi cần.
- Nếu context có thông tin về hiệu lực, phạm vi, đối tượng áp dụng và chúng liên quan, hãy nêu rõ.
- Giải thích bằng tiếng Việt dễ hiểu, trả lời có cấu trúc, đi thẳng trọng tâm.
- Nếu context không đủ để trả lời, nói rõ hạn chế, không bịa thêm.
- Câu trả lời chỉ mang tính tham khảo, không thay thế tư vấn pháp lý chính thức.
- Không hiển thị quá trình suy luận nội bộ (thinking/chain-of-thought); chỉ đưa ra câu trả lời cuối cùng.

Luôn trung thành với nội dung trong context.
"""



LEGAL_USER_PROMPT_TEMPLATE = """
Thông tin từ văn bản pháp luật:
{context}

Câu hỏi của người dùng: {question}

Yêu cầu:
- Chỉ dùng thông tin trong phần "Thông tin từ văn bản pháp luật", không dùng kiến thức bên ngoài.
- Nếu có nhiều văn bản, chỉ dùng những văn bản liên quan và nêu rõ nội dung thuộc văn bản nào (tên, số, ký hiệu).
- Nếu có luật sửa đổi, bổ sung, hiểu đó là nội dung cập nhật của luật gốc và nêu rõ luật gốc + luật sửa đổi khi cần.
- Trích dẫn càng chính xác càng tốt: tên văn bản, số/ký hiệu, Điều/Khoản/Điểm (nếu có).
- Giải thích rõ ràng, dễ hiểu, ưu tiên đúng trọng tâm.
- Nếu context chưa đủ, hãy nêu rõ phần còn thiếu và không suy đoán.

Trả lời:
"""




# ============================================================================
# STRUCTURED ANSWER PROMPTS
# ============================================================================

LEGAL_STRUCTURED_ANSWER_TEMPLATE = """
Thông tin từ văn bản pháp luật:
{context}

Câu hỏi: {question}

Chỉ dùng thông tin trong phần "Thông tin từ văn bản pháp luật", không dùng kiến thức bên ngoài.

Trả lời theo cấu trúc:

**1. QUY ĐỊNH PHÁP LUẬT**
- Liệt kê quy định liên quan:
  + Tên văn bản, số/ký hiệu.
  + Điều/Khoản/Điểm và tóm lược nội dung chính.
- Nếu có luật sửa đổi, bổ sung, nêu rõ luật gốc + luật sửa đổi và nội dung sau sửa đổi.

**2. GIẢI THÍCH**
- Phân tích ý nghĩa, phạm vi, đối tượng áp dụng.
- Nếu liên quan thời điểm áp dụng, làm rõ trước/sau khi sửa đổi (nếu context cho phép).
- Có thể kèm 1–2 ví dụ minh họa ngắn.

**3. LƯU Ý**
- Nêu điều kiện, ngoại lệ, hoặc quy định liên quan khác (nếu có).
- Nếu context chưa đủ, nêu rõ còn thiếu gì/cần thêm văn bản nào.
- Nhắc lại: Câu trả lời chỉ mang tính tham khảo, không thay thế tư vấn pháp lý chính thức.

Trả lời:
"""




# ============================================================================
# LEGAL SUMMARY PROMPTS
# ============================================================================

LEGAL_SUMMARY_TEMPLATE = """Hãy tóm tắt văn bản pháp luật sau đây.

Nội dung tóm tắt cần bao gồm:
1. **Thông tin cơ bản**: Tên văn bản, số hiệu, cơ quan ban hành, ngày ban hành
2. **Phạm vi điều chỉnh**: Văn bản này quy định về vấn đề gì
3. **Đối tượng áp dụng**: Áp dụng cho ai/tổ chức nào
4. **Nội dung chính**: Các quy định quan trọng nhất
5. **Hiệu lực**: Thời điểm có hiệu lực và các quy định liên quan

Nội dung văn bản:
{content}

Tóm tắt:"""

# ============================================================================
# LEGAL COMPARISON PROMPTS
# ============================================================================

LEGAL_COMPARISON_TEMPLATE = """So sánh hai văn bản pháp luật sau:

**Văn bản 1**: {doc1_title}
{doc1_content}

**Văn bản 2**: {doc2_title}
{doc2_content}

Vui lòng phân tích theo các khía cạnh:

**1. HIỆU LỰC PHÁP LÝ**
- Văn bản nào có thứ bậc cao hơn (Luật > Nghị định > Thông tư)
- Thời điểm ban hành và hiệu lực

**2. PHẠM VI ĐIỀU CHỈNH**
- Nội dung điều chỉnh của mỗi văn bản
- Đối tượng áp dụng

**3. NỘI DUNG QUY ĐỊNH**
- Điểm giống nhau
- Điểm khác biệt
- Quy định bổ sung/thay thế

**4. ÁP DỤNG TRONG THỰC TẾ**
- Văn bản nào ưu tiên áp dụng trong các trường hợp cụ thể
- Mối quan hệ giữa hai văn bản (thay thế, sửa đổi, bổ sung)

Phân tích so sánh:"""

# ============================================================================
# ARTICLE EXPLANATION PROMPTS
# ============================================================================

ARTICLE_EXPLANATION_TEMPLATE = """Dựa vào ngữ cảnh từ văn bản pháp luật:
{context}

Hãy giải thích chi tiết về Điều {article_number} (nếu có Khoản {clause_number}, Điểm {point}).

Nội dung giải thích cần bao gồm:
1. **Nội dung quy định**: Trích dẫn đầy đủ nội dung Điều/Khoản/Điểm
2. **Ý nghĩa**: Giải thích mục đích, ý nghĩa của quy định
3. **Đối tượng áp dụng**: Áp dụng cho ai/trường hợp nào
4. **Điều kiện**: Các điều kiện cần có (nếu có)
5. **Quy định liên quan**: Các Điều khác có liên quan
6. **Ví dụ minh họa**: Ví dụ cụ thể về cách áp dụng (nếu có thể)

Giải thích:"""

# ============================================================================
# LEGAL CITATION PROMPTS
# ============================================================================

LEGAL_CITATION_TEMPLATE = """Dựa vào văn bản pháp luật sau:
{context}

Hãy tạo trích dẫn chuẩn cho quy định liên quan đến: {topic}

Format trích dẫn chuẩn:
"Theo [Loại văn bản] [Tên văn bản] số [Số hiệu], ban hành ngày [Ngày/Tháng/Năm], [Điều X], [Khoản Y], [Điểm z]: '[Nội dung quy định]'"

Ví dụ:
"Theo Bộ luật Dân sự số 91/2015/QH13, ban hành ngày 24/11/2015, Điều 1, Khoản 1: 'Bộ luật này quy định về các quan hệ dân sự...'"

Trích dẫn:"""

# ============================================================================
# RIGHTS AND OBLIGATIONS PROMPTS
# ============================================================================

RIGHTS_OBLIGATIONS_TEMPLATE = """Từ văn bản pháp luật:
{context}

Hãy phân tích quyền và nghĩa vụ của {subject} theo quy định.

Cấu trúc trả lời:

**1. QUYỀN**
Liệt kê các quyền được quy định, trích dẫn cụ thể Điều, Khoản

**2. NGHĨA VỤ**
Liệt kê các nghĩa vụ phải thực hiện, trích dẫn cụ thể Điều, Khoản

**3. CHẾ TÀI VI PHẠM**
Hậu quả khi không thực hiện nghĩa vụ hoặc vi phạm quy định

**4. ĐIỀU KIỆN THỰC HIỆN**
Các điều kiện, thủ tục cần thiết để thực hiện quyền hoặc nghĩa vụ

Phân tích:"""

# ============================================================================
# PROCEDURE EXPLANATION PROMPTS
# ============================================================================

PROCEDURE_TEMPLATE = """Dựa vào văn bản pháp luật:
{context}

Hãy hướng dẫn chi tiết thủ tục: {procedure_name}

Nội dung hướng dẫn:

**1. CĂN CỨ PHÁP LÝ**
Trích dẫn các Điều, Khoản quy định về thủ tục này

**2. ĐỐI TƯỢNG**
Ai có th���/phải thực hiện thủ tục này

**3. HỒ SƠ YÊU CẦU**
Danh sách các giấy tờ, tài liệu cần chuẩn bị

**4. CÁC BƯỚC THỰC HIỆN**
Quy trình từng bước thực hiện thủ tục

**5. THỜI HẠN**
Thời hạn nộp hồ sơ, thời gian xử lý, thời hạn giải quyết

**6. CƠ QUAN TIẾP NHẬN**
Cơ quan/tổ chức có thẩm quyền tiếp nhận và giải quyết

**7. PHÍ VÀ LỆ PHÍ**
Chi phí cần nộp (nếu có)

Hướng dẫn:"""

# ============================================================================
# VIOLATION AND PENALTY PROMPTS
# ============================================================================

VIOLATION_PENALTY_TEMPLATE = """Từ văn bản pháp luật:
{context}

Phân tích về hành vi vi phạm: {violation_description}

Nội dung phân tích:

**1. ĐỊNH TÍNH VI PHẠM**
Hành vi này có cấu thành vi phạm pháp luật không? Căn cứ trên Điều, Khoản nào?

**2. HÌNH THỨC XỬ PHẠT**
- Mức phạt tiền (nếu có)
- Hình thức phạt bổ sung
- Biện pháp khắc phục hậu quả

**3. THẨM QUYỀN XỬ PHẠT**
Cơ quan/cá nhân nào có thẩm quyền xử phạt

**4. THỜI HIỆU XỬ PHẠT**
Thời hạn được xử phạt vi phạm hành chính

**5. QUYỀN KHIẾU NẠI**
Quyền và thủ tục khiếu nại quyết định xử phạt

Phân tích:"""

# ============================================================================
# LangChain PromptTemplates
# ============================================================================

# Legal Q&A Chat Prompt
legal_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", LEGAL_SYSTEM_PROMPT),
    ("human", LEGAL_USER_PROMPT_TEMPLATE)
])

# Structured Legal Answer Prompt
legal_structured_prompt = ChatPromptTemplate.from_messages([
    ("system", LEGAL_SYSTEM_PROMPT),
    ("human", LEGAL_STRUCTURED_ANSWER_TEMPLATE)
])

# Legal Summary Prompt
legal_summary_prompt = PromptTemplate(
    input_variables=["content"],
    template=LEGAL_SUMMARY_TEMPLATE
)

# Legal Comparison Prompt
legal_comparison_prompt = PromptTemplate(
    input_variables=["doc1_title", "doc1_content", "doc2_title", "doc2_content"],
    template=LEGAL_COMPARISON_TEMPLATE
)

# Article Explanation Prompt
article_explanation_prompt = PromptTemplate(
    input_variables=["context", "article_number"],
    template=ARTICLE_EXPLANATION_TEMPLATE,
    partial_variables={"clause_number": "", "point": ""}
)

# Legal Citation Prompt
legal_citation_prompt = PromptTemplate(
    input_variables=["context", "topic"],
    template=LEGAL_CITATION_TEMPLATE
)

# Rights and Obligations Prompt
rights_obligations_prompt = PromptTemplate(
    input_variables=["context", "subject"],
    template=RIGHTS_OBLIGATIONS_TEMPLATE
)

# Procedure Explanation Prompt
procedure_prompt = PromptTemplate(
    input_variables=["context", "procedure_name"],
    template=PROCEDURE_TEMPLATE
)

# Violation and Penalty Prompt
violation_penalty_prompt = PromptTemplate(
    input_variables=["context", "violation_description"],
    template=VIOLATION_PENALTY_TEMPLATE
)

# ============================================================================
# Helper Functions
# ============================================================================

def build_legal_qa_prompt(question: str, context: str) -> str:
    """Build legal Q&A prompt for Vietnamese legal documents"""
    return legal_qa_prompt.format(context=context, question=question)


def build_legal_structured_prompt(question: str, context: str) -> str:
    """Build structured legal answer prompt"""
    return legal_structured_prompt.format(context=context, question=question)


def build_legal_summary(content: str) -> str:
    """Build legal document summary prompt"""
    return legal_summary_prompt.format(content=content)


def build_legal_comparison(doc1_title: str, doc1_content: str,
                          doc2_title: str, doc2_content: str) -> str:
    """Build legal document comparison prompt"""
    return legal_comparison_prompt.format(
        doc1_title=doc1_title,
        doc1_content=doc1_content,
        doc2_title=doc2_title,
        doc2_content=doc2_content
    )


def build_article_explanation(context: str, article_number: int,
                             clause_number: int = None, point: str = None) -> str:
    """Build article explanation prompt"""
    return article_explanation_prompt.format(
        context=context,
        article_number=article_number,
        clause_number=clause_number or "",
        point=point or ""
    )


def build_rights_obligations(context: str, subject: str) -> str:
    """Build rights and obligations analysis prompt"""
    return rights_obligations_prompt.format(context=context, subject=subject)


def build_procedure_guide(context: str, procedure_name: str) -> str:
    """Build procedure explanation prompt"""
    return procedure_prompt.format(context=context, procedure_name=procedure_name)


def build_violation_analysis(context: str, violation_description: str) -> str:
    """Build violation and penalty analysis prompt"""
    return violation_penalty_prompt.format(
        context=context,
        violation_description=violation_description
    )

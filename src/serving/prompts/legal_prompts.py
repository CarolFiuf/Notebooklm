
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

LEGAL_SYSTEM_PROMPT = """
Bạn là trợ lý AI chuyên về pháp luật Việt Nam, giúp người dùng tra cứu và hiểu rõ các văn bản pháp luật.

Nguyên tắc làm việc:

1. Chỉ dựa trên văn bản được cung cấp
- Chỉ sử dụng thông tin xuất hiện trong phần "Thông tin từ văn bản pháp luật" do hệ thống cung cấp.
- Không được suy đoán, không tự bịa thêm Điều, Khoản, Điểm, tên luật, số luật, năm ban hành nếu context không có.

2. Trích dẫn đầy đủ, rõ ràng
- Khi trích dẫn, ghi rõ: tên văn bản (ví dụ: Luật Đường sắt), số, ký hiệu (ví dụ: 95/2025/QH15), và Điều, Khoản, Điểm liên quan.
- Nếu trả lời dựa trên Luật sửa đổi, bổ sung (ví dụ: Luật sửa đổi, bổ sung một số điều của Luật Mặt trận Tổ quốc Việt Nam...), hãy nêu rõ luật nào đang được sửa đổi và trích dẫn theo phiên bản đã được sửa đổi.

3. Xử lý nhiều văn bản và Luật sửa đổi/bổ sung
- Nếu context có nhiều văn bản khác nhau, chỉ sử dụng những văn bản thực sự liên quan đến câu hỏi.
- Nếu context có cả luật gốc và luật sửa đổi, bổ sung:
  + Hiểu rằng nội dung hiện hành là nội dung đã được sửa đổi, bổ sung.
  + Khi trình bày, nêu rõ luật gốc và luật sửa đổi (ví dụ: “Luật Công đoàn số 50/2024/QH15, được sửa đổi, bổ sung bởi Luật số 97/2025/QH15, quy định rằng…”).
- Nếu trong context chỉ có đoạn Luật sửa đổi, bổ sung (kiểu “Sửa đổi Điều 1 như sau: …”), hãy trả lời theo nội dung mới đó và làm rõ đây là nội dung sau sửa đổi của luật gốc.

4. Thời điểm hiệu lực và phạm vi áp dụng
- Nếu trong context có Điều khoản thi hành, thời điểm hiệu lực, đối tượng, phạm vi áp dụng, hãy nêu rõ khi liên quan đến câu hỏi.
- Nếu câu hỏi liên quan “hiện nay”, “tại thời điểm luật có hiệu lực” mà context có nhiều mốc thời gian, hãy giải thích rõ giai đoạn áp dụng (trước hay sau khi sửa đổi).

5. Ngôn ngữ, cấu trúc, phong cách
- Sử dụng thuật ngữ pháp lý chính xác nhưng giải thích bằng ngôn ngữ dễ hiểu cho người không chuyên.
- Trả lời có cấu trúc, rõ ràng, ưu tiên dùng gạch đầu dòng, mục nhỏ.
- Đi thẳng vào trọng tâm câu hỏi trước, sau đó mới phân tích chi tiết nếu cần.

6. Khách quan và giới hạn
- Không đưa ra ý kiến cá nhân, không suy đoán ý chí của cơ quan nhà nước.
- Nếu thông tin trong context không đủ để trả lời đầy đủ, phải nói rõ phần nào không đủ, và tuyệt đối không bịa thêm.
- Câu trả lời chỉ mang tính tham khảo, không thay thế ý kiến tư vấn pháp lý chính thức của luật sư hoặc cơ quan nhà nước có thẩm quyền.

7. Không hiển thị nội dung suy luận nội bộ (thinking content)
- Bạn có thể suy luận nhiều bước ở bên trong để tìm câu trả lời chính xác.
- Tuyệt đối KHÔNG được hiển thị bất kỳ phần nào mô tả quá trình suy nghĩ nội bộ như:
  + "Suy nghĩ:", "Phân tích:", "Reasoning:", "Chain-of-thought:", "Thought:", v.v.
  + Các bước liệt kê kiểu "Bước 1: ...", "Bước 2: ..." chỉ để mô tả quá trình bạn đang suy nghĩ.
  + Các cụm như "hãy cùng phân tích", "let's think step by step", "let's think", "I will think step by step", v.v.
- Chỉ xuất ra phần **kết quả cuối cùng** theo đúng cấu trúc đã được yêu cầu (Quy định pháp luật, Giải thích, Lưu ý...).
- Không được nhắc đến việc bạn là mô hình AI đang suy nghĩ hay mô tả cơ chế suy luận của mình.

Luôn cung cấp thông tin hữu ích, chính xác, dễ hiểu và trung thành với nội dung văn bản được cung cấp.
"""


LEGAL_USER_PROMPT_TEMPLATE = """
Thông tin từ văn bản pháp luật:
{context}

Câu hỏi của người dùng: {question}

Yêu cầu:
1. Chỉ sử dụng thông tin trong phần "Thông tin từ văn bản pháp luật" ở trên, không dùng kiến thức bên ngoài.
2. Nếu có nhiều văn bản khác nhau trong context:
   - Chỉ chọn những văn bản thực sự liên quan đến câu hỏi.
   - Nêu rõ nội dung thuộc về văn bản nào (tên luật, số, ký hiệu).
3. Nếu context có Luật sửa đổi, bổ sung một số điều của luật khác:
   - Phải hiểu đây là nội dung cập nhật của luật gốc.
   - Khi trả lời, trích dẫn theo phiên bản đã được sửa đổi (nêu rõ luật gốc và luật sửa đổi).
4. Trích dẫn chính xác và đầy đủ: tên văn bản, số, ký hiệu, Điều, Khoản, Điểm (nếu có).
5. Giải thích rõ ràng, dễ hiểu, ưu tiên trả lời đúng trọng tâm câu hỏi.
6. Nếu thông tin trong context chưa đủ để trả lời đầy đủ, hãy nêu rõ phần còn thiếu và không suy đoán.

Trả lời:
"""


# ============================================================================
# STRUCTURED ANSWER PROMPTS
# ============================================================================

LEGAL_STRUCTURED_ANSWER_TEMPLATE = """
Thông tin từ văn bản pháp luật:
{context}

Câu hỏi: {question}

Chỉ sử dụng thông tin trong phần "Thông tin từ văn bản pháp luật" ở trên, không dùng kiến thức bên ngoài.

Vui lòng trả lời theo cấu trúc sau:

**1. QUY ĐỊNH PHÁP LUẬT**
- Liệt kê các quy định liên quan theo thứ tự:
  + Tên văn bản, số, ký hiệu (ví dụ: Luật Đường sắt số 95/2025/QH15).
  + Điều, Khoản, Điểm và trích dẫn nội dung chính (có thể tóm lược, không cần chép nguyên văn quá dài).
- Nếu có luật sửa đổi, bổ sung:
  + Nêu rõ luật gốc và luật sửa đổi (ví dụ: “Luật Mặt trận Tổ quốc Việt Nam số 75/2015/QH13, được sửa đổi, bổ sung bởi Luật số 97/2025/QH15, Điều …”).
  + Trình bày nội dung sau khi đã được sửa đổi.

**2. GIẢI THÍCH**
- Phân tích ý nghĩa quy định, phạm vi áp dụng, đối tượng áp dụng.
- Làm rõ sự khác nhau (nếu có) giữa trước và sau khi sửa đổi, nếu câu hỏi liên quan đến thời điểm áp dụng.
- Có thể kèm 1–2 ví dụ minh họa ngắn, miễn không trái với tinh thần quy định.

**3. LƯU Ý**
- Nêu các điều kiện, ngoại lệ, quy định liên quan khác trong context (nếu có).
- Nếu thông tin trong context chưa đủ để trả lời trọn vẹn, hãy nêu rõ: còn thiếu điều gì, cần thêm văn bản nào khác.
- Nhắc lại ngắn gọn: Câu trả lời chỉ mang tính tham khảo, không thay thế tư vấn pháp lý chính thức.

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
    input_variables=["context", "article_number", "clause_number", "point"],
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

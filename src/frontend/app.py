import streamlit as st
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import time

# Configure logging
from src.utils.logging_config import setup_logging
setup_logging()

# Import application components
from config.settings import settings
from src.utils.database import init_database, get_db_session, Document
from src.utils.file_utils import save_uploaded_file
from src.processing.document_processor import process_and_save_document
from src.rag.rag_engine import initialize_rag_system
from src.utils.exceptions import NotebookLMError
from src.rag.embedding_pipeline import process_document_embeddings

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="NotebookLM",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "Legal RAG - AI-powered document chat system"
    }
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        background-color: #f8f9fa;
    }
    .source-info {
        background-color: #e9ecef;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        font-size: 0.9rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.75rem;
        border-radius: 0.25rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'rag_engine' not in st.session_state:
        st.session_state.rag_engine = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'system_initialized' not in st.session_state:
        st.session_state.system_initialized = False
    if 'selected_documents' not in st.session_state:
        st.session_state.selected_documents = []
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {}

@st.cache_resource
def initialize_system():
    """Initialize the RAG system with caching"""
    try:
        with st.spinner("🔄 Initializing system components..."):
            logger.info("Starting system initialization")
            
            # Initialize database
            init_database()
            logger.info("✅ Database initialized")
            
            # Initialize RAG engine
            rag_engine = initialize_rag_system()
            logger.info("✅ RAG engine initialized")
            
            return rag_engine, True
            
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        st.error(f"❌ System initialization failed: {str(e)}")
        return None, False

def display_header():
    """Display application header"""
    st.markdown('<h1 class="main-header">Legal RAG</h1>', unsafe_allow_html=True)
    st.markdown("---")

def handle_file_upload():
    """Handle file upload in sidebar"""
    st.sidebar.subheader("Upload Documents")
    
    uploaded_files = st.sidebar.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'md', 'docx', 'doc'],
        help="Upload PDF, TXT, Markdown, or Word files (max 100MB each)",
        key="file_uploader"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_key = f"process_{uploaded_file.name}_{uploaded_file.size}"
            
            if st.sidebar.button(f"📄 Process {uploaded_file.name}", key=file_key):
                process_uploaded_file(uploaded_file)

def process_uploaded_file(uploaded_file):
    """Process a single uploaded file with duplicate handling"""
    try:
        with st.sidebar:
            progress_bar = st.progress(0, text="Starting processing...")
            status_text = st.empty()
            
            # Step 1: Save file
            status_text.text("Saving file...")
            progress_bar.progress(20, text="Saving file...")
            
            file_path, unique_filename = save_uploaded_file(uploaded_file)
            logger.info(f"File saved: {unique_filename}")
            
            # Step 2: Process document (with duplicate check)
            status_text.text("Processing document...")
            progress_bar.progress(50, text="Processing document...")
            
            document_id = process_and_save_document(file_path, uploaded_file.name)
            logger.info(f"Document processed: ID {document_id}")
            
            # Step 3: Check if this was a duplicate
            db = get_db_session()
            try:
                doc = db.query(Document).filter(Document.id == document_id).first()
                # Check document_metadata dict for 'already_exists' key
                is_duplicate = doc and doc.document_metadata and doc.document_metadata.get('already_exists', False)
            except Exception as e:
                logger.error(f"Error checking duplicate status for document {document_id}: {e}")
                is_duplicate = False
            finally:
                if db:
                    db.close()
            
            if is_duplicate:
                progress_bar.progress(100, text="Document already exists!")
                status_text.markdown(
                    f'<div class="success-message">Document "{uploaded_file.name}" already exists in the system. Skipping duplicate processing.</div>',
                    unsafe_allow_html=True
                )
                st.session_state.processing_status[document_id] = "duplicate"
            else:
                # Step 3: Generate embeddings for new documents
                status_text.text("Generating embeddings...")
                progress_bar.progress(80, text="Generating embeddings...")
                
                embedding_result = process_document_embeddings(document_id)
                
                if embedding_result.get('success'):
                    progress_bar.progress(100, text="Processing completed!")
                    status_text.markdown(
                        f'<div class="success-message">Successfully processed {uploaded_file.name}!</div>',
                        unsafe_allow_html=True
                    )
                    st.session_state.processing_status[document_id] = "completed"
                else:
                    error_msg = embedding_result.get('error', 'Unknown error')
                    status_text.markdown(
                        f'<div class="error-message">Processing failed: {error_msg}</div>',
                        unsafe_allow_html=True
                    )
            
            # 🔧 FIX: Clear document cache before refresh to show new document
            load_documents_from_db.clear()

            # Auto-refresh to show document status
            time.sleep(1)
            st.rerun()
                
    except Exception as e:
        logger.error(f"Error processing file {uploaded_file.name}: {e}")
        
        # Handle specific duplicate error messages
        if "duplicate key value violates unique constraint" in str(e):
            st.sidebar.info(f"Document {uploaded_file.name} already exists in the system.")
        else:
            st.sidebar.error(f"Error processing {uploaded_file.name}: {str(e)}")

@st.cache_data(ttl=100, show_spinner=False)  
def load_documents_from_db() -> List[Dict[str, Any]]:
    """
    🔧 FIXED: Load documents from database with optimized caching

    Cache for 5 minutes to significantly reduce DB load.
    Users can manually refresh using the "🔄 Refresh Documents" button.
    """
    db = None
    try:
        db = get_db_session()
        documents = db.query(Document).order_by(Document.upload_date.desc()).all()

        doc_list = []
        for doc in documents:
            doc_dict = {
                'id': doc.id,
                'filename': doc.original_filename,
                'file_type': doc.file_type,
                'file_size': doc.file_size,
                'upload_date': doc.upload_date,
                'processing_status': doc.processing_status,
                'total_chunks': doc.total_chunks,
                'summary': doc.summary,
                'document_metadata': doc.document_metadata or {}  # Include metadata for legal documents
            }
            doc_list.append(doc_dict)

        logger.debug(f"Loaded {len(doc_list)} documents from database (cached for 5min)")
        return doc_list

    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return []
    finally:
        # ✅ Always close DB session to prevent connection leaks
        if db:
            db.close()

def display_document_library():
    """Display document library in sidebar with duplicate status"""
    st.sidebar.subheader("Document Library")
    
    documents = load_documents_from_db()
    
    if not documents:
        st.sidebar.info("No documents uploaded yet. Upload some documents to get started!")
        return []
    
    # Document selection for chat
    st.sidebar.subheader("Select Documents for Chat")
    
    # Create options for multiselect with status indicators
    doc_options = {}
    for doc in documents:
        status_emoji = get_status_emoji(doc['processing_status'])

        # Add legal document indicator
        legal_metadata = doc.get('document_metadata', {}).get('legal_metadata', {})
        legal_indicator = ""
        if legal_metadata and legal_metadata.get('document_type'):
            legal_indicator = "⚖️ "  # Legal document indicator

        display_name = f"{status_emoji} {legal_indicator}{doc['filename'][:30]}{'...' if len(doc['filename']) > 30 else ''}"
        doc_options[display_name] = doc['id']
    
    # Multi-select for documents
    selected_display_names = st.sidebar.multiselect(
        "Choose documents to chat with:",
        options=list(doc_options.keys()),
        default=[],
        help="Select one or more documents to include in your chat context"
    )
    
    selected_doc_ids = [doc_options[name] for name in selected_display_names]
    
    # Display document details with duplicate info
    st.sidebar.subheader("Document Details")

    for doc in documents[:5]:  # Show first 5 documents
        with st.sidebar.expander(f"{doc['filename'][:25]}..."):
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Type:**", doc['file_type'])
                st.write("**Size:**", f"{doc['file_size'] / (1024*1024):.1f} MB")

            with col2:
                st.write("**Chunks:**", doc['total_chunks'])
                st.write("**Status:**", format_status(doc['processing_status']))

            st.write("**Uploaded:**", doc['upload_date'].strftime('%Y-%m-%d %H:%M'))

            # Show legal metadata if available
            legal_metadata = doc.get('document_metadata', {}).get('legal_metadata', {})
            if legal_metadata and legal_metadata.get('document_type'):
                st.markdown("---")
                st.markdown("**⚖️ Thông Tin Pháp Lý:**")

                # Document type with hierarchy badge
                doc_type = legal_metadata.get('document_type')
                hierarchy = legal_metadata.get('hierarchy_level')
                if doc_type:
                    if hierarchy:
                        st.write(f"📋 **Loại:** {doc_type} (Cấp {hierarchy})")
                    else:
                        st.write(f"📋 **Loại:** {doc_type}")

                # Document number
                doc_num = legal_metadata.get('document_number')
                if doc_num:
                    st.write(f"🔢 **Số hiệu:** {doc_num}")

                # Issue date
                issue_date = legal_metadata.get('issue_date')
                if issue_date:
                    st.write(f"📅 **Ban hành:** {issue_date}")

                # Effective date
                effective_date = legal_metadata.get('effective_date')
                if effective_date:
                    st.write(f"✅ **Hiệu lực:** {effective_date}")

                # Issuing authority
                authority = legal_metadata.get('issuing_authority')
                if authority:
                    st.write(f"🏛️ **Cơ quan:** {authority}")

                # Related documents
                replaces = legal_metadata.get('replaces', [])
                if replaces:
                    st.write(f"🔄 **Thay thế:** {', '.join(replaces[:2])}")

                amended_by = legal_metadata.get('amended_by', [])
                if amended_by:
                    st.write(f"📝 **Sửa đổi bởi:** {', '.join(amended_by[:2])}")

            # Show duplicate warning if applicable
            processing_status = st.session_state.processing_status.get(doc['id'])
            if processing_status == "duplicate":
                st.warning("This document is a duplicate of an existing file.")

            # Generate summary button
            if st.button(f"Generate Summary", key=f"summary_{doc['id']}"):
                generate_document_summary(doc['id'])
    
    if len(documents) > 5:
        st.sidebar.info(f"Showing 5 of {len(documents)} documents")
    
    return selected_doc_ids

def get_status_emoji(status):
    """Get emoji for document processing status"""
    status_map = {
        'completed': '✅',
        'processing': '⏳',
        'pending': '📄',
        'failed': '❌',
        'duplicate': '🔄'
    }
    return status_map.get(status, '📄')

def format_status(status):
    """Format status for display"""
    status_map = {
        'completed': 'Completed',
        'processing': 'Processing',
        'pending': 'Pending',
        'failed': 'Failed',
        'duplicate': 'Duplicate'
    }
    return status_map.get(status, status.title())

def generate_document_summary(document_id: int):
    """Generate and display document summary"""
    if not st.session_state.rag_engine:
        st.sidebar.error("❌ System not initialized")
        return
    
    try:
        with st.sidebar:
            with st.spinner("🔄 Generating summary..."):
                summary = st.session_state.rag_engine.get_document_summary(document_id)
                
                if summary:
                    st.success("✅ Summary generated!")
                    st.write("**Summary:**")
                    st.write(summary)
                else:
                    st.warning("⚠️ Could not generate summary")
                    
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        st.sidebar.error(f"❌ Error generating summary: {str(e)}")

def display_chat_interface(selected_doc_ids: List[int]):
    """Display main chat interface with dual modes: RAG (with docs) and Normal Chat (without docs)"""
    if not st.session_state.rag_engine:
        st.error("❌ System not initialized. Please check the logs and restart.")
        return

    # Mode indicator and chat header
    if not selected_doc_ids:
        # Normal Chat Mode (no documents)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Normal Chat Mode")
        with col2:
            st.info("Direct LLM")

        st.markdown("""
        **Chatting directly with Qwen3-8B LLM** (no document context)

        💡 **Tip**: Upload and select documents to enable RAG mode for document-based Q&A
        """)
    else:
        # RAG Mode (with documents)
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Chat with Your Documents")
        with col2:
            st.success(f"RAG Mode ({len(selected_doc_ids)} docs)")

    # Show getting started info only if no conversation history and no documents
    if not selected_doc_ids and not st.session_state.conversation_history:
        with st.expander("🚀 Getting Started", expanded=True):
            st.markdown("""
            ### RAG Mode (Recommended for document Q&A)

            1. **Upload Documents**: Use the sidebar to upload PDF, TXT, DOCX, or Markdown files
            2. **Wait for Processing**: Files will be processed automatically (text extraction + embeddings)
            3. **Select Documents**: Choose which documents you want to chat with
            4. **Start Chatting**: Ask questions about your documents!

            ### Normal Chat Mode (Current)

            - Chat directly with the AI without document context
            - Good for general questions and conversations
            - No sources or citations will be provided

            ### 💡 Tips
            - You can switch between modes by selecting/deselecting documents
            - RAG mode provides sources and citations from your documents
            - Ask specific questions for better results in RAG mode
            """)

    # Show selected documents info (RAG mode only)
    if selected_doc_ids:
        # Display selected documents info
        with st.expander("Selected Documents", expanded=False):
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                st.write(f"**Chatting with {len(selected_doc_ids)} document(s)**")

            with col2:
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.conversation_history = []
                    st.rerun()

            with col3:
                if st.button("🔄 Refresh Documents"):
                    # 🔧 FIX: Clear cache before rerunning
                    load_documents_from_db.clear()
                    st.rerun()
    
    # Display conversation history
    display_conversation_history()
    
    # Chat input
    handle_chat_input(selected_doc_ids)

def display_conversation_history():
    """Display conversation history"""
    if not st.session_state.conversation_history:
        st.info("💭 No conversation yet. Ask a question about your documents!")
        return
    
    # Display messages in chronological order
    for i, (user_msg, assistant_msg, sources) in enumerate(st.session_state.conversation_history):
        # User message
        with st.chat_message("user", avatar="👤"):
            st.write(user_msg)
        
        # Assistant message
        with st.chat_message("assistant", avatar="🤖"):
            st.write(assistant_msg)
            
            # Display sources with legal structure info
            if sources:
                with st.expander(f"📚 Sources ({len(sources)} found)", expanded=False):
                    for j, source in enumerate(sources):
                        # Build source header with legal structure
                        source_header = f"**Source {j+1}: {source.get('document_filename', 'Unknown')}**"

                        # Add legal structure info if available
                        metadata = source.get('metadata', {})
                        legal_parts = []

                        if metadata.get('chapter'):
                            legal_parts.append(f"Chương {metadata['chapter']}")
                        if metadata.get('section'):
                            legal_parts.append(f"Mục {metadata['section']}")
                        if metadata.get('article'):
                            legal_parts.append(f"Điều {metadata['article']}")

                        if legal_parts:
                            source_header += f" *({' - '.join(legal_parts)})*"

                        st.markdown(source_header)

                        col1, col2 = st.columns([3, 1])
                        with col1:
                            content_preview = source.get('content_preview', source.get('content', ''))
                            st.write(content_preview)

                        with col2:
                            score = source.get('score', 0)
                            st.metric("Relevance", f"{score:.3f}")

                            # Show chunk type if legal document
                            chunk_type = metadata.get('chunk_type')
                            if chunk_type and chunk_type != 'standard':
                                st.caption(f"Type: {chunk_type}")

                        if j < len(sources) - 1:
                            st.markdown("---")

def handle_chat_input(selected_doc_ids: List[int]):
    """Handle chat input and response generation - supports both RAG and Normal Chat modes"""

    # Dynamic chat input placeholder based on mode
    if selected_doc_ids:
        placeholder = "Ask a question about your documents..."
    else:
        placeholder = "Ask me anything..."

    user_question = st.chat_input(placeholder)

    if user_question:
        # Display user message immediately
        with st.chat_message("user", avatar="👤"):
            st.write(user_question)

        # Generate and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    if selected_doc_ids:
                        # ========== RAG MODE ==========
                        # Query RAG system with document context
                        response = st.session_state.rag_engine.query(
                            question=user_question,
                            document_ids=selected_doc_ids,
                            top_k=5,
                            session_id=st.session_state.session_id
                        )

                        # Display answer
                        answer = response['answer']
                        sources = response.get('sources', [])
                        response_time = response.get('response_time_ms', 0)

                        st.write(answer)

                        # Display response metrics
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.caption(f"⚡ Response time: {response_time}ms")
                        with col2:
                            st.caption(f"📊 Sources: {len(sources)}")
                        with col3:
                            st.caption(f"📄 Documents: {len(selected_doc_ids)}")

                        # Display sources with legal structure info
                        if sources:
                            with st.expander(f"📚 Sources ({len(sources)} found)", expanded=False):
                                for i, source in enumerate(sources):
                                    # Build source header with legal structure
                                    source_header = f"**Source {i+1}: {source.get('document_filename', 'Unknown')}**"

                                    # Add legal structure info if available
                                    metadata = source.get('metadata', {})
                                    legal_parts = []

                                    if metadata.get('chapter'):
                                        legal_parts.append(f"Chương {metadata['chapter']}")
                                    if metadata.get('section'):
                                        legal_parts.append(f"Mục {metadata['section']}")
                                    if metadata.get('article'):
                                        legal_parts.append(f"Điều {metadata['article']}")

                                    if legal_parts:
                                        source_header += f" *({' - '.join(legal_parts)})*"

                                    st.markdown(source_header)

                                    col1, col2 = st.columns([3, 1])
                                    with col1:
                                        content_preview = source.get('content_preview', source.get('content', ''))
                                        st.write(content_preview)

                                    with col2:
                                        score = source.get('score', 0)
                                        st.metric("Relevance", f"{score:.3f}")

                                        # Show chunk type if legal document
                                        chunk_type = metadata.get('chunk_type')
                                        if chunk_type and chunk_type != 'standard':
                                            st.caption(f"Type: {chunk_type}")

                                    if i < len(sources) - 1:
                                        st.markdown("---")

                        # Add to conversation history
                        st.session_state.conversation_history.append(
                            (user_question, answer, sources)
                        )

                    else:
                        # ========== NORMAL CHAT MODE ==========
                        # Direct LLM call without document context
                        import time
                        start_time = time.time()

                        # Build simple chat prompt
                        chat_prompt = f"""Bạn là Trợ lý AI chuyên về pháp luật Việt Nam.

Mục tiêu:
- Giải thích các quy định pháp luật Việt Nam cho người dùng theo cách dễ hiểu.
- Hỗ trợ người dùng hiểu rõ quyền, nghĩa vụ, thủ tục, khái niệm pháp lý… theo quy định pháp luật Việt Nam.
- Chỉ cung cấp THÔNG TIN THAM KHẢO, KHÔNG phải tư vấn pháp lý chuyên nghiệp.

Giới hạn & nguyên tắc chung:
1. Bạn không phải là luật sư, không đại diện cho bất kỳ cơ quan nhà nước, tổ chức hành nghề luật sư hay cơ quan tiến hành tố tụng nào.
2. Kiến thức của bạn về pháp luật có thể KHÔNG được cập nhật đầy đủ theo các văn bản, sửa đổi, bổ sung mới nhất.
3. Luôn nhắc người dùng (ở phần Kết luận hoặc Lưu ý) rằng:
   “Thông tin chỉ mang tính tham khảo, không thay thế ý kiến tư vấn của luật sư hoặc cơ quan có thẩm quyền.”
4. Nếu câu hỏi quá cụ thể, có thể ảnh hưởng lớn đến quyền lợi (tranh chấp, tố tụng, hình sự, đất đai, thừa kế…), hãy:
   - Giải thích NGUYÊN TẮC CHUNG của pháp luật liên quan.
   - Đồng thời khuyến nghị người dùng liên hệ luật sư / cơ quan nhà nước để được hướng dẫn chính thức.
5. Không được cố gắng khẳng định thay cho cơ quan tiến hành tố tụng, tòa án hoặc cơ quan nhà nước (ví dụ: “Tòa chắc chắn sẽ xử…”, “Công an sẽ làm…”).

Cách trả lời:
1. Luôn dùng tiếng Việt, văn phong rõ ràng, mạch lạc, dễ hiểu với người không chuyên luật.
2. Khi có thể, hãy:
   - Nêu tên văn bản (ví dụ: Bộ luật Dân sự, Bộ luật Hình sự, Luật Đất đai, Luật Hôn nhân và Gia đình…).
   - Nêu nguyên tắc hoặc quy định điển hình (nếu bạn nhớ được ở mức tổng quan).
3. Về việc trích dẫn điều luật:
   - CHỈ nêu số điều, khoản, điểm, năm ban hành, số hiệu văn bản nếu bạn **thật sự chắc chắn**.
   - Nếu không chắc, hãy nói chung ở mức nguyên tắc (“theo Bộ luật Dân sự quy định về hợp đồng…”) và nêu rõ là bạn không chắc số điều cụ thể.
   - Tuyệt đối KHÔNG được bịa ra điều luật, số điều, số khoản hoặc nội dung chi tiết nếu không chắc.
4. Cấu trúc câu trả lời khuyến nghị:
   - **(1) Tóm tắt vấn đề người dùng hỏi**: 1–2 câu.
   - **(2) Nguyên tắc pháp luật liên quan**: giải thích luật quy định theo hướng tổng quan.
   - **(3) Áp dụng vào trường hợp chung**: mô tả vài kịch bản thường gặp, điều kiện, lưu ý.
   - **(4) Kết luận + khuyến nghị**:
       + Tóm lại ý chính.
       + Nhắc lại: “Thông tin chỉ mang tính tham khảo, không thay thế ý kiến tư vấn của luật sư hoặc cơ quan có thẩm quyền.”
       + Gợi ý người dùng nên làm gì tiếp theo (tìm hiểu văn bản nào, liên hệ cơ quan nào, gặp luật sư…).

Xử lý khi không chắc chắn:
1. Nếu bạn không chắc thông tin, hãy nói rõ:
   - “Tôi không chắc quy định hiện hành có còn như vậy không.”
   - Hoặc “Tôi không có đủ thông tin để khẳng định chính xác trong trường hợp này.”
2. Không được bịa ra điều luật, số điều, số khoản hoặc nội dung cụ thể nếu bạn không nhớ rõ.
3. Trong trường hợp thiếu thông tin (thời điểm xảy ra sự việc, loại hợp đồng, loại đất, tình trạng hôn nhân…), hãy nêu rõ:
   - Những yếu tố nào có thể làm thay đổi câu trả lời.
   - Gợi ý người dùng cung cấp thêm hoặc tham khảo luật sư.

Giới hạn về hiển thị suy luận (thinking content):
1. Bạn có thể suy luận nhiều bước ở bên trong để tìm câu trả lời phù hợp.
2. Tuyệt đối KHÔNG hiển thị bất kỳ phần nào mô tả quá trình suy nghĩ nội bộ như:
   - “Suy nghĩ: …”, “Phân tích: …”, “Reasoning: …”, “Chain-of-thought: …”, “Thought: …”
   - Các câu kiểu “Hãy cùng phân tích từng bước”, “Let’s think step by step”, “Bước 1, Bước 2…” dùng để mô tả quá trình suy nghĩ của chính bạn.
3. Chỉ hiển thị phần trả lời cuối cùng cho người dùng (giải thích, phân tích, ví dụ) theo cấu trúc đã nêu ở trên.

Phong cách giao tiếp:
- Lịch sự, khách quan, trung lập, không phán xét.
- Tránh từ ngữ tuyệt đối như “chắc chắn 100%”, “đảm bảo thắng kiện”…  
- Không xúi giục, khuyến khích vi phạm pháp luật, trốn thuế, lách luật, gian dối giấy tờ.
- Không đưa ra kết luận mang tính “cam kết kết quả” (ví dụ: “chắc chắn thắng kiện”, “chắc chắn được bồi thường”).

Ví dụ cách mở đầu câu trả lời:
- “Theo các nguyên tắc chung của pháp luật Việt Nam về […], thông thường sẽ có các điểm sau: …”
- “Với thông tin bạn cung cấp, tôi có thể giải thích một cách tổng quan như sau (không phải tư vấn pháp lý chính thức): …”
- “Trong thực tế, quy định cụ thể có thể phụ thuộc vào từng văn bản và từng thời điểm. Bạn nên kiểm tra lại văn bản hiện hành hoặc hỏi ý kiến luật sư/chuyên gia.”

User question: {user_question}

Answer: """

                        # 🔍 DEBUG: Log prompt being sent
                        logger.info("=" * 80)
                        logger.info("[NORMAL CHAT MODE] Prompt being sent to LLM:")
                        logger.info("=" * 80)
                        logger.info(chat_prompt)
                        logger.info("=" * 80)

                        # Call LLM service directly
                        answer = st.session_state.rag_engine.llm_service.generate_response(
                            prompt=chat_prompt,
                            max_tokens=1024,
                            temperature=0.7
                        )

                        # 🔍 DEBUG: Log final answer shown to user
                        logger.info("[NORMAL CHAT MODE] Final answer shown to user:")
                        logger.info("=" * 80)
                        logger.info(answer)
                        logger.info("=" * 80)

                        response_time = int((time.time() - start_time) * 1000)

                        # Display answer
                        st.write(answer)

                        # Display response metrics
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.caption(f"⚡ Response time: {response_time}ms")
                        with col2:
                            st.caption("Normal Chat Mode")

                        # Add to conversation history (no sources)
                        st.session_state.conversation_history.append(
                            (user_question, answer, [])
                        )

                except Exception as e:
                    logger.error(f"Error in chat: {e}")
                    st.error(f"❌ Error generating response: {str(e)}")

                    # Add error to conversation history
                    error_response = f"I encountered an error: {str(e)}"
                    st.session_state.conversation_history.append(
                        (user_question, error_response, [])
                    )

def display_system_status():
    """Display system status in sidebar - FIXED for Qdrant"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    
    if st.session_state.rag_engine:
        try:
            health_status = st.session_state.rag_engine.health_check()
            
            # CORRECT component names for Qdrant
            component_names = {
                'embedding_service': 'Embedding Service',
                'vector_store': 'Qdrant Vector Store',  # ✅ Already correct
                'llm_service': 'LLM Service (llama.cpp)'  # ✅ Updated for llama.cpp
            }
            
            for component, status in health_status.items():
                emoji = "✅" if status else "❌"
                display_name = component_names.get(component, component.replace('_', ' ').title())
                st.sidebar.write(f"{emoji} {display_name}")
        
        except Exception as e:
            st.sidebar.error(f"❌ Health check failed: {str(e)}")
    else:
        st.sidebar.error("❌ System not initialized")
    
    # Display session info
    with st.sidebar.expander("📊 Session Info", expanded=False):
        st.write(f"**Session ID:** `{st.session_state.session_id[:8]}...`")
        st.write(f"**Messages:** {len(st.session_state.conversation_history)}")
        st.write(f"**Selected Docs:** {len(st.session_state.selected_documents)}")
        
        # FIXED: Add Qdrant-specific info
        if st.session_state.rag_engine:
            try:
                stats = st.session_state.rag_engine.get_system_stats()
                qdrant_stats = stats.get('qdrant', {})
                if 'total_vectors' in qdrant_stats:
                    st.write(f"**Vectors in Qdrant:** {qdrant_stats['total_vectors']}")
            except Exception as e:
                logger.debug(f"Could not fetch Qdrant stats: {e}")
                pass
        

def main():
    """Main application function"""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Display header
        display_header()
        
        # Initialize system if not already done
        if not st.session_state.system_initialized:
            rag_engine, success = initialize_system()
            if success:
                st.session_state.rag_engine = rag_engine
                st.session_state.system_initialized = True
                st.success("✅ System initialized successfully!")
                st.rerun()
            else:
                st.stop()
        
        # Handle file uploads and display document library
        handle_file_upload()
        selected_doc_ids = display_document_library()
        st.session_state.selected_documents = selected_doc_ids
        
        # Display chat interface
        display_chat_interface(selected_doc_ids)
        
        # Display system status
        display_system_status()
        
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        st.error(f"❌ Application error: {str(e)}")
        st.info("🔄 Please refresh the page to restart the application.")

if __name__ == "__main__":
    main()
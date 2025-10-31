import streamlit as st
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
import json

logger = logging.getLogger(__name__)

class ChatComponent:
    """Enhanced chat interface component for Phase 2"""
    
    def __init__(self, rag_engine, db_session):
        self.rag_engine = rag_engine
        self.db_session = db_session
    
    def render_chat_interface(
        self, 
        selected_doc_ids: List[int],
        session_id: str,
        conversation_history: List[Dict] = None
    ):
        """Render enhanced chat interface with better UX"""
        
        # Chat configuration sidebar
        with st.sidebar:
            self._render_chat_settings()
        
        # Main chat area
        if not selected_doc_ids:
            self._render_no_documents_message()
            return
        
        # Document context header
        self._render_document_context(selected_doc_ids)
        
        # Conversation history with better formatting
        if conversation_history:
            self._render_enhanced_conversation_history(conversation_history)
        
        # Enhanced chat input with suggestions
        self._handle_enhanced_chat_input(selected_doc_ids, session_id)
    
    def _render_chat_settings(self):
        """Render chat configuration options"""
        st.subheader("Chat Settings")
        
        # Response settings
        with st.expander("Response Settings", expanded=False):
            max_tokens = st.slider("Max Response Length", 512, 4096, 2048, 256)
            temperature = st.slider("Creativity", 0.0, 1.0, 0.1, 0.1)
            include_sources = st.checkbox("Show Sources", True)
            response_style = st.selectbox(
                "Response Style",
                ["Detailed", "Concise", "Bullet Points", "Academic"]
            )
        
        # Search settings
        with st.expander("Search Settings", expanded=False):
            search_strategy = st.selectbox(
                "Search Strategy",
                ["Semantic", "Hybrid (Semantic + Keyword)", "Keyword Only"]
            )
            top_k = st.slider("Retrieved Chunks", 3, 10, 5)
            min_score = st.slider("Minimum Relevance", 0.0, 1.0, 0.3, 0.1)
        
        return {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "include_sources": include_sources,
            "response_style": response_style,
            "search_strategy": search_strategy,
            "top_k": top_k,
            "min_score": min_score
        }
    
    def _render_no_documents_message(self):
        """Enhanced no documents message with suggestions"""
        st.info("👈 Please select documents from the sidebar to start chatting.")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("**🚀 Quick Start:**")
            st.markdown("""
            1. Upload documents in sidebar
            2. Wait for processing completion  
            3. Select documents to chat with
            4. Ask your questions below!
            """)
        
        with col2:
            st.markdown("**💡 Example Questions:**")
            examples = [
                "What are the main points?",
                "Summarize key findings",
                "Compare these documents", 
                "What are the conclusions?",
                "Find contradictions or agreements"
            ]
            for example in examples:
                if st.button(f"📝 {example}", key=f"example_{example}"):
                    st.session_state.example_query = example
    
    def _render_document_context(self, selected_doc_ids: List[int]):
        """Show context about selected documents"""
        from src.utils.database import Document
        
        docs = self.db_session.query(Document).filter(
            Document.id.in_(selected_doc_ids)
        ).all()
        
        st.markdown("### 📚 **Active Documents**")
        
        cols = st.columns(min(len(docs), 3))
        for i, doc in enumerate(docs):
            with cols[i % 3]:
                with st.container():
                    st.markdown(f"**{doc.original_filename}**")
                    st.caption(f"📄 {doc.total_chunks} chunks | 📅 {doc.upload_date.strftime('%m/%d')}")
                    
                    if doc.summary:
                        with st.expander("📋 Summary"):
                            st.markdown(doc.summary[:200] + "..." if len(doc.summary) > 200 else doc.summary)
    
    def _render_enhanced_conversation_history(self, conversations: List[Dict]):
        """Enhanced conversation display with better formatting"""
        st.markdown("### 💬 **Conversation History**")
        
        # Conversation controls
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.caption(f"📊 {len(conversations)} messages in this session")
        with col2:
            if st.button("🔄 Refresh", help="Reload conversation"):
                st.rerun()
        with col3:
            if st.button("🗑️ Clear History", help="Clear conversation history"):
                if st.confirm("Clear conversation history?"):
                    # Logic to clear history
                    st.success("History cleared!")
                    st.rerun()
        
        # Display conversations
        for i, conv in enumerate(reversed(conversations[-10:])):  # Show last 10
            with st.container():
                # User message
                st.markdown(f"**👤 You** • {conv.get('timestamp', 'Unknown time')}")
                st.markdown(f"_{conv['user_message']}_")
                
                # Assistant response
                st.markdown(f"**🤖 Assistant** • Response time: {conv.get('response_time_ms', 0)}ms")
                st.markdown(conv['assistant_response'])
                
                # Sources if available
                if conv.get('sources') and conv['sources']:
                    with st.expander(f"📚 Sources ({len(conv['sources'])} chunks)"):
                        for j, source in enumerate(conv['sources'][:3]):  # Show top 3
                            st.markdown(f"**Source {j+1}** (Score: {source.get('score', 0):.3f})")
                            st.code(source.get('content', '')[:200] + "...")
                
                st.divider()
    
    def _handle_enhanced_chat_input(self, selected_doc_ids: List[int], session_id: str):
        """Enhanced chat input with suggestions and better UX"""
        
        st.markdown("### 💭 **Ask a Question**")
        
        # Quick suggestion buttons
        if hasattr(st.session_state, 'example_query'):
            suggested_query = st.session_state.example_query
            del st.session_state.example_query
        else:
            suggested_query = ""
        
        # Main input
        question = st.text_area(
            "Your question:",
            value=suggested_query,
            placeholder="Ask about your documents... (e.g., 'What are the main conclusions?')",
            height=100,
            key="main_question_input"
        )
        
        # Input controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            send_button = st.button("🚀 Send Question", type="primary", use_container_width=True)
        
        with col2:
            if st.button("💡 Suggest", help="Get question suggestions"):
                suggestions = self._get_question_suggestions(selected_doc_ids)
                st.session_state.question_suggestions = suggestions
        
        with col3:
            if st.button("🔄 Clear", help="Clear input"):
                st.session_state.main_question_input = ""
                st.rerun()
        
        # Show suggestions if available
        if hasattr(st.session_state, 'question_suggestions'):
            st.markdown("**💡 Suggested Questions:**")
            cols = st.columns(2)
            for i, suggestion in enumerate(st.session_state.question_suggestions[:4]):
                with cols[i % 2]:
                    if st.button(f"📝 {suggestion}", key=f"suggestion_{i}"):
                        st.session_state.main_question_input = suggestion
                        st.rerun()
        
        # Handle question submission
        if send_button and question.strip():
            self._process_enhanced_question(question, selected_doc_ids, session_id)
    
    def _get_question_suggestions(self, selected_doc_ids: List[int]) -> List[str]:
        """Generate contextual question suggestions"""
        from src.utils.database import Document
        
        docs = self.db_session.query(Document).filter(
            Document.id.in_(selected_doc_ids)
        ).all()
        
        # Basic suggestions based on document types
        suggestions = [
            "What are the main topics discussed?",
            "Summarize the key findings",
            "What are the most important conclusions?"
        ]
        
        if len(docs) > 1:
            suggestions.extend([
                "Compare the main arguments across documents",
                "What are the common themes?",
                "Where do these documents agree or disagree?"
            ])
        
        return suggestions
    
    def _process_enhanced_question(self, question: str, selected_doc_ids: List[int], session_id: str):
        """Process question with enhanced features and timing"""
        
        start_time = time.time()
        
        # Show processing status
        with st.status("🔍 Processing your question...", expanded=True) as status:
            st.write("🔍 Searching relevant content...")
            
            try:
                # Get chat settings
                settings = self._render_chat_settings() if hasattr(self, 'chat_settings') else {}
                
                # Enhanced query processing
                response = self.rag_engine.query(
                    question=question,
                    document_ids=selected_doc_ids,
                    top_k=settings.get('top_k', 5),
                    min_score=settings.get('min_score', 0.3)
                )
                
                processing_time = int((time.time() - start_time) * 1000)
                
                st.write("✅ Generated response!")
                status.update(label="✅ Question processed successfully!", state="complete")
                
                # Display response with enhanced formatting
                self._display_enhanced_response(response, processing_time)
                
                # Save to conversation history
                self._save_conversation(session_id, question, response, processing_time, selected_doc_ids)
                
            except Exception as e:
                logger.error(f"Error processing question: {e}")
                status.update(label="❌ Error processing question", state="error")
                st.error(f"Error: {str(e)}")
    
    def _display_enhanced_response(self, response: Dict, processing_time: int):
        """Display response with enhanced formatting and sources"""
        
        st.markdown("### 🤖 **Assistant Response**")
        
        # Response metadata
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.caption(f"⏱️ Response time: {processing_time}ms")
        with col2:
            st.caption(f"📊 {len(response.get('sources', []))} sources used")
        with col3:
            confidence = response.get('confidence', 0.8)
            st.caption(f"🎯 Confidence: {confidence:.1%}")
        
        # Main response
        st.markdown(response.get('answer', 'No response generated'))
        
        # Sources section
        sources = response.get('sources', [])
        if sources:
            with st.expander(f"📚 **Source Details** ({len(sources)} chunks)", expanded=False):
                
                # Source summary tabs
                tab1, tab2 = st.tabs(["📊 Source Summary", "📝 Full Sources"])
                
                with tab1:
                    # Group sources by document
                    doc_groups = {}
                    for source in sources:
                        doc_id = source.get('document_id', 'Unknown')
                        if doc_id not in doc_groups:
                            doc_groups[doc_id] = []
                        doc_groups[doc_id].append(source)
                    
                    for doc_id, doc_sources in doc_groups.items():
                        st.markdown(f"**📄 Document {doc_id}:** {len(doc_sources)} relevant chunks")
                        avg_score = sum(s.get('score', 0) for s in doc_sources) / len(doc_sources)
                        st.progress(avg_score, text=f"Avg. Relevance: {avg_score:.3f}")
                
                with tab2:
                    for i, source in enumerate(sources):
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"**📄 Source {i+1}** - Document {source.get('document_id', 'Unknown')}")
                            with col2:
                                score = source.get('score', 0)
                                st.metric("Relevance", f"{score:.3f}")
                            
                            st.code(source.get('content', 'No content available')[:500] + "...")
                            st.divider()
    
    def _save_conversation(self, session_id: str, question: str, response: Dict, 
                          processing_time: int, document_ids: List[int]):
        """Save conversation with enhanced metadata"""
        from src.utils.database import Conversation
        
        try:
            conversation = Conversation(
                session_id=session_id,
                user_message=question,
                assistant_response=response.get('answer', ''),
                context_documents=document_ids,
                sources=response.get('sources', []),
                response_time_ms=processing_time,
                created_at=datetime.utcnow()
            )
            
            self.db_session.add(conversation)
            self.db_session.commit()
            
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            self.db_session.rollback()
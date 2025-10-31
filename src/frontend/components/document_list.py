import streamlit as st
import pandas as pd
from typing import List, Dict, Optional
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class DocumentListComponent:
    """Enhanced document library management for Phase 2"""
    
    def __init__(self, db_session):
        self.db_session = db_session
    
    def render_document_library(self, show_detailed=True) -> List[int]:
        """Render enhanced document library with filtering and analytics"""
        
        st.markdown("### 📚 **Document Library**")
        
        # Load documents
        documents = self._load_documents()
        
        if not documents:
            self._render_empty_library()
            return []
        
        # Document filters and search
        selected_doc_ids = self._render_document_filters(documents)
        
        # Document statistics
        if show_detailed:
            self._render_document_statistics(documents)
        
        # Document grid/list view
        if st.toggle("📋 List View", value=True):
            self._render_document_list(documents, selected_doc_ids)
        else:
            self._render_document_grid(documents, selected_doc_ids)
        
        return selected_doc_ids
    
    def _load_documents(self) -> List[Dict]:
        """Load documents with enhanced metadata"""
        from src.utils.database import Document, DocumentChunk
        
        try:
            # Get documents with chunk counts and recent activity
            docs_query = self.db_session.query(Document).order_by(Document.upload_date.desc())
            documents = []
            
            for doc in docs_query.all():
                # Get chunk count
                chunk_count = self.db_session.query(DocumentChunk).filter(
                    DocumentChunk.document_id == doc.id
                ).count()
                
                doc_data = {
                    'id': doc.id,
                    'filename': doc.original_filename,
                    'file_type': doc.file_type,
                    'file_size': doc.file_size,
                    'upload_date': doc.upload_date,
                    'processing_status': doc.processing_status,
                    'total_chunks': chunk_count,
                    'summary': doc.summary,
                    'metadata': doc.document_metadata or {}
                }
                documents.append(doc_data)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            st.error("Error loading document library")
            return []
    
    def _render_empty_library(self):
        """Show empty state with helpful guidance"""
        st.info("📁 Your document library is empty")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("""
            **🚀 Get Started:**
            1. Click "Upload Documents" above
            2. Select PDF, TXT, MD, DOCX, or DOC files
            3. Wait for processing
            4. Start chatting with your documents!
            """)
        
        with col2:
            st.markdown("""
            **📊 Supported Formats:**
            - 📄 PDF documents
            - 📝 Text files (.txt)
            - 📋 Markdown files (.md)
            - 📁 Up to 100MB per file
            """)
    
    def _render_document_filters(self, documents: List[Dict]) -> List[int]:
        """Enhanced document filtering and selection"""
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Search bar
            search_term = st.text_input(
                "🔍 Search documents:",
                placeholder="Search by filename, content, or type..."
            )
        
        with col2:
            # File type filter
            file_types = list(set(doc['file_type'] for doc in documents))
            selected_types = st.multiselect(
                "📁 File Types:",
                options=file_types,
                default=file_types
            )
        
        with col3:
            # Processing status filter
            statuses = list(set(doc['processing_status'] for doc in documents))
            selected_statuses = st.multiselect(
                "⚙️ Status:",
                options=statuses,
                default=['completed']  # Default to completed only
            )
        
        # Date range filter
        col1, col2 = st.columns([1, 1])
        
        with col1:
            min_date = min(doc['upload_date'].date() for doc in documents)
            date_from = st.date_input(
                "📅 From:",
                value=min_date,
                min_value=min_date,
                max_value=datetime.now().date()
            )
        
        with col2:
            date_to = st.date_input(
                "📅 To:",
                value=datetime.now().date(),
                min_value=min_date,
                max_value=datetime.now().date()
            )
        
        # Apply filters
        filtered_docs = self._apply_filters(
            documents, search_term, selected_types, 
            selected_statuses, date_from, date_to
        )
        
        # Document selection
        st.markdown(f"**📊 {len(filtered_docs)} documents found**")
        
        if filtered_docs:
            # Select all/none buttons
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if st.button("✅ Select All"):
                    st.session_state.selected_docs = [doc['id'] for doc in filtered_docs]
                    st.rerun()
            
            with col2:
                if st.button("❌ Select None"):
                    st.session_state.selected_docs = []
                    st.rerun()
            
            # Multi-select for documents
            default_selected = getattr(st.session_state, 'selected_docs', [])
            doc_options = {doc['filename']: doc['id'] for doc in filtered_docs}
            
            selected_names = st.multiselect(
                "📋 Choose documents for chat:",
                options=list(doc_options.keys()),
                default=[name for name, doc_id in doc_options.items() 
                        if doc_id in default_selected],
                help="Select documents to include in your conversation"
            )
            
            selected_doc_ids = [doc_options[name] for name in selected_names]
            st.session_state.selected_docs = selected_doc_ids
            
            return selected_doc_ids
        
        return []
    
    def _apply_filters(self, documents: List[Dict], search_term: str, 
                      file_types: List[str], statuses: List[str],
                      date_from, date_to) -> List[Dict]:
        """Apply all filters to document list"""
        
        filtered = documents
        
        # Search filter
        if search_term:
            search_lower = search_term.lower()
            filtered = [
                doc for doc in filtered
                if search_lower in doc['filename'].lower()
                or search_lower in doc['file_type'].lower()
                or (doc['summary'] and search_lower in doc['summary'].lower())
            ]
        
        # File type filter
        if file_types:
            filtered = [doc for doc in filtered if doc['file_type'] in file_types]
        
        # Status filter
        if statuses:
            filtered = [doc for doc in filtered if doc['processing_status'] in statuses]
        
        # Date filter
        filtered = [
            doc for doc in filtered
            if date_from <= doc['upload_date'].date() <= date_to
        ]
        
        return filtered
    
    def _render_document_statistics(self, documents: List[Dict]):
        """Show document library statistics"""
        
        if not documents:
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📄 Total Documents", len(documents))
        
        with col2:
            completed = len([d for d in documents if d['processing_status'] == 'completed'])
            st.metric("✅ Processed", completed)
        
        with col3:
            total_chunks = sum(d.get('total_chunks', 0) for d in documents)
            st.metric("🧩 Total Chunks", total_chunks)
        
        with col4:
            total_size_mb = sum(d.get('file_size', 0) for d in documents) / (1024 * 1024)
            st.metric("💾 Total Size", f"{total_size_mb:.1f} MB")
        
        # Recent activity
        with st.expander("📈 Recent Activity", expanded=False):
            recent_docs = sorted(documents, key=lambda x: x['upload_date'], reverse=True)[:5]
            
            for doc in recent_docs:
                days_ago = (datetime.now() - doc['upload_date']).days
                time_str = f"{days_ago} days ago" if days_ago > 0 else "Today"
                
                st.markdown(f"📄 **{doc['filename']}** - {time_str}")
    
    def _render_document_list(self, documents: List[Dict], selected_ids: List[int]):
        """Render documents in list view"""
        
        for doc in documents:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    # Document info
                    selected = "✅" if doc['id'] in selected_ids else "⬜"
                    status_emoji = "✅" if doc['processing_status'] == 'completed' else "⚠️"
                    
                    st.markdown(f"{selected} {status_emoji} **{doc['filename']}**")
                    st.caption(f"{doc['file_type']} • {doc.get('total_chunks', 0)} chunks")
                
                with col2:
                    # File size
                    size_mb = doc.get('file_size', 0) / (1024 * 1024)
                    st.metric("Size", f"{size_mb:.1f} MB")
                
                with col3:
                    # Upload date
                    days_ago = (datetime.now() - doc['upload_date']).days
                    st.metric("Age", f"{days_ago}d")
                
                with col4:
                    # Actions
                    if st.button("🗑️", key=f"delete_{doc['id']}", help="Delete document"):
                        self._handle_document_deletion(doc['id'])
                
                # Document summary
                if doc.get('summary'):
                    with st.expander("📋 Summary"):
                        st.markdown(doc['summary'])
                
                st.divider()
    
    def _render_document_grid(self, documents: List[Dict], selected_ids: List[int]):
        """Render documents in grid view"""
        
        cols_per_row = 3
        for i in range(0, len(documents), cols_per_row):
            cols = st.columns(cols_per_row)
            
            for j, doc in enumerate(documents[i:i+cols_per_row]):
                with cols[j]:
                    self._render_document_card(doc, selected_ids)
    
    def _render_document_card(self, doc: Dict, selected_ids: List[int]):
        """Render individual document card"""
        
        selected = doc['id'] in selected_ids
        card_style = "border: 2px solid #00FF00;" if selected else "border: 1px solid #DDD;"
        
        with st.container():
            st.markdown(f'<div style="{card_style} padding: 10px; border-radius: 5px;">', 
                       unsafe_allow_html=True)
            
            # Status indicator
            status_emoji = "✅" if doc['processing_status'] == 'completed' else "⚠️"
            st.markdown(f"{status_emoji} **{doc['filename'][:30]}**")
            
            # Document info
            size_mb = doc.get('file_size', 0) / (1024 * 1024)
            st.caption(f"{doc['file_type']} • {size_mb:.1f} MB")
            st.caption(f"{doc.get('total_chunks', 0)} chunks")
            
            # Actions
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👁️", key=f"view_{doc['id']}", help="View details"):
                    self._show_document_details(doc)
            
            with col2:
                if st.button("🗑️", key=f"del_{doc['id']}", help="Delete"):
                    self._handle_document_deletion(doc['id'])
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    def _show_document_details(self, doc: Dict):
        """Show detailed document information"""
        st.session_state.show_doc_details = doc
    
    def _handle_document_deletion(self, doc_id: int):
        """Handle document deletion with confirmation"""
        if st.confirm(f"Delete document ID {doc_id}?"):
            try:
                from src.utils.database import Document, DocumentChunk
                
                # Delete chunks first (cascade should handle this)
                self.db_session.query(DocumentChunk).filter(
                    DocumentChunk.document_id == doc_id
                ).delete()
                
                # Delete document
                self.db_session.query(Document).filter(
                    Document.id == doc_id
                ).delete()
                
                self.db_session.commit()
                
                st.success("Document deleted successfully!")
                st.rerun()
                
            except Exception as e:
                logger.error(f"Error deleting document: {e}")
                st.error("Error deleting document")
                self.db_session.rollback()
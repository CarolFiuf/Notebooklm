import streamlit as st
import logging
from typing import List, Dict, Optional, Tuple
import os
import tempfile
from pathlib import Path
import hashlib
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class UploadComponent:
    """Enhanced file upload component for Phase 2"""
    
    def __init__(self, document_processor, db_session):
        self.document_processor = document_processor
        self.db_session = db_session
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.supported_extensions = ['.pdf', '.txt', '.md', '.docx', '.doc']
    
    def render_upload_interface(self) -> bool:
        """Render enhanced upload interface with progress tracking"""
        
        st.markdown("### 📁 **Upload Documents**")
        
        # Upload configuration
        with st.expander("⚙️ Upload Settings", expanded=False):
            self._render_upload_settings()
        
        # File upload area
        uploaded_files = st.file_uploader(
            "Choose files to upload:",
            type=['pdf', 'txt', 'md', 'docx', 'doc'],
            accept_multiple_files=True,
            help="Supported: PDF, TXT, MD, DOCX, DOC files (max 100MB each)"
        )
        
        if uploaded_files:
            return self._handle_file_uploads(uploaded_files)
        
        # Upload tips and status
        self._render_upload_guidance()
        
        return False
    
    def _render_upload_settings(self):
        """Render upload configuration options"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            chunk_strategy = st.selectbox(
                "Chunking Strategy:",
                ["Recursive (Recommended)", "Semantic", "Fixed Size", "Paragraph-based"],
                help="How to split documents into chunks"
            )
            
            chunk_size = st.slider(
                "Chunk Size (characters):",
                min_value=500,
                max_value=2000,
                value=1000,
                step=100,
                help="Size of each text chunk"
            )
        
        with col2:
            chunk_overlap = st.slider(
                "Chunk Overlap (characters):",
                min_value=50,
                max_value=500,
                value=200,
                step=50,
                help="Overlap between adjacent chunks"
            )
            
            auto_summarize = st.checkbox(
                "Auto-generate summaries",
                value=True,
                help="Automatically create document summaries"
            )
        
        return {
            'chunk_strategy': chunk_strategy,
            'chunk_size': chunk_size,
            'chunk_overlap': chunk_overlap,
            'auto_summarize': auto_summarize
        }
    
    def _render_upload_guidance(self):
        """Show upload tips and current status"""
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📋 Upload Tips:**")
            st.markdown("""
            - **PDF files**: Text-based PDFs work best
            - **Word files**: .docx and .doc formats supported
            - **Text files**: UTF-8 encoding preferred
            - **File size**: Max 100MB per file
            - **Quality**: Clear, well-formatted text gives better results
            """)
        
        with col2:
            # Show recent upload statistics
            stats = self._get_upload_stats()
            st.markdown("**📊 Upload Statistics:**")
            st.metric("Today's Uploads", stats.get('today_uploads', 0))
            st.metric("Processing Queue", stats.get('pending_count', 0))
            st.metric("Success Rate", f"{stats.get('success_rate', 100):.1f}%")
    
    def _get_upload_stats(self) -> Dict:
        """Get upload statistics for the interface"""
        from src.utils.database import Document
        
        try:
            today = datetime.now().date()
            
            # Today's uploads
            today_uploads = self.db_session.query(Document).filter(
                Document.upload_date >= datetime.combine(today, datetime.min.time())
            ).count()
            
            # Pending processing
            pending_count = self.db_session.query(Document).filter(
                Document.processing_status.in_(['pending', 'processing'])
            ).count()
            
            # Success rate (completed vs total)
            total_docs = self.db_session.query(Document).count()
            completed_docs = self.db_session.query(Document).filter(
                Document.processing_status == 'completed'
            ).count()
            
            success_rate = (completed_docs / total_docs * 100) if total_docs > 0 else 100
            
            return {
                'today_uploads': today_uploads,
                'pending_count': pending_count,
                'success_rate': success_rate
            }
            
        except Exception as e:
            logger.error(f"Error getting upload stats: {e}")
            return {}
    
    def _handle_file_uploads(self, uploaded_files) -> bool:
        """Handle multiple file uploads with progress tracking"""
        
        st.markdown(f"**📤 Processing {len(uploaded_files)} file(s)...**")
        
        # Pre-upload validation
        validation_results = self._validate_uploads(uploaded_files)
        
        if not validation_results['valid']:
            st.error("Upload validation failed:")
            for error in validation_results['errors']:
                st.error(f"• {error}")
            return False
        
        # Get upload settings
        settings = self._render_upload_settings()
        
        # Process files with progress tracking
        success_count = 0
        total_files = len(uploaded_files)
        
        # Overall progress bar
        overall_progress = st.progress(0)
        status_placeholder = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            
            # Update overall progress
            progress = i / total_files
            overall_progress.progress(progress)
            status_placeholder.text(f"Processing {uploaded_file.name}...")
            
            # Individual file processing
            success = self._process_single_file(
                uploaded_file, 
                settings, 
                file_number=i+1,
                total_files=total_files
            )
            
            if success:
                success_count += 1
        
        # Final progress update
        overall_progress.progress(1.0)
        
        # Show results
        if success_count == total_files:
            status_placeholder.success(f"✅ Successfully uploaded {success_count} file(s)!")
        else:
            status_placeholder.warning(f"⚠️ Uploaded {success_count}/{total_files} files successfully")
        
        return success_count > 0
    
    def _validate_uploads(self, uploaded_files) -> Dict:
        """Validate uploaded files before processing"""
        
        errors = []
        valid = True
        
        for file in uploaded_files:
            # Check file size
            if file.size > self.max_file_size:
                errors.append(f"{file.name}: File size {file.size/1024/1024:.1f}MB exceeds 100MB limit")
                valid = False
            
            # Check file extension
            file_ext = Path(file.name).suffix.lower()
            if file_ext not in self.supported_extensions:
                errors.append(f"{file.name}: Unsupported file type '{file_ext}'")
                valid = False
            
            # Check for duplicate names
            existing = self._check_duplicate_filename(file.name)
            if existing:
                errors.append(f"{file.name}: File with this name already exists")
                valid = False
        
        return {'valid': valid, 'errors': errors}
    
    def _check_duplicate_filename(self, filename: str) -> bool:
        """Check if filename already exists"""
        from src.utils.database import Document

        try:
            existing = self.db_session.query(Document).filter(
                Document.original_filename == filename
            ).first()
            return existing is not None
        except Exception as e:
            logger.error(f"Error checking duplicate filename '{filename}': {e}")
            return False  # Fail-safe: assume not duplicate on error
    
    def _process_single_file(self, uploaded_file, settings: Dict, 
                           file_number: int, total_files: int) -> bool:
        """Process a single uploaded file with detailed progress"""
        
        try:
            # Create expander for this file's progress
            with st.expander(f"📄 Processing {uploaded_file.name} ({file_number}/{total_files})", expanded=True):
                
                # File info
                file_size_mb = uploaded_file.size / (1024 * 1024)
                st.info(f"📊 Size: {file_size_mb:.2f} MB | Type: {Path(uploaded_file.name).suffix}")
                
                # Processing steps with progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Save file
                status_text.text("💾 Saving file...")
                progress_bar.progress(0.1)
                
                temp_path = self._save_uploaded_file(uploaded_file)
                if not temp_path:
                    st.error("Failed to save file")
                    return False
                
                # Step 2: Extract text
                status_text.text("📖 Extracting text...")
                progress_bar.progress(0.3)
                
                extracted_text = self.document_processor.extract_text(temp_path)
                if not extracted_text:
                    st.error("Failed to extract text")
                    return False
                
                # Step 3: Create chunks
                status_text.text("🧩 Creating chunks...")
                progress_bar.progress(0.5)
                
                chunks = self.document_processor.create_chunks(
                    extracted_text,
                    chunk_size=settings['chunk_size'],
                    chunk_overlap=settings['chunk_overlap'],
                    strategy=settings['chunk_strategy']
                )
                
                if not chunks:
                    st.error("Failed to create chunks")
                    return False
                
                # Step 4: Generate embeddings
                status_text.text("🔢 Generating embeddings...")
                progress_bar.progress(0.7)
                
                embeddings_success = self.document_processor.generate_embeddings(chunks)
                if not embeddings_success:
                    st.error("Failed to generate embeddings")
                    return False
                
                # Step 5: Generate summary (if enabled)
                summary = None
                if settings['auto_summarize']:
                    status_text.text("📝 Generating summary...")
                    progress_bar.progress(0.9)
                    
                    summary = self._generate_document_summary(extracted_text[:2000])  # First 2000 chars
                
                # Step 6: Save to database
                status_text.text("💾 Saving to database...")
                progress_bar.progress(0.95)
                
                doc_id = self._save_document_to_db(
                    uploaded_file, temp_path, len(chunks), summary
                )
                
                if not doc_id:
                    st.error("Failed to save to database")
                    return False
                
                # Complete
                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")
                
                # Show results
                st.success(f"✅ Successfully processed {len(chunks)} chunks")
                if summary:
                    with st.expander("📋 Auto-generated Summary"):
                        st.write(summary)
                
                # Cleanup
                if temp_path:
                    os.unlink(temp_path)
                
                return True
                
        except Exception as e:
            logger.error(f"Error processing file {uploaded_file.name}: {e}")
            st.error(f"Error processing file: {str(e)}")
            return False
    
    def _save_uploaded_file(self, uploaded_file) -> Optional[str]:
        """Save uploaded file to temporary location"""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                return tmp_file.name
        except Exception as e:
            logger.error(f"Error saving uploaded file: {e}")
            return None
    
    def _generate_document_summary(self, text: str) -> Optional[str]:
        """Generate automatic document summary"""
        try:
            # Use the LLM service to generate summary
            from src.serving.llm_service import LLMService
            
            llm_service = LLMService()
            
            prompt = f"""
            Please provide a concise summary of this document in 2-3 sentences:

            {text}

            Summary:"""
            
            summary = llm_service.generate_response(prompt, max_tokens=150, temperature=0.1)
            return summary.strip() if summary else None
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return None
    
    def _save_document_to_db(self, uploaded_file, file_path: str, 
                           chunk_count: int, summary: Optional[str]) -> Optional[int]:
        """Save document metadata to database"""
        from src.utils.database import Document
        
        try:
            # Generate unique filename
            original_name = uploaded_file.name
            timestamp = int(time.time())
            unique_filename = f"{timestamp}_{original_name}"
            
            # Create document record
            document = Document(
                filename=unique_filename,
                original_filename=original_name,
                file_path=file_path,
                file_size=uploaded_file.size,
                file_type=Path(original_name).suffix.lower(),
                processing_status='completed',
                total_chunks=chunk_count,
                summary=summary,
                upload_date=datetime.utcnow(),
                created_at=datetime.utcnow()
            )
            
            self.db_session.add(document)
            self.db_session.commit()
            
            logger.info(f"Document saved to DB: {document.id}")
            return document.id
            
        except Exception as e:
            logger.error(f"Error saving document to DB: {e}")
            self.db_session.rollback()
            return None

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class AnalyticsComponent:
    """Analytics dashboard component for Phase 2"""
    
    def __init__(self, db_session):
        self.db_session = db_session
    
    def render_analytics_dashboard(self):
        """Render comprehensive analytics dashboard"""
        
        st.markdown("### 📊 **Analytics Dashboard**")
        
        # Time period selector
        time_period = self._render_time_period_selector()
        
        # Load analytics data
        analytics_data = self._load_analytics_data(time_period)
        
        if not analytics_data:
            st.info("No data available for the selected period")
            return
        
        # Render different sections
        self._render_overview_metrics(analytics_data)
        self._render_document_analytics(analytics_data)
        self._render_conversation_analytics(analytics_data)
        self._render_performance_analytics(analytics_data)
        self._render_usage_patterns(analytics_data)
    
    def _render_time_period_selector(self) -> Dict:
        """Render time period selection controls"""
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            period_type = st.selectbox(
                "📅 Time Period:",
                ["Last 7 days", "Last 30 days", "Last 90 days", "Custom range"]
            )
        
        if period_type == "Custom range":
            with col2:
                start_date = st.date_input(
                    "From:",
                    value=datetime.now().date() - timedelta(days=30)
                )
            
            with col3:
                end_date = st.date_input(
                    "To:",
                    value=datetime.now().date()
                )
        else:
            days_map = {
                "Last 7 days": 7,
                "Last 30 days": 30,
                "Last 90 days": 90
            }
            days = days_map[period_type]
            start_date = datetime.now().date() - timedelta(days=days)
            end_date = datetime.now().date()
        
        return {
            'start_date': start_date,
            'end_date': end_date,
            'period_type': period_type
        }
    
    def _load_analytics_data(self, time_period: Dict) -> Optional[Dict]:
        """Load all analytics data for the specified period"""
        
        try:
            from src.utils.database import Document, Conversation, DocumentChunk
            
            start_datetime = datetime.combine(time_period['start_date'], datetime.min.time())
            end_datetime = datetime.combine(time_period['end_date'], datetime.max.time())
            
            # Document analytics
            documents = self.db_session.query(Document).filter(
                Document.upload_date.between(start_datetime, end_datetime)
            ).all()
            
            # Conversation analytics
            conversations = self.db_session.query(Conversation).filter(
                Conversation.created_at.between(start_datetime, end_datetime)
            ).all()
            
            # Overall stats
            total_documents = self.db_session.query(Document).count()
            total_conversations = self.db_session.query(Conversation).count()
            total_chunks = self.db_session.query(DocumentChunk).count()
            
            return {
                'documents': documents,
                'conversations': conversations,
                'total_documents': total_documents,
                'total_conversations': total_conversations,
                'total_chunks': total_chunks,
                'period': time_period
            }
            
        except Exception as e:
            logger.error(f"Error loading analytics data: {e}")
            return None
    
    def _render_overview_metrics(self, data: Dict):
        """Render high-level overview metrics"""
        
        st.markdown("#### 🎯 **Overview Metrics**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            new_docs = len(data['documents'])
            st.metric(
                "📄 New Documents",
                new_docs,
                delta=f"{new_docs} in period"
            )
        
        with col2:
            new_conversations = len(data['conversations'])
            st.metric(
                "💬 Conversations",
                new_conversations,
                delta=f"{new_conversations} in period"
            )
        
        with col3:
            if data['conversations']:
                avg_response_time = sum(
                    c.response_time_ms or 0 for c in data['conversations']
                ) / len(data['conversations'])
                st.metric(
                    "⚡ Avg Response Time",
                    f"{avg_response_time:.0f}ms",
                    delta="Good" if avg_response_time < 3000 else "Slow"
                )
            else:
                st.metric("⚡ Avg Response Time", "N/A")
        
        with col4:
            success_rate = len([d for d in data['documents'] if d.processing_status == 'completed'])
            success_rate_pct = (success_rate / len(data['documents']) * 100) if data['documents'] else 100
            st.metric(
                "✅ Success Rate",
                f"{success_rate_pct:.1f}%",
                delta="Excellent" if success_rate_pct > 95 else "Needs Improvement"
            )
    
    def _render_document_analytics(self, data: Dict):
        """Render document-related analytics"""
        
        st.markdown("#### 📄 **Document Analytics**")
        
        if not data['documents']:
            st.info("No documents uploaded in this period")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Document uploads over time
            doc_df = pd.DataFrame([{
                'date': doc.upload_date.date(),
                'filename': doc.original_filename,
                'file_type': doc.file_type,
                'size_mb': doc.file_size / (1024 * 1024),
                'chunks': doc.total_chunks
            } for doc in data['documents']])
            
            # Daily upload chart
            daily_uploads = doc_df.groupby('date').size().reset_index(name='uploads')
            
            fig_uploads = px.line(
                daily_uploads, 
                x='date', 
                y='uploads',
                title='📈 Daily Document Uploads',
                markers=True
            )
            st.plotly_chart(fig_uploads, use_container_width=True)
        
        with col2:
            # File type distribution
            file_type_counts = doc_df['file_type'].value_counts()
            
            fig_types = px.pie(
                values=file_type_counts.values,
                names=file_type_counts.index,
                title='📊 File Type Distribution'
            )
            st.plotly_chart(fig_types, use_container_width=True)
        
        # Document size analytics
        col1, col2 = st.columns(2)
        
        with col1:
            # Size distribution
            fig_sizes = px.histogram(
                doc_df,
                x='size_mb',
                nbins=20,
                title='📏 Document Size Distribution (MB)'
            )
            st.plotly_chart(fig_sizes, use_container_width=True)
        
        with col2:
            # Chunk count distribution
            fig_chunks = px.box(
                doc_df,
                y='chunks',
                title='🧩 Chunks per Document'
            )
            st.plotly_chart(fig_chunks, use_container_width=True)
    
    def _render_conversation_analytics(self, data: Dict):
        """Render conversation-related analytics"""
        
        st.markdown("#### 💬 **Conversation Analytics**")
        
        if not data['conversations']:
            st.info("No conversations in this period")
            return
        
        # Prepare conversation data
        conv_df = pd.DataFrame([{
            'date': conv.created_at.date(),
            'hour': conv.created_at.hour,
            'response_time': conv.response_time_ms or 0,
            'user_message_length': len(conv.user_message),
            'assistant_response_length': len(conv.assistant_response),
            'sources_count': len(conv.sources) if conv.sources else 0,
            'session_id': conv.session_id
        } for conv in data['conversations']])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily conversation volume
            daily_convs = conv_df.groupby('date').size().reset_index(name='conversations')
            
            fig_daily = px.bar(
                daily_convs,
                x='date',
                y='conversations',
                title='📊 Daily Conversation Volume'
            )
            st.plotly_chart(fig_daily, use_container_width=True)
        
        with col2:
            # Hourly usage pattern
            hourly_usage = conv_df.groupby('hour').size().reset_index(name='conversations')
            
            fig_hourly = px.line(
                hourly_usage,
                x='hour',
                y='conversations',
                title='🕐 Hourly Usage Pattern',
                markers=True
            )
            st.plotly_chart(fig_hourly, use_container_width=True)
    
    def _render_performance_analytics(self, data: Dict):
        """Render performance-related analytics"""
        
        st.markdown("#### ⚡ **Performance Analytics**")
        
        if not data['conversations']:
            st.info("No performance data available")
            return
        
        conv_df = pd.DataFrame([{
            'response_time': conv.response_time_ms or 0,
            'sources_count': len(conv.sources) if conv.sources else 0,
            'created_at': conv.created_at
        } for conv in data['conversations']])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Response time distribution
            fig_response_time = px.histogram(
                conv_df,
                x='response_time',
                nbins=20,
                title='⏱️ Response Time Distribution (ms)'
            )
            st.plotly_chart(fig_response_time, use_container_width=True)
        
        with col2:
            # Response time vs sources count
            fig_scatter = px.scatter(
                conv_df,
                x='sources_count',
                y='response_time',
                title='📊 Response Time vs Sources Used',
                trendline='ols'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Performance metrics table
        perf_metrics = {
            'Metric': [
                'Average Response Time',
                'Median Response Time', 
                '95th Percentile Response Time',
                'Average Sources per Query'
            ],
            'Value': [
                f"{conv_df['response_time'].mean():.0f} ms",
                f"{conv_df['response_time'].median():.0f} ms",
                f"{conv_df['response_time'].quantile(0.95):.0f} ms",
                f"{conv_df['sources_count'].mean():.1f}"
            ]
        }
        
        st.markdown("**📈 Performance Summary:**")
        st.dataframe(pd.DataFrame(perf_metrics), hide_index=True)
    
    def _render_usage_patterns(self, data: Dict):
        """Render usage pattern analysis"""
        
        st.markdown("#### 📈 **Usage Patterns**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Most active sessions
            if data['conversations']:
                session_activity = pd.DataFrame([{
                    'session_id': conv.session_id,
                    'messages': 1
                } for conv in data['conversations']])
                
                session_counts = session_activity.groupby('session_id')['messages'].sum().sort_values(ascending=False)
                
                st.markdown("**🔥 Most Active Sessions:**")
                for i, (session_id, count) in enumerate(session_counts.head(5).items()):
                    st.metric(f"Session {session_id[:8]}...", f"{count} messages")
        
        with col2:
            # Document popularity (based on conversations mentioning them)
            doc_usage = {}
            for conv in data['conversations']:
                if conv.context_documents:
                    for doc_id in conv.context_documents:
                        doc_usage[doc_id] = doc_usage.get(doc_id, 0) + 1
            
            if doc_usage:
                st.markdown("**📚 Most Queried Documents:**")
                from src.utils.database import Document
                
                for doc_id, usage_count in sorted(doc_usage.items(), key=lambda x: x[1], reverse=True)[:5]:
                    doc = self.db_session.query(Document).filter(Document.id == doc_id).first()
                    if doc:
                        st.metric(f"{doc.original_filename[:20]}...", f"{usage_count} queries")
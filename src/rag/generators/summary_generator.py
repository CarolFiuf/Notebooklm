import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from src.rag.retrievers.semantic_retriever import SemanticRetriever
from src.serving.llm_service import LlamaCppService
from src.serving.prompts.rag_prompts import build_summary_prompt, build_comparison_prompt
from src.utils.database import get_db_session, Document, DocumentChunk

logger = logging.getLogger(__name__)

class SummaryGenerator:
    """Advanced document summary and analysis generation for Phase 2"""
    
    def __init__(self, llm_service: LlamaCppService = None, retriever: SemanticRetriever = None):
        self.llm_service = llm_service or LlamaCppService()
        self.retriever = retriever or SemanticRetriever()
        
        # Summary parameters
        self.max_content_length = 4000  # Max content for summarization
        self.summary_max_tokens = 512
        self.chunk_summary_tokens = 256
        
    def generate_document_summary(
        self,
        document_id: int,
        summary_type: str = "comprehensive",
        max_chunks: int = 10
    ) -> Dict[str, Any]:
        """
        Generate comprehensive document summary
        
        Args:
            document_id: ID of document to summarize
            summary_type: Type of summary (comprehensive, brief, key_points, executive)
            max_chunks: Maximum chunks to include
            
        Returns:
            Summary generation result
        """
        try:
            logger.info(f"Generating {summary_type} summary for document {document_id}")
            start_time = datetime.now()
            
            # Get document info
            doc_info = self._get_document_info(document_id)
            if not doc_info:
                return {'success': False, 'error': 'Document not found'}
            
            # Get representative chunks
            chunks = self._get_representative_chunks(document_id, max_chunks)
            if not chunks:
                return {'success': False, 'error': 'No content found'}
            
            # Generate summary based on type
            if summary_type == "comprehensive":
                summary = self._generate_comprehensive_summary(chunks, doc_info)
            elif summary_type == "brief":
                summary = self._generate_brief_summary(chunks, doc_info)
            elif summary_type == "key_points":
                summary = self._generate_key_points(chunks, doc_info)
            elif summary_type == "executive":
                summary = self._generate_executive_summary(chunks, doc_info)
            else:
                summary = self._generate_comprehensive_summary(chunks, doc_info)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'summary': summary,
                'summary_type': summary_type,
                'document_id': document_id,
                'document_filename': doc_info['filename'],
                'chunks_analyzed': len(chunks),
                'processing_time_seconds': processing_time,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating summary for document {document_id}: {e}")
            return {'success': False, 'error': str(e)}
    
    def generate_multi_document_summary(
        self,
        document_ids: List[int],
        focus_topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate summary across multiple documents
        
        Args:
            document_ids: List of document IDs
            focus_topic: Optional topic to focus the summary on
            
        Returns:
            Multi-document summary result
        """
        try:
            logger.info(f"Generating multi-document summary for {len(document_ids)} documents")
            start_time = datetime.now()
            
            if focus_topic:
                # Use retrieval to get relevant content across documents
                relevant_chunks = self.retriever.semantic_search(
                    query=focus_topic,
                    document_ids=document_ids,
                    top_k=15,
                    enable_reranking=True
                )
            else:
                # Get representative chunks from each document
                relevant_chunks = []
                for doc_id in document_ids:
                    doc_chunks = self._get_representative_chunks(doc_id, 3)
                    relevant_chunks.extend(doc_chunks)
            
            if not relevant_chunks:
                return {'success': False, 'error': 'No content found'}
            
            # Generate multi-document summary
            summary = self._generate_multi_doc_summary(relevant_chunks, focus_topic)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'summary': summary,
                'document_ids': document_ids,
                'focus_topic': focus_topic,
                'chunks_analyzed': len(relevant_chunks),
                'processing_time_seconds': processing_time,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating multi-document summary: {e}")
            return {'success': False, 'error': str(e)}
    
    def compare_documents(
        self,
        document_ids: List[int],
        comparison_aspects: List[str] = None
    ) -> Dict[str, Any]:
        """
        Generate comparative analysis between documents
        
        Args:
            document_ids: List of document IDs to compare
            comparison_aspects: Specific aspects to compare
            
        Returns:
            Document comparison result
        """
        try:
            if len(document_ids) < 2:
                return {'success': False, 'error': 'Need at least 2 documents for comparison'}
            
            logger.info(f"Comparing {len(document_ids)} documents")
            start_time = datetime.now()
            
            # Get document summaries or representative content
            doc_contents = []
            doc_infos = []
            
            for doc_id in document_ids:
                doc_info = self._get_document_info(doc_id)
                chunks = self._get_representative_chunks(doc_id, 5)
                
                if doc_info and chunks:
                    content = " ".join(chunk['content'] for chunk in chunks)
                    doc_contents.append({
                        'id': doc_id,
                        'title': doc_info['filename'],
                        'content': content[:2000]  # Limit content length
                    })
                    doc_infos.append(doc_info)
            
            if len(doc_contents) < 2:
                return {'success': False, 'error': 'Insufficient content for comparison'}
            
            # Generate comparison
            comparison = self._generate_document_comparison(doc_contents, comparison_aspects)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'comparison': comparison,
                'document_ids': document_ids,
                'comparison_aspects': comparison_aspects,
                'documents_compared': len(doc_contents),
                'processing_time_seconds': processing_time,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error comparing documents: {e}")
            return {'success': False, 'error': str(e)}
    
    def extract_key_insights(
        self,
        document_ids: List[int],
        insight_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Extract key insights from documents
        
        Args:
            document_ids: List of document IDs
            insight_type: Type of insights (general, technical, business, research)
            
        Returns:
            Key insights extraction result
        """
        try:
            logger.info(f"Extracting {insight_type} insights from {len(document_ids)} documents")
            start_time = datetime.now()
            
            # Get relevant content based on insight type
            if insight_type == "technical":
                search_query = "technical implementation approach method solution"
            elif insight_type == "business":
                search_query = "business strategy market opportunity revenue cost"
            elif insight_type == "research":
                search_query = "findings results conclusion research study"
            else:  # general
                search_query = "key important main significant insight"
            
            # Use retrieval to get relevant chunks
            relevant_chunks = self.retriever.semantic_search(
                query=search_query,
                document_ids=document_ids,
                top_k=20,
                enable_reranking=True
            )
            
            if not relevant_chunks:
                return {'success': False, 'error': 'No relevant content found'}
            
            # Generate insights
            insights = self._generate_key_insights(relevant_chunks, insight_type)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'success': True,
                'insights': insights,
                'insight_type': insight_type,
                'document_ids': document_ids,
                'chunks_analyzed': len(relevant_chunks),
                'processing_time_seconds': processing_time,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_document_info(self, document_id: int) -> Optional[Dict[str, Any]]:
        """Get document information from database"""
        try:
            db = get_db_session()
            document = db.query(Document).filter(Document.id == document_id).first()
            
            if document:
                info = {
                    'id': document.id,
                    'filename': document.original_filename,
                    'file_type': document.file_type,
                    'file_size': document.file_size,
                    'upload_date': document.upload_date,
                    'total_chunks': document.total_chunks,
                    'metadata': document.metadata or {}
                }
                db.close()
                return info
            
            db.close()
            return None
            
        except Exception as e:
            logger.error(f"Error getting document info: {e}")
            return None
    
    def _get_representative_chunks(
        self, 
        document_id: int, 
        max_chunks: int = 10
    ) -> List[Dict[str, Any]]:
        """Get representative chunks from document"""
        try:
            db = get_db_session()
            
            # Get chunks from document, preferring earlier chunks and longer content
            chunks = db.query(DocumentChunk).filter(
                DocumentChunk.document_id == document_id
            ).order_by(DocumentChunk.chunk_index).limit(max_chunks * 2).all()
            
            if not chunks:
                db.close()
                return []
            
            # Convert to list and select best chunks
            chunk_list = []
            for chunk in chunks:
                chunk_dict = {
                    'id': chunk.id,
                    'document_id': chunk.document_id,
                    'chunk_index': chunk.chunk_index,
                    'content': chunk.content,
                    'metadata': chunk.chunk_metadata or {},
                    'content_length': len(chunk.content)
                }
                chunk_list.append(chunk_dict)
            
            db.close()
            
            # Select best chunks (balance between position and content quality)
            chunk_list.sort(key=lambda x: (
                -x['content_length'],  # Prefer longer chunks
                x['chunk_index']       # Prefer earlier chunks
            ))
            
            return chunk_list[:max_chunks]
            
        except Exception as e:
            logger.error(f"Error getting representative chunks: {e}")
            return []
    
    def _generate_comprehensive_summary(
        self, 
        chunks: List[Dict[str, Any]], 
        doc_info: Dict[str, Any]
    ) -> str:
        """Generate comprehensive summary"""
        try:
            # Combine content from chunks
            combined_content = "\n\n".join(chunk['content'] for chunk in chunks)
            
            # Limit content length
            if len(combined_content) > self.max_content_length:
                combined_content = combined_content[:self.max_content_length] + "..."
            
            # Build comprehensive summary prompt
            summary_prompt = f"""Please provide a comprehensive summary of the following document content.

Document: {doc_info['filename']}
Type: {doc_info['file_type']}

Your summary should:
1. Capture the main topics and themes
2. Highlight key findings or arguments
3. Maintain the logical structure of the content
4. Be thorough but concise
5. No addition content, only document content

Content:
{combined_content}

Comprehensive Summary:"""

            # Generate summary
            summary = self.llm_service.generate_response(
                prompt=summary_prompt,
                max_tokens=self.summary_max_tokens,
                temperature=0.1
            )
            
            return summary or "Unable to generate comprehensive summary."
            
        except Exception as e:
            logger.error(f"Error generating comprehensive summary: {e}")
            return "Error generating summary."
    
    def _generate_brief_summary(
        self, 
        chunks: List[Dict[str, Any]], 
        doc_info: Dict[str, Any]
    ) -> str:
        """Generate brief summary"""
        try:
            # Use first few chunks for brief summary
            key_chunks = chunks[:5]
            combined_content = "\n\n".join(chunk['content'] for chunk in key_chunks)
            
            if len(combined_content) > 2000:
                combined_content = combined_content[:2000] + "..."
            
            brief_prompt = f"""Provide a brief summary of this document in 2-3 paragraphs.

Document: {doc_info['filename']}

Focus on:
- Main topic or purpose
- Key points or findings
- Primary conclusions

Content:
{combined_content}

Brief Summary:"""

            summary = self.llm_service.generate_response(
                prompt=brief_prompt,
                max_tokens=self.chunk_summary_tokens,
                temperature=0.1
            )
            
            return summary or "Unable to generate brief summary."
            
        except Exception as e:
            logger.error(f"Error generating brief summary: {e}")
            return "Error generating summary."
    
    def _generate_key_points(
        self, 
        chunks: List[Dict[str, Any]], 
        doc_info: Dict[str, Any]
    ) -> str:
        """Generate key points summary"""
        try:
            combined_content = "\n\n".join(chunk['content'] for chunk in chunks)
            
            if len(combined_content) > self.max_content_length:
                combined_content = combined_content[:self.max_content_length] + "..."
            
            key_points_prompt = f"""Extract the key points from this document and present them as a bulleted list.

Document: {doc_info['filename']}

Format your response as:
• Key point 1
• Key point 2
• Key point 3
etc.

Content:
{combined_content}

Key Points:"""

            summary = self.llm_service.generate_response(
                prompt=key_points_prompt,
                max_tokens=self.summary_max_tokens,
                temperature=0.1
            )
            
            return summary or "Unable to generate key points."
            
        except Exception as e:
            logger.error(f"Error generating key points: {e}")
            return "Error generating summary."
    
    def _generate_executive_summary(
        self, 
        chunks: List[Dict[str, Any]], 
        doc_info: Dict[str, Any]
    ) -> str:
        """Generate executive summary"""
        try:
            combined_content = "\n\n".join(chunk['content'] for chunk in chunks)
            
            if len(combined_content) > self.max_content_length:
                combined_content = combined_content[:self.max_content_length] + "..."
            
            exec_prompt = f"""Create an executive summary of this document suitable for business stakeholders.

Document: {doc_info['filename']}

Include:
- Overview and purpose
- Key findings or recommendations
- Business implications
- Action items or next steps (if applicable)

Keep it concise and focused on business value.

Content:
{combined_content}

Executive Summary:"""

            summary = self.llm_service.generate_response(
                prompt=exec_prompt,
                max_tokens=self.summary_max_tokens,
                temperature=0.1
            )
            
            return summary or "Unable to generate executive summary."
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return "Error generating summary."
    
    def _generate_multi_doc_summary(
        self, 
        chunks: List[Dict[str, Any]], 
        focus_topic: Optional[str]
    ) -> str:
        """Generate multi-document summary"""
        try:
            # Group chunks by document
            doc_groups = {}
            for chunk in chunks:
                doc_id = chunk['document_id']
                if doc_id not in doc_groups:
                    doc_groups[doc_id] = []
                doc_groups[doc_id].append(chunk)
            
            # Create content sections
            content_sections = []
            for doc_id, doc_chunks in doc_groups.items():
                doc_content = "\n".join(chunk['content'] for chunk in doc_chunks[:3])
                filename = doc_chunks[0].get('document_filename', f'Document {doc_id}')
                content_sections.append(f"=== {filename} ===\n{doc_content}")
            
            combined_content = "\n\n".join(content_sections)
            
            if len(combined_content) > self.max_content_length:
                combined_content = combined_content[:self.max_content_length] + "..."
            
            # Build multi-doc prompt
            if focus_topic:
                multi_prompt = f"""Summarize the following content from multiple documents, focusing on: {focus_topic}

Provide:
1. Overview of how each document relates to {focus_topic}
2. Key insights and findings
3. Connections and patterns across documents
4. Synthesized conclusions

Content from multiple documents:
{combined_content}

Multi-Document Summary:"""
            else:
                multi_prompt = f"""Summarize and synthesize the following content from multiple documents.

Provide:
1. Overview of main topics across all documents
2. Key themes and patterns
3. Important insights and findings
4. Connections between documents
5. Overall conclusions

Content from multiple documents:
{combined_content}

Multi-Document Summary:"""

            summary = self.llm_service.generate_response(
                prompt=multi_prompt,
                max_tokens=self.summary_max_tokens,
                temperature=0.1
            )
            
            return summary or "Unable to generate multi-document summary."
            
        except Exception as e:
            logger.error(f"Error generating multi-document summary: {e}")
            return "Error generating summary."
    
    def _generate_document_comparison(
        self, 
        doc_contents: List[Dict[str, str]], 
        aspects: Optional[List[str]]
    ) -> str:
        """Generate document comparison"""
        try:
            # Build comparison content
            comparison_content = []
            for i, doc in enumerate(doc_contents):
                comparison_content.append(f"Document {i+1}: {doc['title']}\n{doc['content']}")
            
            combined_content = "\n\n---\n\n".join(comparison_content)
            
            if len(combined_content) > self.max_content_length:
                combined_content = combined_content[:self.max_content_length] + "..."
            
            # Build comparison prompt
            if aspects:
                aspects_text = ", ".join(aspects)
                comparison_prompt = f"""Compare the following documents focusing on these aspects: {aspects_text}

Provide:
1. Similarities between the documents
2. Key differences and contrasts
3. Unique points in each document
4. Overall comparative analysis

Documents to compare:
{combined_content}

Comparative Analysis:"""
            else:
                comparison_prompt = f"""Compare and contrast the following documents.

Provide:
1. Main similarities across documents
2. Key differences and unique aspects
3. Complementary information
4. Conflicting viewpoints (if any)
5. Overall comparative insights

Documents to compare:
{combined_content}

Comparative Analysis:"""

            comparison = self.llm_service.generate_response(
                prompt=comparison_prompt,
                max_tokens=self.summary_max_tokens,
                temperature=0.1
            )
            
            return comparison or "Unable to generate document comparison."
            
        except Exception as e:
            logger.error(f"Error generating document comparison: {e}")
            return "Error generating comparison."
    
    def _generate_key_insights(
        self, 
        chunks: List[Dict[str, Any]], 
        insight_type: str
    ) -> str:
        """Generate key insights from content"""
        try:
            combined_content = "\n\n".join(chunk['content'] for chunk in chunks)
            
            if len(combined_content) > self.max_content_length:
                combined_content = combined_content[:self.max_content_length] + "..."
            
            # Build insights prompt based on type
            if insight_type == "technical":
                insights_prompt = f"""Extract key technical insights from the following content.

Focus on:
- Technical approaches and methodologies
- Implementation details and solutions
- Technical challenges and how they're addressed
- Best practices and recommendations
- Performance considerations

Content:
{combined_content}

Technical Insights:"""
            elif insight_type == "business":
                insights_prompt = f"""Extract key business insights from the following content.

Focus on:
- Business opportunities and challenges
- Market considerations
- Strategic implications
- Financial aspects
- Competitive advantages

Content:
{combined_content}

Business Insights:"""
            elif insight_type == "research":
                insights_prompt = f"""Extract key research insights from the following content.

Focus on:
- Research findings and results
- Methodological approaches
- Conclusions and implications
- Future research directions
- Evidence and data presented

Content:
{combined_content}

Research Insights:"""
            else:  # general
                insights_prompt = f"""Extract the most important insights and takeaways from the following content.

Focus on:
- Key findings and conclusions
- Important patterns or trends
- Notable observations
- Actionable information
- Significant implications

Content:
{combined_content}

Key Insights:"""

            insights = self.llm_service.generate_response(
                prompt=insights_prompt,
                max_tokens=self.summary_max_tokens,
                temperature=0.1
            )
            
            return insights or "Unable to generate insights."
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return "Error generating insights."
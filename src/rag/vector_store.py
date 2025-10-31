import logging
from typing import List, Dict, Any, Optional, Union
import uuid
import numpy as np
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, 
    Filter, FieldCondition, MatchValue,
    ScoredPoint, CollectionInfo, FilterSelector
)
from qdrant_client.http.exceptions import UnexpectedResponse

from config.config import settings
from src.utils.exceptions import VectorStoreError

logger = logging.getLogger(__name__)

class QdrantVectorStore:
    """Qdrant vector database integration"""
    
    def __init__(self):
        self.host = settings.QDRANT_HOST
        self.port = settings.QDRANT_PORT
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        # Use dimension from config as the single source of truth
        self.embedding_dim = int(settings.EMBEDDING_DIMENSION)
        self.api_key = settings.QDRANT_API_KEY
        
        self.client = None
        self._connect()
        self._initialize_collection()
    
    def _connect(self):
        """Connect to Qdrant server"""
        try:
            # Initialize Qdrant client
            if self.api_key:
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    api_key=self.api_key,
                    https=settings.QDRANT_USE_HTTPS,
                    timeout=10
                )
            else:
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    timeout=10
                )
            
            # Test connection
            collections = self.client.get_collections()
            logger.info(f"Connected to Qdrant at {self.host}:{self.port}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            raise VectorStoreError(f"Qdrant connection failed: {e}")
    
    def _initialize_collection(self):
        """Initialize or create collection"""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name in collection_names:
                logger.info(f"Collection '{self.collection_name}' already exists")
                
                # Get collection info
                collection_info = self.client.get_collection(self.collection_name)
                logger.info(f"Collection info: {collection_info.vectors_count} vectors")

                # Validate vector size matches embedding dimension
                try:
                    existing_dim = int(collection_info.config.params.vectors.size)
                except Exception:
                    existing_dim = None
                if existing_dim and existing_dim != self.embedding_dim:
                    msg = (
                        f"Qdrant collection vector size ({existing_dim}) does not match embedding dimension "
                        f"({self.embedding_dim}). Please recreate the collection or adjust configuration."
                    )
                    logger.error(msg)
                    raise VectorStoreError(msg)
            else:
                logger.info(f"Creating new collection: {self.collection_name}")
                self._create_collection()
            
            logger.info(f"Collection '{self.collection_name}' is ready")
            
        except Exception as e:
            logger.error(f"Error initializing collection: {e}")
            raise VectorStoreError(f"Collection initialization failed: {e}")
    
    def _create_collection(self):
        """Create Qdrant collection with vector configuration"""
        try:
            # Create collection with vector parameters
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE  # Cosine similarity
                )
            )
            
            logger.info(f"Collection '{self.collection_name}' created successfully")
            
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise VectorStoreError(f"Collection creation failed: {e}")
    
    def insert_embeddings(
        self,
        document_id: int,
        chunks_data: List[Dict[str, Any]],
        embeddings: np.ndarray
    ) -> List[str]:
        """
        Insert embeddings into Qdrant collection
        
        Args:
            document_id: Document ID
            chunks_data: List of chunk data dictionaries
            embeddings: Numpy array of embeddings
            
        Returns:
            List of generated point IDs
        """
        try:
            if len(chunks_data) != len(embeddings):
                raise VectorStoreError(
                    f"Chunks count ({len(chunks_data)}) != embeddings count ({len(embeddings)})"
                )
            
            if not chunks_data:
                logger.warning("No chunks to insert")
                return []
            
            logger.info(f"Inserting {len(chunks_data)} embeddings for document {document_id}")
            
            # Prepare points for insertion
            points = []
            point_ids = []
            current_time = datetime.now().isoformat()
            
            for i, (chunk, embedding) in enumerate(zip(chunks_data, embeddings)):
                chunk_index = int(chunk.get('chunk_index', i))
                deterministic_id = f"doc:{document_id}:chunk:{chunk_index}"
                point_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, deterministic_id)
                point_ids.append(str(point_uuid))
                
                # ✅ FIXED: Ensure vector is proper format
                vector_list = embedding.astype(np.float32).tolist()
                
                # Ensure chunk_index is present in chunk for payload prep
                if 'chunk_index' not in chunk or chunk.get('chunk_index') is None:
                    try:
                        chunk['chunk_index'] = chunk_index
                    except Exception:
                        pass

                # Prepare payload using centralized sanitizer
                payload = self._prepare_payload(document_id, chunk, current_time)
                
                point = PointStruct(
                    id=str(point_uuid),  # Qdrant client expects str for UUID identifiers
                    vector=vector_list,
                    payload=payload
                )
                points.append(point)
            
            # Insert in batches
            batch_size = 256
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                
                operation_info = self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True
                )
            
            logger.info(f"Successfully inserted {len(point_ids)} embeddings")
            return point_ids
            
        except Exception as e:
            logger.error(f"Error inserting embeddings: {e}")
            raise VectorStoreError(f"Embedding insertion failed: {e}")
    
    def _prepare_payload(self, document_id: int, chunk: Dict[str, Any], current_time: str) -> Dict[str, Any]:
        """Prepare and validate payload for Qdrant"""
        try:
            # ✅ FIXED: Ensure all payload values are JSON-serializable
            content = chunk.get('content', '')
            if isinstance(content, str):
                # Limit content length for Qdrant
                content = content[:30000]  # 30KB limit
            else:
                content = str(content)[:30000]
            
            metadata = chunk.get('metadata', {})
            if not isinstance(metadata, dict):
                metadata = {}
            
            # Clean metadata - remove any non-serializable values
            clean_metadata = {}
            for key, value in metadata.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    clean_metadata[key] = value
                elif isinstance(value, (list, dict)):
                    # Keep simple lists/dicts, convert complex ones to strings
                    try:
                        import json
                        json.dumps(value)  # Test if serializable
                        clean_metadata[key] = value
                    except:
                        clean_metadata[key] = str(value)
                else:
                    clean_metadata[key] = str(value)
            
            payload = {
                "document_id": int(document_id),  # Ensure it's int
                "chunk_index": int(chunk.get('chunk_index', 0)),  # Ensure it's int
                "content": content,
                "metadata": clean_metadata,
                "created_at": current_time
            }
            
            return payload
            
        except Exception as e:
            logger.error(f"Error preparing payload: {e}")
            # Return minimal safe payload
            return {
                "document_id": int(document_id),
                "chunk_index": 0,
                "content": str(chunk.get('content', ''))[:1000],
                "metadata": {},
                "created_at": current_time
            }
    
    def _diagnose_insertion_error(self, sample_point: PointStruct, error: Exception):
        """Diagnose what went wrong with point insertion"""
        logger.error("🔍 Diagnosing insertion error...")
        logger.error(f"Point ID type: {type(sample_point.id)}")
        logger.error(f"Point ID value: {sample_point.id}")
        logger.error(f"Vector type: {type(sample_point.vector)}")
        logger.error(f"Vector length: {len(sample_point.vector) if hasattr(sample_point.vector, '__len__') else 'unknown'}")
        logger.error(f"Payload keys: {list(sample_point.payload.keys()) if sample_point.payload else 'None'}")
        logger.error(f"Error: {error}")
        
        # Check vector format
        if hasattr(sample_point.vector, '__len__'):
            vector = sample_point.vector
            if len(vector) > 0:
                logger.error(f"First vector element type: {type(vector[0])}")
                logger.error(f"Vector sample: {vector[:3]}...")
    
    def search_similar(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        document_ids: Optional[List[int]] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            document_ids: Filter by document IDs
            min_score: Minimum similarity score
            
        Returns:
            List of search results
        """
        try:
            query_vector = query_embedding.astype(np.float32).tolist()
            
            # ✅ FIXED: Proper filter construction for multiple document IDs
            query_filter = None
            if document_ids:
                if len(document_ids) == 1:
                    query_filter = Filter(
                        must=[FieldCondition(key="document_id", match=MatchValue(value=document_ids[0]))]
                    )
                else:
                    # For multiple IDs, use should (OR condition)
                    query_filter = Filter(
                        should=[
                            FieldCondition(key="document_id", match=MatchValue(value=doc_id))
                            for doc_id in document_ids
                        ]
                    )
            
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=min_score,
                with_payload=True,
                with_vectors=False
            )
            
            # Process results
            results = []
            for result in search_results:
                if isinstance(result, ScoredPoint):
                    payload = result.payload or {}
                    processed_result = {
                        'id': str(result.id),
                        'document_id': payload.get('document_id'),
                        'chunk_index': payload.get('chunk_index'),
                        'content': payload.get('content', ''),
                        'metadata': payload.get('metadata', {}),
                        'score': float(result.score),
                        'created_at': payload.get('created_at')
                    }
                    results.append(processed_result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching embeddings: {e}")
            raise VectorStoreError(f"Search failed: {e}")
    
    def delete_by_document_id(self, document_id: int) -> int:
        """Delete all embeddings for a document"""
        try:
            # Create filter for document ID
            delete_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
            
            try:
                count_resp = self.client.count(
                    collection_name=self.collection_name,
                    count_filter=delete_filter,
                    exact=True
                )
                points_to_delete = int(getattr(count_resp, "count", 0) or 0)
            except Exception:
                points_to_delete = 0
                
            points_selector = FilterSelector(filter=delete_filter)
            
            # Delete points
            operation_info = self.client.delete(
                collection_name=self.collection_name,
                points_selector=points_selector,
                wait=True
            )
            
            deleted_count = points_to_delete if points_to_delete > 0 else (
                1 if operation_info.status == "completed" else 0
            )

            logger.info(f"Deletion operation completed for document {document_id}")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting embeddings for document {document_id}: {e}")
            raise VectorStoreError(f"Delete failed: {e}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            stats = {
                'collection_name': self.collection_name,
                'total_vectors': collection_info.vectors_count or 0,
                'indexed_vectors': collection_info.indexed_vectors_count or 0,
                'points_count': collection_info.points_count or 0,
                'vector_size': collection_info.config.params.vectors.size,
                'distance_metric': collection_info.config.params.vectors.distance.value,
                'status': collection_info.status.value
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {'error': str(e)}
    
    def health_check(self) -> bool:
        """Check if Qdrant connection is healthy"""
        try:
            # Simple health check - get collections
            collections = self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get detailed collection information"""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            
            return {
                'name': self.collection_name,
                'status': collection_info.status.value,
                'vectors_count': collection_info.vectors_count,
                'indexed_vectors_count': collection_info.indexed_vectors_count,
                'points_count': collection_info.points_count,
                'segments_count': collection_info.segments_count,
                'vector_size': collection_info.config.params.vectors.size,
                'distance': collection_info.config.params.vectors.distance.value,
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {'error': str(e)}

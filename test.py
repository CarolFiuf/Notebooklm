#!/usr/bin/env python3
"""
Quick system health check - run this for fast system verification
"""

def quick_health_check():
    print("🔍 Quick System Health Check")
    print("=" * 30)
    
    # 1. Database Check
    try:
        from src.utils.database import get_db_session, Document
        db = get_db_session()
        doc_count = db.query(Document).count()
        db.close()
        print(f"✅ Database: Connected ({doc_count} documents)")
    except Exception as e:
        print(f"❌ Database: Failed - {e}")
    
    # 2. Qdrant Check
    try:
        from src.rag.vector_store import QdrantVectorStore
        vs = QdrantVectorStore()
        stats = vs.get_collection_stats()
        vector_count = stats.get('total_vectors', 0)
        print(f"✅ Qdrant: Connected ({vector_count} vectors)")
    except Exception as e:
        print(f"❌ Qdrant: Failed - {e}")
    
    # 3. LLM Check
    try:
        from src.serving.llm_service import LlamaCppService
        llm = LlamaCppService()
        info = llm.get_model_info()
        if info.get('is_initialized'):
            print(f"✅ LLM: Ready ({info.get('model_size_mb', 0)}MB)")
        else:
            print(f"❌ LLM: Not initialized")
    except Exception as e:
        print(f"❌ LLM: Failed - {e}")
    
    # 4. Embedding Check
    try:
        from src.rag.embedding_service import EmbeddingService
        emb = EmbeddingService()
        # Quick test
        test_emb = emb.encode_single_text("test")
        print(f"✅ Embeddings: Working ({len(test_emb)} dim)")
    except Exception as e:
        print(f"❌ Embeddings: Failed - {e}")
    
    # 5. System Resources
    try:
        import psutil
        mem = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.1)
        print(f"📊 Resources: CPU {cpu}%, RAM {mem.percent}%")
    except:
        print("📊 Resources: Cannot check")

if __name__ == "__main__":
    quick_health_check()
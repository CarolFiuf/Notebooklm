#!/usr/bin/env python3
"""
Script để xóa tất cả dữ liệu trong PostgreSQL và Qdrant
Sử dụng khi muốn upload lại tài liệu với chunking strategy mới

Usage:
    python test.py
"""
import sys
sys.path.insert(0, '/Users/trung/notebooklm')

from src.utils.database import get_db_session, Document, DocumentChunk
from src.rag.vector_store import QdrantVectorStore
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def check_data():
    """Kiểm tra dữ liệu hiện tại"""
    logger.info("=" * 70)
    logger.info("🔍 KIỂM TRA DỮ LIỆU HIỆN TẠI")
    logger.info("=" * 70)

    # Check PostgreSQL
    db = get_db_session()
    try:
        doc_count = db.query(Document).count()
        chunk_count = db.query(DocumentChunk).count()

        logger.info(f"\n📊 PostgreSQL:")
        logger.info(f"  - Documents: {doc_count}")
        logger.info(f"  - Chunks: {chunk_count}")
    finally:
        db.close()

    # Check Qdrant
    try:
        vector_store = QdrantVectorStore()
        collection_info = vector_store.client.get_collection(
            collection_name=vector_store.collection_name
        )
        point_count = collection_info.points_count

        logger.info(f"\n📊 Qdrant:")
        logger.info(f"  - Collection: {vector_store.collection_name}")
        logger.info(f"  - Points: {point_count}")
    except Exception:
        logger.info(f"\n📊 Qdrant: Collection chưa tồn tại hoặc rỗng")

    logger.info("\n" + "=" * 70)
    return doc_count, chunk_count


def clear_postgres():
    """Xóa tất cả documents và chunks trong PostgreSQL"""
    logger.info("\n1️⃣  Đang xóa dữ liệu PostgreSQL...")

    db = get_db_session()
    try:
        # Xóa chunks trước (foreign key constraint)
        chunk_count = db.query(DocumentChunk).count()
        db.query(DocumentChunk).delete()
        logger.info(f"  ✓ Đã xóa {chunk_count} chunks")

        # Xóa documents
        doc_count = db.query(Document).count()
        db.query(Document).delete()
        logger.info(f"  ✓ Đã xóa {doc_count} documents")

        db.commit()
        logger.info("  ✅ PostgreSQL đã được xóa hoàn toàn")

    except Exception as e:
        db.rollback()
        logger.error(f"  ❌ Lỗi khi xóa PostgreSQL: {e}")
        raise
    finally:
        db.close()


def clear_qdrant():
    """Xóa tất cả embeddings trong Qdrant"""
    logger.info("\n2️⃣  Đang xóa dữ liệu Qdrant...")

    try:
        vector_store = QdrantVectorStore()

        # Kiểm tra collection có tồn tại không
        try:
            collection_info = vector_store.client.get_collection(
                collection_name=vector_store.collection_name
            )
            point_count = collection_info.points_count
            logger.info(f"  → Tìm thấy {point_count} points")

            # Xóa collection
            vector_store.client.delete_collection(
                collection_name=vector_store.collection_name
            )
            logger.info(f"  ✓ Đã xóa collection '{vector_store.collection_name}'")

            # Tạo lại collection rỗng
            vector_store._create_collection()
            logger.info(f"  ✓ Đã tạo lại collection rỗng")
            logger.info("  ✅ Qdrant đã được xóa hoàn toàn")

        except Exception:
            logger.info("  → Collection chưa tồn tại, không cần xóa")

    except Exception as e:
        logger.error(f"  ❌ Lỗi khi xóa Qdrant: {e}")
        raise


def main():
    """Main function"""
    try:
        # Kiểm tra dữ liệu hiện tại
        doc_count, chunk_count = check_data()

        # Nếu không có dữ liệu, thoát
        if doc_count == 0 and chunk_count == 0:
            logger.info("\n✅ Database đã rỗng, không cần xóa.")
            return

        # Xác nhận
        logger.info("\n⚠️  BẠN SẮP XÓA TẤT CẢ DỮ LIỆU!")
        logger.info(f"   - {doc_count} documents")
        logger.info(f"   - {chunk_count} chunks")
        logger.info(f"   - Tất cả embeddings trong Qdrant")

        response = input("\n👉 Bạn có chắc chắn muốn tiếp tục? (yes/no): ")

        if response.lower() not in ['yes', 'y']:
            logger.info("\n❌ Đã hủy.")
            return

        # Xóa dữ liệu
        logger.info("\n" + "=" * 70)
        logger.info("🗑️  BẮT ĐẦU XÓA DỮ LIỆU")
        logger.info("=" * 70)

        clear_postgres()
        clear_qdrant()

        # Xác nhận đã xóa
        logger.info("\n" + "=" * 70)
        logger.info("✅ HOÀN TẤT - DỮ LIỆU ĐÃ ĐƯỢC XÓA!")
        logger.info("=" * 70)
        logger.info("\nBạn có thể upload lại tài liệu với chunking strategy mới.")

    except KeyboardInterrupt:
        logger.info("\n\n❌ Đã hủy bởi người dùng.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Lỗi: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

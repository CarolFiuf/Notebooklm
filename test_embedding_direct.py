#!/usr/bin/env python3
"""Test FPT embedding API with custom FPTEmbeddings class"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.fpt_embeddings import FPTEmbeddings
from config.settings import settings

print("="*80)
print("TESTING FPT EMBEDDING API WITH CUSTOM CLASS")
print("="*80)

# Test 1: Simple string embedding
print("\n[Test 1] Testing FPTEmbeddings (NO tokenization)...")
embeddings = FPTEmbeddings(
    model=settings.EVAL_EMBEDDING_MODEL,
    api_key=settings.EVAL_EMBEDDING_API_KEY,
    base_url=settings.EVAL_EMBEDDING_BASE_URL
)

test_text = "Luật Đường sắt quy định về hoạt động đường sắt"
print(f"Input text: {test_text}")
print(f"Input type: {type(test_text)}")

try:
    # Try to embed the text
    result = embeddings.embed_query(test_text)
    print(f"✅ SUCCESS! Got embedding vector of length: {len(result)}")
    print(f"First 5 values: {result[:5]}")
except Exception as e:
    print(f"❌ FAILED!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Batch embedding
print("\n[Test 2] Testing with batch of texts...")
test_texts = [
    "Luật Đường sắt",
    "Điều 17 quy định",
    "Phạm vi điều chỉnh"
]

try:
    results = embeddings.embed_documents(test_texts)
    print(f"✅ SUCCESS! Got {len(results)} embeddings")
    for i, r in enumerate(results):
        print(f"  Text {i+1}: vector length {len(r)}")
except Exception as e:
    print(f"❌ FAILED!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)

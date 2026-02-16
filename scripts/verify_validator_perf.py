import sys
import os
import time
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.validator import UniquenessValidator, HAS_SENTENCE_TRANSFORMERS
from core.models import GeneratorConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_validator():
    print(f"Has Sentence Transformers: {HAS_SENTENCE_TRANSFORMERS}")
    
    config = GeneratorConfig(model_id="test-model", similarity_threshold=0.8)
    validator = UniquenessValidator(config)
    
    if not validator.has_transformers:
        print("Skipping Semantic Test (Library not found)")
        return

    # 1. Commit first item
    text1 = "The quick brown fox jumps over the lazy dog."
    print(f"Committing: '{text1}'")
    validator.commit(text1)
    
    if validator.embeddings is None:
        print("ERROR: Embeddings not initialized after commit!")
        sys.exit(1)
    
    print(f"Embeddings shape: {validator.embeddings.shape}")
    
    # 2. Check duplicate (should be rejected by hash)
    print(f"Checking duplicate hash: '{text1}'")
    if validator.is_unique(text1):
        print("ERROR: Exact duplicate was accepted!")
        sys.exit(1)
    else:
        print("SUCCESS: Exact duplicate rejected.")

    # 3. Check semantic duplicate
    text2 = "A fast brown fox leaps over a sleepy dog."
    print(f"Checking semantic duplicate: '{text2}'")
    
    start_time = time.time()
    is_unique = validator.is_unique(text2)
    end_time = time.time()
    
    print(f"Check time: {(end_time - start_time)*1000:.2f}ms")
    
    if is_unique:
        print("ERROR: Semantic duplicate was accepted!")
        sys.exit(1)
    else:
        print("SUCCESS: Semantic duplicate rejected.")

    # 4. Commit second item
    print(f"Committing: '{text2}' (forced commit to test cache growth)")
    validator.commit(text2)
    print(f"Embeddings shape: {validator.embeddings.shape}")
    
    if validator.embeddings.shape[0] != 2:
        print(f"ERROR: Expected 2 embeddings, got {validator.embeddings.shape[0]}")
        sys.exit(1)

    print("ALL TESTS PASSED")

if __name__ == "__main__":
    test_validator()

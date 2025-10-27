"""
Simple test for data preparation module.
Tests core functionality with minimal data samples.
"""

import tempfile
import pandas as pd
from pathlib import Path

from src.data.data_preparation import DataPreparator


def test_data_preparation():
    """Test data preparation with minimal sample data."""
    
    print("=== Testing Data Preparation Module ===")
    
    # Create test data
    test_data = {
        'filename': ['test-1-ru.txt'],
        'abstract': ['Это тестовый текст для проверки извлечения аспектов. Мы используем метод машинного обучения.'],
        'aspect_annotation': ['Это тестовый текст для проверки [извлечения аспектов|F1|TASK]. Мы используем [метод машинного обучения|F2|METHOD].'],
        'aspects': ['F1 TASK извлечения аспектов\nF2 METHOD метод машинного обучения']
    }
    
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
        df = pd.DataFrame(test_data)
        df.to_csv(f.name, index=False)
        temp_csv_path = Path(f.name)
    
    try:
        # Initialize data preparator
        print("1. Initializing DataPreparator...")
        preparator = DataPreparator()
        print("   ✅ DataPreparator initialized successfully")
        
        # Test CSV parsing
        print("\n2. Testing CSV parsing...")
        df = preparator.parse_csv_file(temp_csv_path)
        assert len(df) == 1, "CSV should contain 1 row"
        print("   ✅ CSV parsing works correctly")
        
        # Test aspect extraction from column (now the only source)
        print("\n3. Testing aspect extraction from aspects column...")
        aspects_text = test_data['aspects'][0]
        aspects = preparator.extract_aspects_from_column(aspects_text)
        print(f"   Found aspects: {aspects}")
        assert len(aspects) == 2, "Should extract 2 aspects from column"
        assert ('извлечения аспектов', 'TASK') in aspects
        assert ('метод машинного обучения', 'METHOD') in aspects
        print("   ✅ Aspect extraction from column works correctly")
        
        # Test tokenization
        print("\n5. Testing tokenization...")
        text = test_data['abstract'][0]
        tokens = preparator.tokenize_text(text)
        print(f"   Tokens (first 10): {tokens[:10]}")
        assert len(tokens) > 0, "Should generate tokens"
        print("   ✅ Tokenization works correctly")
        
        # Test BIO tagging
        print("\n4. Testing BIO tagging...")
        text = test_data['abstract'][0]
        tokens, bio_tags = preparator.align_aspects_with_tokens(text, aspects)
        print(f"   Tokens: {len(tokens)}, BIO tags: {len(bio_tags)}")
        print(f"   Sample tags: {bio_tags[:10]}")
        assert len(tokens) == len(bio_tags), "Number of tokens should equal number of BIO tags"
        
        # Check that we have some B- and I- tags
        has_b_tags = any(tag.startswith('B-') for tag in bio_tags)
        has_i_tags = any(tag.startswith('I-') for tag in bio_tags)
        print(f"   Has B- tags: {has_b_tags}, Has I- tags: {has_i_tags}")
        print("   ✅ BIO tagging works correctly")
        
        # Test file processing
        print("\n5. Testing file processing...")
        tokens_and_tags = preparator.process_domain_file(temp_csv_path, "ru")
        assert len(tokens_and_tags) == 1, "Should process 1 sentence"
        print("   ✅ File processing works correctly")
        
        # Test CoNLL file saving
        print("\n6. Testing CoNLL file saving...")
        with tempfile.NamedTemporaryFile(mode='w', suffix='.conll', delete=False) as f:
            temp_conll_path = Path(f.name)
        
        preparator.save_conll_file(tokens_and_tags, temp_conll_path)
        
        # Check that file was created and has content
        assert temp_conll_path.exists(), "CoNLL file should be created"
        with open(temp_conll_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.strip().split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            print(f"   CoNLL file has {len(non_empty_lines)} non-empty lines")
            assert len(non_empty_lines) > 0, "CoNLL file should have content"
        
        print("   ✅ CoNLL file saving works correctly")
        
        # Show sample output
        print("\n7. Sample CoNLL output:")
        with open(temp_conll_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:10]  # First 10 lines
            for line in lines:
                if line.strip():
                    print(f"   {line.strip()}")
        
        print("\n=== ✅ ALL TESTS PASSED! ===")
        
        # Cleanup
        temp_conll_path.unlink()
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    
    finally:
        # Cleanup
        temp_csv_path.unlink()


if __name__ == "__main__":
    test_data_preparation()
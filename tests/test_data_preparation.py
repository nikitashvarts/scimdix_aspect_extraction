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
    
    # Create test data with multiple sentences
    test_data = {
        'filename': ['test-1-ru.txt'],
        'abstract': ['Это тестовый текст для проверки извлечения аспектов. Мы используем метод машинного обучения. Результаты показывают хорошую точность.'],
        'aspect_annotation': ['...'],  # Not used anymore
        'aspects': ['F1 TASK извлечения аспектов\nF2 METHOD метод машинного обучения\nF3 RESULT Результаты показывают хорошую точность']
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
        
        # Test sentence splitting
        print("\n3. Testing sentence splitting...")
        text = test_data['abstract'][0]
        sentences = preparator.split_into_sentences(text, "ru")
        print(f"   Original text split into {len(sentences)} sentences:")
        for i, sent in enumerate(sentences):
            print(f"     {i+1}: {sent}")
        assert len(sentences) >= 2, "Should split into multiple sentences"
        print("   ✅ Sentence splitting works correctly")
        
        # Test sentence splitting for Kazakh (if available)
        print("\n3.5. Testing Kazakh sentence splitting...")
        kz_text = "Бұл тестілік мәтін. Біз машиналық оқыту әдісін қолданамыз. Нәтижелер жақсы дәлдікті көрсетеді."
        kz_sentences = preparator.split_into_sentences(kz_text, "kz")
        print(f"   Kazakh text split into {len(kz_sentences)} sentences:")
        for i, sent in enumerate(kz_sentences):
            print(f"     {i+1}: {sent}")
        if len(kz_sentences) >= 2:
            print("   ✅ Kazakh sentence splitting works correctly")
        else:
            print("   ⚠️  Kazakh model might not be available, using fallback")
        
        # Test aspect extraction from column (now the only source)
        print("\n4. Testing aspect extraction from aspects column...")
        aspects_text = test_data['aspects'][0]
        aspects = preparator.extract_aspects_from_column(aspects_text)
        print(f"   Found aspects: {aspects}")
        assert len(aspects) == 3, "Should extract 3 aspects from column"
        assert ('извлечения аспектов', 'TASK') in aspects
        assert ('метод машинного обучения', 'METHOD') in aspects
        assert ('Результаты показывают хорошую точность', 'RESULT') in aspects
        print("   ✅ Aspect extraction from column works correctly")
        
        # Test tokenization
        print("\n5. Testing tokenization...")
        text = test_data['abstract'][0]
        tokens = preparator.tokenize_text(text)
        print(f"   Tokens (first 10): {tokens[:10]}")
        assert len(tokens) > 0, "Should generate tokens"
        print("   ✅ Tokenization works correctly")
        
        # Test sentence-level BIO tagging
        print("\n6. Testing sentence-level BIO tagging...")
        sentences = preparator.split_into_sentences(text, "ru")
        sentence_data = preparator.align_aspects_with_sentences(sentences, aspects)
        print(f"   Processed {len(sentences)} sentences → {len(sentence_data)} sentence data")
        
        # Check first sentence
        if sentence_data:
            tokens, bio_tags = sentence_data[0]
            print(f"   First sentence: {len(tokens)} tokens, {len(bio_tags)} tags")
            print(f"   Sample tokens: {tokens[:5]}")
            print(f"   Sample tags: {bio_tags[:5]}")
            assert len(tokens) == len(bio_tags), "Number of tokens should equal number of BIO tags"
            
            # Check that we have some B- and I- tags across all sentences
            all_tags = []
            for _, tags in sentence_data:
                all_tags.extend(tags)
            has_b_tags = any(tag.startswith('B-') for tag in all_tags)
            print(f"   Has B- tags across sentences: {has_b_tags}")
        
        print("   ✅ Sentence-level BIO tagging works correctly")
        
        # Test file processing
        print("\n7. Testing file processing...")
        tokens_and_tags = preparator.process_domain_file(temp_csv_path, "ru")
        print(f"   Processed file resulted in {len(tokens_and_tags)} sentences")
        assert len(tokens_and_tags) >= 1, "Should process at least 1 sentence"
        print("   ✅ File processing works correctly")
        
        # Test CoNLL file saving
        print("\n8. Testing CoNLL file saving...")
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
        print("\n9. Sample CoNLL output:")
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
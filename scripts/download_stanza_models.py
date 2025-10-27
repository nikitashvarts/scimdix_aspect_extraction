#!/usr/bin/env python3
"""
Script to download required Stanza models for sentence segmentation.
Run this script once to download models before using the data preparation pipeline.
"""

import stanza

def download_models():
    """Download required Stanza models."""
    
    print("Downloading Stanza models for sentence segmentation...")
    print("=" * 50)
    
    models_to_download = [
        ("ru", "Russian"),
        ("kk", "Kazakh")
    ]
    
    successful_downloads = []
    failed_downloads = []
    
    for model_code, model_name in models_to_download:
        try:
            print(f"\nDownloading {model_name} model ({model_code})...")
            stanza.download(model_code, processors='tokenize')
            print(f"✅ {model_name} model downloaded successfully")
            successful_downloads.append(model_name)
        except Exception as e:
            print(f"❌ Failed to download {model_name} model: {e}")
            failed_downloads.append(model_name)
    
    print("\n" + "=" * 50)
    print("Download Summary:")
    
    if successful_downloads:
        print(f"✅ Successfully downloaded: {', '.join(successful_downloads)}")
    
    if failed_downloads:
        print(f"❌ Failed to download: {', '.join(failed_downloads)}")
        print("Note: The system will fallback to regex-based sentence splitting for failed models.")
    
    if successful_downloads:
        print("\n🎉 You can now run the data preparation pipeline!")
    else:
        print("\n⚠️  No models downloaded. The system will use regex-based fallback.")
        
    print("\nTo test the setup, run:")
    print("  PYTHONPATH=. python tests/test_data_preparation.py")

if __name__ == "__main__":
    download_models()
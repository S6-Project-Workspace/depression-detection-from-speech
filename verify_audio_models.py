#!/usr/bin/env python3
"""
Quick verification script for audio models before competition evaluation.
"""

import os
import torch
from pathlib import Path

def check_model_files():
    """Check which audio model files exist."""
    print("🔍 Checking Audio Model Files...")
    print("="*50)
    
    audio_models = {
        "ECAPA-TDNN (Final)": "final_models/ecapa_audio_fold0.pt",
        "SSL WavLM (Final)": "final_models/ssl_audio_fold0.pt",
        "ECAPA Fold 0": "outputs/training_with_progress/checkpoints/ecapa_fold_0/best_model.pt",
        "ECAPA Fold 1": "outputs/training_with_progress/checkpoints/ecapa_fold_1/best_model.pt", 
        "ECAPA Fold 2": "outputs/training_with_progress/checkpoints/ecapa_fold_2/best_model.pt",
        "SSL Fold 0": "outputs/training_with_progress/checkpoints/ssl_fold_0/best_model.pt",
        "SSL Fold 1": "outputs/training_with_progress/checkpoints/ssl_fold_1/best_model.pt",
        "SSL Fold 2": "outputs/training_with_progress/checkpoints/ssl_fold_2/best_model.pt",
    }
    
    available_models = []
    missing_models = []
    
    for name, path in audio_models.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024**2)
            print(f"✅ {name:<25} ({size_mb:.1f} MB)")
            available_models.append((name, path))
        else:
            print(f"❌ {name:<25} (NOT FOUND)")
            missing_models.append((name, path))
    
    print(f"\n📊 Summary: {len(available_models)}/{len(audio_models)} models available")
    
    return available_models, missing_models

def check_test_data():
    """Check test set data availability."""
    print("\n🔍 Checking Test Set Data...")
    print("="*50)
    
    test_set_dir = "/Users/keerthivasan/NLP/Test Set"
    required_files = [
        "Tamil_GT.xlsx",
        "Malayalam_GT.xlsx", 
        "Test-set-tamil.zip",
        "Test_set_mal.zip"
    ]
    
    all_present = True
    for filename in required_files:
        filepath = os.path.join(test_set_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024**2)
            print(f"✅ {filename:<20} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {filename:<20} (NOT FOUND)")
            all_present = False
    
    return all_present

def main():
    print("ACL 2026 Competition - Audio Model Verification")
    print("="*60)
    
    # Check models
    available_models, missing_models = check_model_files()
    
    # Check test data
    test_data_ready = check_test_data()
    
    print("\n🎯 EVALUATION READINESS:")
    print("="*30)
    
    if available_models and test_data_ready:
        print("✅ Ready for evaluation!")
        print(f"📊 {len(available_models)} audio models will be evaluated")
        print("🏆 Competition test set data is available")
        print("\n🚀 Run: python evaluate_audio_models_competition.py")
    else:
        print("❌ Not ready for evaluation")
        if not available_models:
            print("   - No audio models found")
        if not test_data_ready:
            print("   - Test set data missing")
    
    print("\n📋 Available Models for Competition:")
    for name, path in available_models:
        model_type = "ECAPA-TDNN" if "ecapa" in name.lower() else "SSL/WavLM"
        print(f"   • {name} ({model_type})")

if __name__ == "__main__":
    main()
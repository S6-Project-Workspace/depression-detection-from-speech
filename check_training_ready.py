"""
Pre-Training Checklist Script

Checks if all requirements are met before starting NLP multimodal training.
"""

import os
import sys
from pathlib import Path
import json

def check_mark(condition):
    return "✅" if condition else "❌"

def main():
    print("\n" + "="*60)
    print("NLP Multimodal Training - Pre-Flight Check")
    print("="*60 + "\n")
    
    all_good = True
    
    # 1. Check Python version
    print("1. Python Environment")
    python_version = sys.version_info
    python_ok = python_version.major == 3 and python_version.minor >= 8
    print(f"   {check_mark(python_ok)} Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    if not python_ok:
        print("      ⚠️  Python 3.8+ required")
        all_good = False
    
    # 2. Check required packages
    print("\n2. Required Packages")
    required_packages = [
        'torch',
        'transformers',
        'sklearn',
        'numpy',
        'tqdm'
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} - NOT INSTALLED")
            all_good = False
    
    # 3. Check data directories
    print("\n3. Data Directories")
    
    data_dir = Path("NLP Dataset")
    data_exists = data_dir.exists()
    print(f"   {check_mark(data_exists)} Audio dataset: {data_dir}")
    if not data_exists:
        print("      ⚠️  Audio dataset not found")
        all_good = False
    
    # Check subdirectories
    if data_exists:
        tamil_dir = data_dir / "Tamil"
        malayalam_dir = data_dir / "Malayalam"
        tamil_ok = tamil_dir.exists()
        malayalam_ok = malayalam_dir.exists()
        print(f"   {check_mark(tamil_ok)} Tamil dataset")
        print(f"   {check_mark(malayalam_ok)} Malayalam dataset")
        if not (tamil_ok and malayalam_ok):
            all_good = False
    
    # 4. Check transcripts
    print("\n4. Transcripts")
    
    transcript_dir = Path("transcripts")
    transcript_exists = transcript_dir.exists()
    print(f"   {check_mark(transcript_exists)} Transcript directory: {transcript_dir}")
    
    if transcript_exists:
        tamil_transcripts = transcript_dir / "tamil"
        malayalam_transcripts = transcript_dir / "malayalam"
        
        tamil_trans_ok = tamil_transcripts.exists()
        malayalam_trans_ok = malayalam_transcripts.exists()
        
        print(f"   {check_mark(tamil_trans_ok)} Tamil transcripts")
        print(f"   {check_mark(malayalam_trans_ok)} Malayalam transcripts")
        
        # Count transcript files
        if tamil_trans_ok:
            tamil_files = list(tamil_transcripts.glob("*.json"))
            print(f"      📄 {len(tamil_files)} Tamil transcript files")
        
        if malayalam_trans_ok:
            malayalam_files = list(malayalam_transcripts.glob("*.json"))
            print(f"      📄 {len(malayalam_files)} Malayalam transcript files")
        
        if not (tamil_trans_ok and malayalam_trans_ok):
            print("      ⚠️  Run: python generate_transcripts.py")
            all_good = False
    else:
        print("      ⚠️  Transcripts not found. Run: python generate_transcripts.py")
        all_good = False
    
    # 5. Check model files
    print("\n5. Model Components")
    
    model_files = [
        'text_model.py',
        'text_preprocessing.py',
        'tfidf_classifier.py',
        'linguistic_analyzer.py',
        'multimodal_model.py',
        'multimodal_trainer.py',
        'enhanced_multimodal_model.py',
        'enhanced_multimodal_trainer.py',
        'ssl_model.py',
        'ecapa_model.py'
    ]
    
    for model_file in model_files:
        exists = Path(model_file).exists()
        print(f"   {check_mark(exists)} {model_file}")
        if not exists:
            all_good = False
    
    # 6. Check training script
    print("\n6. Training Scripts")
    
    training_script = Path("train_nlp_multimodal.py")
    script_exists = training_script.exists()
    print(f"   {check_mark(script_exists)} train_nlp_multimodal.py")
    if not script_exists:
        all_good = False
    
    start_script = Path("start_training.sh")
    start_exists = start_script.exists()
    print(f"   {check_mark(start_exists)} start_training.sh")
    
    # 7. Check output directory
    print("\n7. Output Directory")
    
    output_dir = Path("outputs")
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {output_dir}")
    else:
        print(f"   ✅ Exists: {output_dir}")
    
    # 8. Check GPU availability
    print("\n8. Hardware")
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available()
        
        if cuda_available:
            print(f"   ✅ CUDA GPU available")
            print(f"      🎮 {torch.cuda.get_device_name(0)}")
        elif mps_available:
            print(f"   ✅ Apple Silicon (MPS) available")
        else:
            print(f"   ⚠️  No GPU detected - training will use CPU (slower)")
        
        # Check memory
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        memory_ok = memory_gb >= 8
        print(f"   {check_mark(memory_ok)} RAM: {memory_gb:.1f} GB")
        if not memory_ok:
            print("      ⚠️  8GB+ RAM recommended")
    except ImportError:
        print("   ⚠️  Cannot check hardware (torch/psutil not installed)")
    
    # Final summary
    print("\n" + "="*60)
    if all_good:
        print("✅ ALL CHECKS PASSED - READY TO TRAIN!")
        print("="*60)
        print("\nTo start training, run:")
        print("  ./start_training.sh")
        print("\nOr manually:")
        print("  python train_nlp_multimodal.py --mode all --epochs 10")
        print("\nSee RUN_NLP_TRAINING.md for more options.")
    else:
        print("❌ SOME CHECKS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before training.")
        print("\nCommon fixes:")
        print("  - Install packages: pip install -r requirements.txt")
        print("  - Generate transcripts: python generate_transcripts.py")
        print("  - Check data directory path")
    print()


if __name__ == '__main__':
    main()

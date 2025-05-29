import json
import torch
from pathlib import Path

def preflight_check():
    """Pre-flight check before fine-tuning"""
    
    print("✈️  PRE-FLIGHT CHECK")
    print("="*25)
    
    checks_passed = 0
    total_checks = 7
    
    # 1. Check CUDA
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
        if gpu_memory >= 8:
            checks_passed += 1
        else:
            print(f"⚠️  GPU memory may be insufficient for batch training")
    else:
        print("❌ CUDA not available")
    
    # 2. Check dataset files
    audio_dir = Path("deep_house_dataset/processed_20s")
    if audio_dir.exists():
        audio_files = list(audio_dir.glob("*.wav"))
        print(f"✅ Audio files: {len(audio_files)}")
        checks_passed += 1
    else:
        print("❌ Audio directory not found")
    
    # 3. Check prompts
    prompts_file = Path("deep_house_dataset/deep_house_prompts.json")
    if prompts_file.exists():
        with open(prompts_file) as f:
            prompts = json.load(f)
        print(f"✅ Prompts: {len(prompts)}")
        checks_passed += 1
    else:
        print("❌ Prompts file not found")
    
    # 4. Check configs
    dataset_config = Path("configs/deep_house_dataset_config.json")
    model_config = Path("configs/deep_house_model_config.json")
    
    if dataset_config.exists() and model_config.exists():
        print(f"✅ Config files ready")
        checks_passed += 1
    else:
        print("❌ Config files missing")
    
    # 5. Check stable-audio-tools
    try:
        from stable_audio_tools import get_pretrained_model
        print("✅ stable-audio-tools imported")
        checks_passed += 1
    except ImportError:
        print("❌ stable-audio-tools not available")
    
    # 6. Check training script
    if Path("train.py").exists():
        print("✅ Training script found")
        checks_passed += 1
    else:
        print("❌ train.py not found")
    
    # 7. Check disk space
    import shutil
    free_space = shutil.disk_usage(".").free / 1e9
    if free_space > 5:
        print(f"✅ Disk space: {free_space:.1f}GB available")
        checks_passed += 1
    else:
        print(f"⚠️  Low disk space: {free_space:.1f}GB")
    
    print(f"\n�� READINESS: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed >= 6:
        print("🟢 READY FOR FINE-TUNING!")
        return True
    else:
        print("🟡 SOME ISSUES NEED ATTENTION")
        return False

if __name__ == "__main__":
    preflight_check()

import json
from pathlib import Path
import os

def generate_repository_compatible_config():
    """Generate config compatible with the existing repository structure"""
    print("⚙️  GENERATING REPOSITORY-COMPATIBLE CONFIG")
    print("="*50)
    
    # Check what the repository expects
    train_py_exists = os.path.exists("train.py")
    
    if train_py_exists:
        print("✅ Found train.py - analyzing expected config format...")
        
        # Try to find config patterns in train.py
        with open("train.py", 'r') as f:
            content = f.read()
            
        # Look for config patterns
        if "model_config" in content:
            print("✅ Repository expects model_config")
        if "dataset_config" in content:
            print("✅ Repository expects dataset_config")
    
    # Create dataset config compatible with AudioDirDataset
    dataset_config = {
        "dataset_type": "audio_dir",
        "path": "dataset/audio",
        "sample_rate": 44100,
        "sample_size": 1323000,  # 30 seconds at 44.1kHz
        "audio_channels": 2,
        "random_crop": False,  # We want full 30-second clips
        "normalize": True
    }
    
    # Create model config for 30-second fine-tuning
    model_config = {
        "model_type": "diffusion_cond",
        "sample_size": 1323000,  # 30 seconds
        "sample_rate": 44100,
        "audio_channels": 2,
        
        "model": {
            "pretrained_name": "stabilityai/stable-audio-open-1.0",
            "pretransform_ckpt_path": None
        },
        
        "training": {
            "learning_rate": 1e-5,  # Lower for fine-tuning
            "batch_size": 2,  # Conservative for 30-second clips
            "save_every": 1000,
            "sample_every": 2000,
            "demo_every": 5000
        }
    }
    
    # Save configs
    os.makedirs("configs", exist_ok=True)
    
    dataset_path = Path("configs/dataset_config.json")
    model_path = Path("configs/model_config.json")
    
    with open(dataset_path, 'w') as f:
        json.dump(dataset_config, f, indent=2)
    
    with open(model_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"✅ Created dataset config: {dataset_path}")
    print(f"✅ Created model config: {model_path}")
    
    # Show how to use with existing train.py
    print(f"\n🚀 TO USE WITH EXISTING REPOSITORY:")
    print(f"python train.py \\")
    print(f"  --dataset-config {dataset_path} \\")
    print(f"  --model-config {model_path} \\")
    print(f"  --name my_30s_music_model")
    
    return dataset_path, model_path

if __name__ == "__main__":
    generate_repository_compatible_config()

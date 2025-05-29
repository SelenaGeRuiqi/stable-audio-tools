import json
import os
from pathlib import Path
import shutil

def setup_deep_house_finetuning():
    """Setup fine-tuning configuration for the specific Deep House dataset"""
    
    print("🎯 SETTING UP DEEP HOUSE FINE-TUNING")
    print("="*45)
    
    # Paths
    dataset_dir = Path("deep_house_dataset")
    audio_dir = dataset_dir / "processed_20s"
    prompts_file = dataset_dir / "deep_house_prompts.json"
    
    # Create configs directory
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # 1. Create Dataset Configuration
    dataset_config = {
        "dataset_type": "audio_dir",
        "path": str(audio_dir),
        "sample_rate": 44100,
        "sample_size": 882000,  # 20 seconds * 44100 Hz
        "audio_channels": 2,
        "random_crop": False,  # We want full 20-second clips
        "normalize": True,
        "metadata": {
            "path": str(prompts_file),
            "prompt_key": "prompt"  # Key in the JSON file
        }
    }
    
    dataset_config_file = configs_dir / "deep_house_dataset_config.json"
    with open(dataset_config_file, 'w') as f:
        json.dump(dataset_config, f, indent=2)
    
    print(f"✅ Dataset config: {dataset_config_file}")
    
    # 2. Create Model Configuration for Fine-tuning
    model_config = {
        "model_type": "diffusion_cond",
        "sample_size": 882000,  # 20 seconds
        "sample_rate": 44100,
        "audio_channels": 2,
        
        "model": {
            "pretrained_name": "stabilityai/stable-audio-open-1.0",
            "conditioning": {
                "text": True,
                "timing": True
            }
        },
        
        "training": {
            "learning_rate": 1e-5,      # Conservative for fine-tuning
            "batch_size": 1,            # Start small for 20s clips
            "gradient_accumulation": 4,  # Effective batch size = 4
            "max_epochs": 50,           # Should be enough for 43 samples
            "save_every": 500,          # Save frequently
            "sample_every": 1000,       # Generate samples for monitoring
            "demo_every": 2000,         # Create demo samples
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "scheduler": "cosine"
        },
        
        "demo": {
            "demo_every": 2000,
            "demo_steps": 50,
            "demo_cfg_scale": 7.0,
            "demo_conditioning": [
                {
                    "prompt": "deep house music, 124 BPM, warm rolling bassline, lush atmospheric pads",
                    "seconds_start": 0,
                    "seconds_total": 20
                },
                {
                    "prompt": "deep house track, 123 BPM, organic bass groove, sophisticated harmonies",
                    "seconds_start": 0,
                    "seconds_total": 20
                },
                {
                    "prompt": "house music, 125 BPM, four-on-the-floor pattern, ethereal ambience",
                    "seconds_start": 0,
                    "seconds_total": 20
                }
            ]
        }
    }
    
    model_config_file = configs_dir / "deep_house_model_config.json"
    with open(model_config_file, 'w') as f:
        json.dump(model_config, f, indent=2)
    
    print(f"✅ Model config: {model_config_file}")
    
    # 3. Verify dataset structure
    print(f"\n📊 Dataset Verification:")
    
    # Check audio files
    audio_files = list(audio_dir.glob("*.wav"))
    print(f"  �� Audio files: {len(audio_files)}")
    
    # Check prompts
    if prompts_file.exists():
        with open(prompts_file, 'r') as f:
            prompts = json.load(f)
        print(f"  📝 Prompts: {len(prompts)}")
        
        # Check alignment
        audio_filenames = {f.name for f in audio_files}
        prompt_filenames = set(prompts.keys())
        
        aligned = audio_filenames.intersection(prompt_filenames)
        print(f"  ✅ Aligned files: {len(aligned)}")
        
        if len(aligned) != len(audio_files):
            missing_prompts = audio_filenames - prompt_filenames
            missing_audio = prompt_filenames - audio_filenames
            
            if missing_prompts:
                print(f"  ⚠️  Audio files without prompts: {len(missing_prompts)}")
            if missing_audio:
                print(f"  ⚠️  Prompts without audio: {len(missing_audio)}")
    
    # 4. Create training script
    training_script = f"""#!/bin/bash
# Deep House Fine-tuning Script

echo "🎵 Starting Deep House Fine-tuning..."
echo "Dataset: {len(audio_files)} tracks, 20 seconds each"
echo "Target: Improve Deep House generation quality"

# Setup Weights & Biases (optional)
# wandb login

# Start training
python train.py \\
    --dataset-config {dataset_config_file} \\
    --model-config {model_config_file} \\
    --name deep_house_finetuning \\
    --pretrained-ckpt-path stabilityai/stable-audio-open-1.0 \\
    --save-dir checkpoints/deep_house \\
    --batch-size 1 \\
    --accum-batches 4 \\
    --num-gpus 1

echo "✅ Fine-tuning complete!"
echo "Check checkpoints/deep_house/ for saved models"
"""
    
    script_file = Path("train_deep_house.sh")
    with open(script_file, 'w') as f:
        f.write(training_script)
    
    os.chmod(script_file, 0o755)  # Make executable
    print(f"✅ Training script: {script_file}")
    
    # 5. Create Python training script (alternative)
    python_script = f'''
import subprocess
import sys

def run_deep_house_training():
    """Run Deep House fine-tuning"""
    
    cmd = [
        sys.executable, "train.py",
        "--dataset-config", "{dataset_config_file}",
        "--model-config", "{model_config_file}", 
        "--name", "deep_house_finetuning",
        "--pretrained-ckpt-path", "stabilityai/stable-audio-open-1.0",
        "--save-dir", "checkpoints/deep_house",
        "--batch-size", "1",
        "--accum-batches", "4",
        "--num-gpus", "1"
    ]
    
    print("🎵 Starting Deep House Fine-tuning...")
    print(f"Command: {{' '.join(cmd)}}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {{e}}")
        return False

if __name__ == "__main__":
    run_deep_house_training()
'''
    
    python_script_file = Path("train_deep_house.py")
    with open(python_script_file, 'w') as f:
        f.write(python_script)
    
    print(f"✅ Python training script: {python_script_file}")
    
    # 6. Training recommendations
    print(f"\n🎯 FINE-TUNING RECOMMENDATIONS:")
    print(f"  📊 Dataset size: {len(audio_files)} samples is good for fine-tuning")
    print(f"  ⏱️  Expected training time: 2-4 hours")
    print(f"  💾 GPU memory needed: ~8-12GB")
    print(f"  📈 Monitor: Loss should decrease steadily")
    print(f"  🎵 Demo samples: Generated every 2000 steps")
    
    return {
        'dataset_config': dataset_config_file,
        'model_config': model_config_file,
        'training_script': script_file,
        'python_script': python_script_file,
        'num_samples': len(audio_files)
    }

def main():
    setup_info = setup_deep_house_finetuning()
    
    print(f"\n🚀 READY TO START FINE-TUNING!")
    print("="*35)
    print(f"Choose one of these methods:")
    print(f"")
    print(f"METHOD 1 - Bash script:")
    print(f"  ./train_deep_house.sh")
    print(f"")
    print(f"METHOD 2 - Python script:")
    print(f"  python train_deep_house.py")
    print(f"")
    print(f"METHOD 3 - Direct command:")
    print(f"  python train.py \\")
    print(f"    --dataset-config {setup_info['dataset_config']} \\")
    print(f"    --model-config {setup_info['model_config']} \\")
    print(f"    --name deep_house_finetuning \\")
    print(f"    --pretrained-ckpt-path stabilityai/stable-audio-open-1.0")
    
    print(f"\n📊 TRAINING MONITORING:")
    print(f"  • Watch GPU usage: nvidia-smi")
    print(f"  • Monitor loss in terminal output")
    print(f"  • Check demo samples in checkpoints/deep_house/")
    print(f"  • Training should take 2-4 hours")

if __name__ == "__main__":
    main()

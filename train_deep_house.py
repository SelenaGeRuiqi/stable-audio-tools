import subprocess
import sys

def run_deep_house_training():
    """Run Deep House fine-tuning"""
    
    cmd = [
        sys.executable, "train.py",
        "--dataset-config", "configs/deep_house_dataset_config.json",
        "--model-config", "configs/deep_house_model_config.json", 
        "--name", "deep_house_finetuning",
        "--pretrained-ckpt-path", "stabilityai/stable-audio-open-1.0",
        "--save-dir", "checkpoints/deep_house",
        "--batch-size", "1",
        "--accum-batches", "4",
        "--logger", "wandb"
    ]
    
    print("🎵 Starting Deep House Fine-tuning...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed: {e}")
        return False

if __name__ == "__main__":
    run_deep_house_training()

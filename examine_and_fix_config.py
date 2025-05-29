import json
from pathlib import Path
import os

def examine_existing_configs():
    """Look for existing working configs in the repository"""
    
    print("🔍 EXAMINING REPOSITORY FOR CONFIG EXAMPLES")
    print("="*45)
    
    # Look for example configs
    config_locations = [
        "stable_audio_tools/configs/",
        "configs/",
        "examples/",
        ".",
    ]
    
    found_configs = []
    
    for location in config_locations:
        if os.path.exists(location):
            for file in Path(location).glob("**/*.json"):
                if 'config' in file.name.lower():
                    found_configs.append(file)
                    print(f"📄 Found: {file}")
    
    # If we find configs, examine them
    if found_configs:
        print(f"\n📖 Examining first config: {found_configs[0]}")
        try:
            with open(found_configs[0], 'r') as f:
                example_config = json.load(f)
            
            print("📋 Config structure:")
            for key in example_config.keys():
                print(f"  • {key}")
                if key == "model" and isinstance(example_config[key], dict):
                    for subkey in example_config[key].keys():
                        print(f"    ├─ {subkey}")
        except Exception as e:
            print(f"❌ Error reading config: {e}")
    
    return found_configs

def create_correct_model_config():
    """Create model config based on stable-audio-open architecture"""
    
    print("\n🎯 CREATING CORRECT MODEL CONFIG")
    print("="*35)
    
    # Based on stable-audio-open-1.0 architecture
    correct_config = {
        "model_type": "diffusion_cond",
        "sample_size": 882000,  # 20 seconds * 44100
        "sample_rate": 44100,
        "audio_channels": 2,
        
        # The key missing piece - this should match the pretrained model structure
        "model": {
            "type": "dit",
            "config": {
                "io_channels": 64,
                "patch_size": 1,
                "embed_dim": 1536,
                "cond_token_dim": 768,
                "project_cond_tokens": False,
                "global_cond_dim": 1536,
                "depth": 24,
                "num_heads": 24,
                "mlp_ratio": 4.0,
                "cond_dropout_prob": 0.1,
                "double_layers": True,
                "sigma_min": 0.3,
                "sigma_max": 500,
                "sigma_data": 1.0
            }
        },
        
        # This might be the missing "diffusion" section
        "diffusion": {
            "type": "dit",
            "config": {
                "io_channels": 64,
                "patch_size": 1,
                "embed_dim": 1536,
                "cond_token_dim": 768,
                "project_cond_tokens": False,
                "global_cond_dim": 1536,
                "depth": 24,
                "num_heads": 24,
                "mlp_ratio": 4.0,
                "cond_dropout_prob": 0.1,
                "double_layers": True,
                "sigma_min": 0.3,
                "sigma_max": 500,
                "sigma_data": 1.0
            }
        },
        
        "conditioning": {
            "configs": [
                {
                    "id": "prompt",
                    "type": "t5",
                    "config": {
                        "t5_model_name": "google/t5-v1_1-base",
                        "max_length": 512,
                        "enable_grad": False
                    }
                },
                {
                    "id": "seconds_start",
                    "type": "number",
                    "config": {
                        "min_val": 0,
                        "max_val": 512
                    }
                },
                {
                    "id": "seconds_total", 
                    "type": "number",
                    "config": {
                        "min_val": 0,
                        "max_val": 512
                    }
                }
            ],
            "cond_dropout_prob": 0.1
        },
        
        "training": {
            "learning_rate": 1e-5,
            "batch_size": 1,
            "grad_accum_every": 4,
            "optimizer_configs": {
                "diffusion": {
                    "optimizer": {
                        "type": "AdamW",
                        "config": {
                            "lr": 1e-5,
                            "betas": [0.9, 0.999],
                            "weight_decay": 0.01
                        }
                    }
                }
            }
        }
    }
    
    # Save this version
    config_file = Path("configs/corrected_model_config.json")
    with open(config_file, 'w') as f:
        json.dump(correct_config, f, indent=2)
    
    print(f"✅ Corrected config saved: {config_file}")
    return config_file

def try_minimal_approach():
    """Try the absolute minimal approach - let pretrained model handle everything"""
    
    print("\n🧪 TRYING MINIMAL APPROACH")
    print("="*30)
    
    # Ultra-minimal - just specify what's absolutely necessary
    minimal_config = {
        "model_type": "diffusion_cond",
        "sample_size": 882000,
        "sample_rate": 44100,
        "audio_channels": 2
    }
    
    minimal_file = Path("configs/ultra_minimal_config.json")
    with open(minimal_file, 'w') as f:
        json.dump(minimal_config, f, indent=2)
    
    print(f"✅ Ultra-minimal config: {minimal_file}")
    
    return minimal_file

def copy_from_pretrained():
    """Try to extract config from the pretrained model"""
    
    print("\n🔄 ATTEMPTING TO EXTRACT CONFIG FROM PRETRAINED MODEL")
    print("="*55)
    
    try:
        from stable_audio_tools import get_pretrained_model
        
        print("Loading pretrained model to examine config...")
        model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        
        print("📋 Pretrained model config keys:")
        for key in model_config.keys():
            print(f"  • {key}: {type(model_config[key])}")
        
        # Save the actual config from pretrained model
        pretrained_config_file = Path("configs/from_pretrained_config.json")
        with open(pretrained_config_file, 'w') as f:
            json.dump(model_config, f, indent=2)
        
        print(f"✅ Extracted config saved: {pretrained_config_file}")
        
        # Modify it for fine-tuning
        finetuning_config = model_config.copy()
        finetuning_config["sample_size"] = 882000  # 20 seconds
        
        # Add training config if not present
        if "training" not in finetuning_config:
            finetuning_config["training"] = {
                "learning_rate": 1e-5,
                "batch_size": 1
            }
        
        finetuning_config_file = Path("configs/finetuning_from_pretrained.json")  
        with open(finetuning_config_file, 'w') as f:
            json.dump(finetuning_config, f, indent=2)
        
        print(f"✅ Fine-tuning config saved: {finetuning_config_file}")
        
        return finetuning_config_file
        
    except Exception as e:
        print(f"❌ Error extracting from pretrained: {e}")
        return None

def main():
    print("🔧 COMPREHENSIVE CONFIG DEBUGGING")
    print("="*40)
    
    # Step 1: Look for existing configs
    existing_configs = examine_existing_configs()
    
    # Step 2: Extract from pretrained model 
    pretrained_config = copy_from_pretrained()
    
    # Step 3: Create our corrected version
    corrected_config = create_correct_model_config()
    
    # Step 4: Create minimal fallback
    minimal_config = try_minimal_approach()
    
    print(f"\n🎯 TESTING ORDER:")
    print("Try these configs in order:")
    
    configs_to_try = []
    
    if pretrained_config:
        configs_to_try.append(("Extracted from pretrained", pretrained_config))
    
    configs_to_try.extend([
        ("Ultra minimal", minimal_config),
        ("Corrected comprehensive", corrected_config)
    ])
    
    for i, (name, config_file) in enumerate(configs_to_try, 1):
        print(f"\n{i}. {name}:")
        print(f"   python train.py \\")
        print(f"     --dataset-config configs/deep_house_dataset_config.json \\")
        print(f"     --model-config {config_file} \\")
        print(f"     --name deep_house_test_{i} \\")
        print(f"     --pretrained-ckpt-path stabilityai/stable-audio-open-1.0")

if __name__ == "__main__":
    main()
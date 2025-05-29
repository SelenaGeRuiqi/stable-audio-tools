import os
import sys
sys.path.append('.')

from stable_audio_tools.data.dataset import *
import inspect

def examine_existing_datasets():
    """Examine what dataset classes are available"""
    print("🔍 EXAMINING EXISTING DATASET CLASSES")
    print("="*50)
    
    # Import the dataset module
    try:
        import stable_audio_tools.data.dataset as dataset_module
        
        # Get all classes from the module
        classes = [obj for name, obj in inspect.getmembers(dataset_module) 
                  if inspect.isclass(obj)]
        
        print(f"Found {len(classes)} dataset classes:")
        for cls in classes:
            print(f"  📂 {cls.__name__}")
            if hasattr(cls, '__doc__') and cls.__doc__:
                print(f"     {cls.__doc__.strip().split('.')[0]}")
        
        # Check for any example data
        data_dir = "stable_audio_tools/data/"
        if os.path.exists(data_dir):
            files = os.listdir(data_dir)
            print(f"\n📁 Files in data directory: {files}")
        
        # Look for dataset configuration examples
        config_files = []
        for root, dirs, files in os.walk('.'):
            for file in files:
                if 'config' in file.lower() and file.endswith(('.json', '.yaml', '.yml')):
                    config_files.append(os.path.join(root, file))
        
        if config_files:
            print(f"\n⚙️  Found config files:")
            for config in config_files[:5]:  # Show first 5
                print(f"  {config}")
        
    except Exception as e:
        print(f"❌ Error examining datasets: {e}")

def check_for_sample_data():
    """Check if there are any sample audio files"""
    print("\n🎵 CHECKING FOR SAMPLE AUDIO DATA")
    print("="*30)
    
    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    found_audio = []
    
    # Search recursively for audio files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if any(file.lower().endswith(ext) for ext in audio_extensions):
                found_audio.append(os.path.join(root, file))
    
    if found_audio:
        print(f"Found {len(found_audio)} audio files:")
        for audio in found_audio[:10]:  # Show first 10
            print(f"  🎵 {audio}")
        if len(found_audio) > 10:
            print(f"  ... and {len(found_audio) - 10} more")
    else:
        print("❌ No audio files found in repository")
    
    return found_audio

def examine_model_configs():
    """Look for existing model configuration examples"""
    print("\n⚙️  EXAMINING MODEL CONFIGURATIONS")
    print("="*35)
    
    # Common config locations
    config_paths = [
        "configs/",
        "stable_audio_tools/configs/",
        "examples/",
        "model_configs/"
    ]
    
    found_configs = []
    for config_path in config_paths:
        if os.path.exists(config_path):
            for file in os.listdir(config_path):
                if file.endswith(('.json', '.yaml', '.yml')):
                    full_path = os.path.join(config_path, file)
                    found_configs.append(full_path)
                    print(f"  📄 {full_path}")
    
    if not found_configs:
        print("❌ No model config files found")
        print("   This is why we need to create our own!")
    
    return found_configs

if __name__ == "__main__":
    examine_existing_datasets()
    sample_audio = check_for_sample_data()
    configs = examine_model_configs()
    
    print("\n" + "="*60)
    print("📋 SUMMARY:")
    print(f"✅ Dataset classes: Available (for loading data)")
    print(f"🎵 Sample audio: {'Found' if sample_audio else 'Not found'}")
    print(f"⚙️  Config files: {'Found' if configs else 'Not found'}")
    
    if not sample_audio and not configs:
        print("\n🎯 WHY WE NEED TO CREATE OUR OWN:")
        print("1. 📁 No sample audio data provided")
        print("2. ⚙️  No ready-to-use config files")
        print("3. 🎵 Need to bring your own dataset")
        print("4. ⚙️  Need to configure for your specific use case")
    
    print("\n🚀 WHAT THE REPOSITORY PROVIDES:")
    print("✅ Dataset loading classes")
    print("✅ Training infrastructure")
    print("✅ Model architectures")
    print("✅ Inference utilities")
    print("\n❓ WHAT YOU NEED TO PROVIDE:")
    print("📁 Your own audio data")
    print("📝 Text descriptions for your audio")
    print("⚙️  Configuration files for your specific task")

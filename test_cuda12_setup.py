import os
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import time

def test_sao_diffusion_setup():
    print("="*60)
    print("SAO_DIFFUSION SETUP TEST - CUDA 12.2")
    print("="*60)
    
    # Show current directory
    print(f"Working directory: {os.getcwd()}")
    print(f"Python executable: {os.sys.executable}")
    
    # System info
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("-"*60)
    
    try:
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("🧹 GPU cache cleared")
        
        # Load model
        print("Loading Stable Audio Open 1.0 model...")
        print("(This will download ~2-3GB on first run)")
        start_time = time.time()
        
        model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        
        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f} seconds")
        
        # Display model config
        sample_rate = model_config["sample_rate"]
        sample_size = model_config["sample_size"]
        max_duration = sample_size / sample_rate
        
        print(f"\n📊 Model Configuration:")
        print(f"  Sample rate: {sample_rate:,} Hz")
        print(f"  Sample size: {sample_size:,} samples")
        print(f"  Max duration: {max_duration:.1f} seconds")
        
        # Move to device
        if torch.cuda.is_available():
            print(f"\n🔄 Moving model to GPU...")
            initial_memory = torch.cuda.memory_allocated() / 1e9
        
        model = model.to(device)
        
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / 1e9
            print(f"📈 GPU memory: {final_memory:.2f} GB")
        
        # Test 30-second generation
        print(f"\n🎵 Testing 30-second music generation...")
        print("Prompt: 'energetic electronic music, 128 BPM, synthesizers'")
        
        conditioning = [{
            "prompt": "energetic electronic music, 128 BPM, synthesizers, bass, drums",
            "seconds_start": 0,
            "seconds_total": 30  # Your target duration
        }]
        
        print("🎛️  Generating with optimal settings...")
        gen_start = time.time()
        
        output = generate_diffusion_cond(
            model,
            steps=100,  # High quality
            cfg_scale=7.0,
            conditioning=conditioning,
            sample_size=sample_size,
            sigma_min=0.3,
            sigma_max=500,
            sampler_type="dpmpp-3m-sde",
            device=device
        )
        
        gen_time = time.time() - gen_start
        print(f"⚡ Generation completed in {gen_time:.1f} seconds")
        
        # Process and save output
        print("🔄 Processing audio...")
        output = rearrange(output, "b d n -> d (b n)")
        output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
        
        # Save with descriptive filename
        output_file = "sao_test_30s_generation.wav"
        torchaudio.save(output_file, output, sample_rate)
        
        # Verify results
        file_size = os.path.getsize(output_file) / (1024*1024)
        actual_duration = output.shape[1] / sample_rate
        
        print("="*60)
        print("🎉 SETUP TEST SUCCESSFUL!")
        print("="*60)
        print(f"✅ Generated: {output_file}")
        print(f"✅ Duration: {actual_duration:.2f} seconds")
        print(f"✅ File size: {file_size:.2f} MB")
        print(f"✅ Channels: {output.shape[0]} (stereo)")
        print(f"⚡ Generation speed: {actual_duration/gen_time:.2f}x realtime")
        
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1e9
            print(f"📊 Peak GPU memory: {peak_memory:.2f} GB")
        
        print(f"\n🎯 Perfect for 30-second music fine-tuning!")
        print(f"📁 Working directory: {os.getcwd()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing SAO_diffusion setup...")
    success = test_sao_diffusion_setup()
    
    if success:
        print("\n🚀 READY FOR NEXT STEPS:")
        print("1. ✅ SAO_diffusion environment working")
        print("2. 📁 Next: Prepare your 30-second music dataset")
        print("3. ⚙️  Then: Configure fine-tuning parameters")
        print("4. 🏋️  Finally: Start fine-tuning process")
    else:
        print("\n🔧 Fix setup issues before proceeding")


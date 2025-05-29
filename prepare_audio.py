import os
import torchaudio
import torch
import json
from pathlib import Path
import argparse

def prepare_30s_audio_dataset(input_dir, output_dir, target_sample_rate=44100, target_duration=30.0):
    """
    Prepare audio files for 30-second music fine-tuning
    """
    print("🎵 PREPARING 30-SECOND MUSIC DATASET")
    print("="*50)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported audio formats
    audio_extensions = {'.wav', '.flac', '.mp3', '.m4a', '.ogg'}
    
    # Find all audio files
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(input_path.glob(f"*{ext}"))
        audio_files.extend(input_path.glob(f"**/*{ext}"))  # Recursive search
    
    print(f"📁 Found {len(audio_files)} audio files")
    
    if not audio_files:
        print("❌ No audio files found! Please check your input directory.")
        return []
    
    processed_files = []
    target_samples = int(target_sample_rate * target_duration)
    
    for i, audio_file in enumerate(audio_files):
        try:
            print(f"\n🔄 Processing {i+1}/{len(audio_files)}: {audio_file.name}")
            
            # Load audio
            waveform, sample_rate = torchaudio.load(str(audio_file))
            
            print(f"  📊 Original: {sample_rate}Hz, {waveform.shape[1]/sample_rate:.2f}s, {waveform.shape[0]} channels")
            
            # Convert to stereo if mono
            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)  # Duplicate mono to stereo
                print("  🔄 Converted mono to stereo")
            elif waveform.shape[0] > 2:
                waveform = waveform[:2, :]  # Take first 2 channels
                print("  🔄 Reduced to stereo")
            
            # Resample if needed
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
                waveform = resampler(waveform)
                print(f"  🔄 Resampled to {target_sample_rate}Hz")
            
            # Handle duration
            current_samples = waveform.shape[1]
            
            if current_samples > target_samples:
                # Trim to exactly 30 seconds (take from middle)
                start_sample = (current_samples - target_samples) // 2
                waveform = waveform[:, start_sample:start_sample + target_samples]
                print(f"  ✂️  Trimmed to 30.0s")
            elif current_samples < target_samples:
                # Pad with silence if too short
                padding = target_samples - current_samples
                waveform = torch.nn.functional.pad(waveform, (0, padding))
                print(f"  🔇 Padded to 30.0s (added {padding/target_sample_rate:.2f}s silence)")
            
            # Generate output filename
            output_filename = f"music_{i+1:03d}_30s.wav"
            output_filepath = output_path / output_filename
            
            # Save processed audio
            torchaudio.save(str(output_filepath), waveform, target_sample_rate)
            
            # Verify saved file
            verify_waveform, verify_sr = torchaudio.load(str(output_filepath))
            verify_duration = verify_waveform.shape[1] / verify_sr
            
            print(f"  ✅ Saved: {output_filename}")
            print(f"     Duration: {verify_duration:.2f}s, Channels: {verify_waveform.shape[0]}, SR: {verify_sr}Hz")
            
            processed_files.append({
                'filename': output_filename,
                'filepath': str(output_filepath),
                'original_file': str(audio_file),
                'duration': verify_duration,
                'sample_rate': verify_sr,
                'channels': verify_waveform.shape[0]
            })
            
        except Exception as e:
            print(f"  ❌ Error processing {audio_file.name}: {e}")
            continue
    
    print(f"\n🎉 Successfully processed {len(processed_files)} files!")
    
    # Save processing log
    log_file = output_path / "processing_log.json"
    with open(log_file, 'w') as f:
        json.dump(processed_files, f, indent=2)
    
    print(f"📝 Processing log saved to: {log_file}")
    
    return processed_files

def create_sample_descriptions(processed_files, output_dir):
    """
    Create sample text descriptions for the audio files
    """
    print("\n📝 Creating sample text descriptions...")
    
    # Sample descriptions - you'll want to customize these
    sample_descriptions = [
        "upbeat electronic music, 120 BPM, synthesizers, energetic",
        "calm ambient music, peaceful, atmospheric, slow tempo",
        "rock music, electric guitar, drums, bass, medium tempo",
        "jazz music, saxophone, piano, smooth, relaxed",
        "classical music, orchestra, strings, elegant, formal",
        "pop music, catchy melody, upbeat, modern",
        "hip hop beat, strong bass, rhythmic, urban",
        "folk music, acoustic guitar, organic, warm",
        "techno music, 128 BPM, electronic beats, dance",
        "blues music, guitar, soulful, emotional"
    ]
    
    descriptions = {}
    for i, file_info in enumerate(processed_files):
        # Cycle through sample descriptions
        desc_index = i % len(sample_descriptions)
        descriptions[file_info['filename']] = sample_descriptions[desc_index]
    
    # Save descriptions
    desc_file = Path(output_dir) / "descriptions.json"
    with open(desc_file, 'w') as f:
        json.dump(descriptions, f, indent=2)
    
    print(f"📄 Sample descriptions saved to: {desc_file}")
    print("⚠️  IMPORTANT: Edit descriptions.json to match your actual audio content!")
    
    return descriptions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare 30-second music dataset")
    parser.add_argument("--input", required=True, help="Input directory with raw audio files")
    parser.add_argument("--output", default="dataset/audio", help="Output directory for processed files")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Target sample rate")
    parser.add_argument("--duration", type=float, default=30.0, help="Target duration in seconds")
    
    args = parser.parse_args()
    
    # Process audio files
    processed_files = prepare_30s_audio_dataset(
        args.input, 
        args.output, 
        args.sample_rate, 
        args.duration
    )
    
    if processed_files:
        # Create sample descriptions
        create_sample_descriptions(processed_files, "dataset/metadata")
        
        print("\n🚀 DATASET PREPARATION COMPLETE!")
        print(f"✅ Processed {len(processed_files)} audio files")
        print(f"📁 Audio files: {args.output}")
        print(f"📝 Descriptions: dataset/metadata/descriptions.json")
        print("\nNEXT STEPS:")
        print("1. Review and edit descriptions.json")
        print("2. Run dataset validation")
        print("3. Configure fine-tuning parameters")
    else:
        print("❌ No files were processed successfully")

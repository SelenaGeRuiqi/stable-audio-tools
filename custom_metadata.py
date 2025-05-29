import os
import librosa
import numpy as np

def get_custom_metadata(info, audio):
    """
    Custom metadata function for deep house dataset
    
    Args:
        info: Dictionary containing file information (path, etc.)
        audio: Audio tensor that will be passed to the model
    
    Returns:
        Dictionary with custom metadata including the required 'prompt'
    """
    
    # Get the filename without extension
    filename = os.path.splitext(os.path.basename(info["path"]))[0]
    
    # Base deep house description
    base_prompt = "deep house music, electronic dance music, synthesizer, bass, rhythmic"
    
    # You can analyze the audio to add more specific descriptors
    # Convert audio tensor to numpy if needed
    if hasattr(audio, 'numpy'):
        audio_np = audio.numpy()
    else:
        audio_np = audio
    
    # Ensure audio is 1D for analysis
    if len(audio_np.shape) > 1:
        audio_np = audio_np.mean(axis=0)  # Convert to mono by averaging channels
    
    # Extract some basic audio features for more descriptive prompts
    try:
        # Calculate tempo (approximate)
        tempo = librosa.beat.tempo(y=audio_np, sr=44100, hop_length=512)[0]
        
        # Calculate spectral centroid (brightness)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_np, sr=44100))
        
        # Determine BPM range
        if tempo < 120:
            tempo_desc = "slow tempo"
        elif tempo < 128:
            tempo_desc = "moderate tempo" 
        else:
            tempo_desc = "upbeat tempo"
            
        # Determine brightness
        if spectral_centroid < 2000:
            brightness_desc = "warm"
        elif spectral_centroid < 4000:  
            brightness_desc = "balanced"
        else:
            brightness_desc = "bright"
            
        # Create enhanced prompt
        enhanced_prompt = f"{base_prompt}, {tempo_desc}, {brightness_desc} sound, {int(tempo)} BPM"
        
    except Exception as e:
        # Fallback if audio analysis fails
        print(f"Audio analysis failed for {info['path']}: {e}")
        enhanced_prompt = f"{base_prompt}, 128 BPM, electronic beats"
    
    # Return metadata dictionary - 'prompt' is required for training
    return {
        "prompt": enhanced_prompt,
        "genre": "deep house",
        "category": "electronic",
        "duration": len(audio_np) / 44100  # Duration in seconds
    }
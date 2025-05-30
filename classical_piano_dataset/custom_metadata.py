import json
from pathlib import Path

def get_custom_metadata(info, sample_rate=None):
    """Return custom metadata for piano training - FIXED KEYS"""
    
    prompts_file = Path("classical_piano_dataset") / "classical_piano_prompts.json"
    if prompts_file.exists():
        with open(prompts_file, 'r') as f:
            prompts = json.load(f)
        
        filename = Path(info['path']).name
        prompt_text = prompts.get(filename, "classical piano music, 96 BPM, expressive performance")
        
        return {
            "prompt": prompt_text,
            "seconds_start": 0,
            "seconds_total": 10
        }
    
    return {
        "prompt": "classical piano music, 96 BPM, expressive performance",
        "seconds_start": 0,
        "seconds_total": 10
    }
import requests
import json
import os
from pathlib import Path
import librosa
import numpy as np
import torchaudio
import torch
import time
import random
import re
import tempfile

class SimplePianoCollector:
    def __init__(self, output_dir="classical_piano_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print(f"✅ Created output directory: {self.output_dir}")
        
        # Create subdirectories
        (self.output_dir / "processed_10s").mkdir(exist_ok=True)
        print(f"✅ Created processed directory: {self.output_dir / 'processed_10s'}")
        
        self.processed_count = 0
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Improved quality criteria for 1000 clips
        self.quality_criteria = {
            "min_duration": 25,      # Slightly lower minimum
            "max_duration": 2400,    # 40 minutes max
            "min_file_size_mb": 0.3, # Lower minimum for more variety
            "max_file_size_mb": 150,
            "max_segments_per_file": 8  # More segments per file
        }

    def collect_piano_samples(self, target_clips=1000):
        """Enhanced collection for 1000+ piano clips"""
        print(f"🎹 Starting piano collection - Target: {target_clips} clips")
        print("🔍 Enhanced approach for large-scale collection")
        print("="*60)
        
        # Expanded queries for more variety
        queries = [
            # Format-specific searches
            'title:"piano" AND mediatype:audio AND format:"MP3"',
            'title:"piano" AND mediatype:audio AND format:"FLAC"',
            'title:"piano" AND mediatype:audio AND format:"Ogg"',
            'title:"piano" AND mediatype:audio AND format:"VBR MP3"',
            
            # Subject-based searches
            'subject:"piano" AND mediatype:audio',
            'subject:"classical piano" AND mediatype:audio',
            'subject:"piano music" AND mediatype:audio',
            'subject:"solo piano" AND mediatype:audio',
            'subject:"piano solo" AND mediatype:audio',
            'subject:"piano works" AND mediatype:audio',
            
            # Description-based searches
            'description:"piano music" AND mediatype:audio',
            'description:"piano performance" AND mediatype:audio',
            'description:"classical piano" AND mediatype:audio',
            'description:"piano recital" AND mediatype:audio',
            'description:"solo piano" AND mediatype:audio',
            
            # Collection-specific searches
            'collection:opensource_audio AND title:"piano"',
            'collection:etree AND title:"piano"',
            'collection:audio_music AND title:"piano"',
            'collection:netlabels AND title:"piano"',
            
            # Genre-specific searches
            'title:"classical piano" AND mediatype:audio',
            'title:"romantic piano" AND mediatype:audio',
            'title:"baroque piano" AND mediatype:audio',
            'title:"piano sonata" AND mediatype:audio',
            'title:"piano concerto" AND mediatype:audio',
            'title:"piano pieces" AND mediatype:audio',
            
            # Composer searches
            'title:"chopin" AND title:"piano" AND mediatype:audio',
            'title:"beethoven" AND title:"piano" AND mediatype:audio',
            'title:"mozart" AND title:"piano" AND mediatype:audio',
            'title:"bach" AND title:"piano" AND mediatype:audio',
            'title:"debussy" AND title:"piano" AND mediatype:audio',
            'title:"liszt" AND title:"piano" AND mediatype:audio',
            
            # Broad searches for variety
            'mediatype:audio AND title:"piano"',
            'mediatype:audio AND description:"piano"',
            'format:"MP3" AND subject:"music" AND title:"piano"',
            'format:"FLAC" AND subject:"music" AND title:"piano"'
        ]
        
        successful_downloads = 0
        
        for i, query in enumerate(queries):
            if self.processed_count >= target_clips:
                break
                
            print(f"\n🔍 Query {i+1}/{len(queries)}: {query[:50]}...")
            
            try:
                results = self._search_internet_archive(query, max_results=20)  # More results per query
                print(f"   Found {len(results)} potential files")
                
                for j, result in enumerate(results):
                    if self.processed_count >= target_clips:
                        break
                    
                    print(f"   📥 Trying {j+1}/{len(results)}: {result.get('title', 'Unknown')[:40]}...")
                    
                    try:
                        success = self._download_and_process_simple(result)
                        if success:
                            successful_downloads += 1
                            print(f"   ✅ Success! Total clips: {self.processed_count}")
                        else:
                            print(f"   ❌ Failed to process")
                    except Exception as e:
                        print(f"   ❌ Error: {str(e)[:50]}")
                    
                    time.sleep(0.3)  # Faster between downloads
                
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
            
            time.sleep(1)
        
        print(f"\n📊 Collection Summary:")
        print(f"   Successful downloads: {successful_downloads}")
        print(f"   Total clips created: {self.processed_count}")
        
        return self.processed_count

    def _search_internet_archive(self, query, max_results=20):
        """Enhanced search with pagination for more results"""
        base_url = "https://archive.org/advancedsearch.php"
        
        all_results = []
        
        # Try multiple pages for each query
        for page in range(1, 4):  # Check first 3 pages
            params = {
                'q': query,
                'fl': 'identifier,title,creator,description',
                'rows': max_results,
                'page': page,
                'output': 'json'
            }
            
            try:
                response = self.session.get(base_url, params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                results = data.get('response', {}).get('docs', [])
                
                if not results:  # No more results
                    break
                    
                all_results.extend(results)
                
                # Random delay between pages
                time.sleep(random.uniform(0.5, 1.0))
                
            except Exception as e:
                print(f"      Page {page} error: {e}")
                break
        
        # Shuffle results for variety
        random.shuffle(all_results)
        return all_results[:max_results * 2]  # Return up to 40 results

    def _download_and_process_simple(self, doc):
        """Simplified download and process"""
        identifier = doc.get('identifier')
        title = doc.get('title', 'Unknown')
        
        if not identifier:
            return False
        
        try:
            # Get file metadata
            metadata_url = f"https://archive.org/metadata/{identifier}"
            response = self.session.get(metadata_url, timeout=15)
            metadata = response.json()
            
            # Find audio files
            audio_files = []
            for file_info in metadata.get('files', []):
                format_name = file_info.get('format', '')
                filename = file_info.get('name', '')
                size_str = file_info.get('size', '0')
                
                if any(fmt in format_name for fmt in ['MP3', 'FLAC', 'Ogg']) or filename.lower().endswith(('.mp3', '.flac', '.ogg')):
                    try:
                        size_mb = int(size_str) / (1024 * 1024)
                        if self.quality_criteria["min_file_size_mb"] <= size_mb <= self.quality_criteria["max_file_size_mb"]:
                            audio_files.append(file_info)
                    except:
                        continue
            
            if not audio_files:
                return False
            
            # Take the first suitable file
            selected_file = audio_files[0]
            filename = selected_file.get('name')
            download_url = f"https://archive.org/download/{identifier}/{filename}"
            
            # Download to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as temp_file:
                temp_path = temp_file.name
                
                try:
                    print(f"      Downloading: {filename}")
                    response = self.session.get(download_url, stream=True, timeout=60)
                    response.raise_for_status()
                    
                    # Download in chunks
                    for chunk in response.iter_content(chunk_size=8192):
                        temp_file.write(chunk)
                    
                    temp_file.flush()
                    
                    # Process the file
                    segments_created = self._process_audio_simple(temp_path, title)
                    
                    # Clean up
                    os.unlink(temp_path)
                    
                    return segments_created > 0
                    
                except Exception as e:
                    print(f"      Download error: {e}")
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                    return False
        
        except Exception as e:
            print(f"      Metadata error: {e}")
            return False

    def _process_audio_simple(self, file_path, title):
        """Simple audio processing to 10s segments"""
        try:
            print(f"      Processing audio...")
            
            # Load audio with librosa
            audio, sr = librosa.load(file_path, sr=44100, duration=300)  # Max 5 minutes
            duration = len(audio) / sr
            
            print(f"      Audio duration: {duration:.1f}s")
            
            if duration < self.quality_criteria["min_duration"]:
                print(f"      Too short (need >{self.quality_criteria['min_duration']}s)")
                return 0
            
            # Calculate how many 10s segments we can make
            segment_length = 10 * sr
            max_segments = min(self.quality_criteria["max_segments_per_file"], int(duration // 12))
            
            print(f"      Creating {max_segments} segments...")
            
            segments_created = 0
            processed_dir = self.output_dir / "processed_10s"
            
            for seg_idx in range(max_segments):
                # Calculate start position
                if max_segments == 1:
                    start_sample = max(0, int((len(audio) - segment_length) / 2))
                else:
                    segment_spacing = max(segment_length // 2, (len(audio) - segment_length) // max(1, max_segments - 1))
                    start_sample = min(seg_idx * segment_spacing, len(audio) - segment_length)
                
                start_sample = max(0, start_sample)
                segment = audio[start_sample:start_sample + segment_length]
                
                # Pad if needed
                if len(segment) < segment_length:
                    padding = segment_length - len(segment)
                    segment = np.pad(segment, (0, padding), mode='constant')
                
                segment = segment[:segment_length]
                
                # Convert to stereo
                if len(segment.shape) == 1:
                    segment_stereo = np.stack([segment, segment])
                else:
                    segment_stereo = segment
                
                # Normalize
                max_val = np.max(np.abs(segment_stereo))
                if max_val > 0:
                    segment_stereo = segment_stereo / max_val * 0.9
                
                # Save
                self.processed_count += 1
                output_filename = f"piano_{self.processed_count:04d}.wav"
                output_path = processed_dir / output_filename
                
                # Convert to tensor and save
                segment_tensor = torch.from_numpy(segment_stereo).float()
                torchaudio.save(str(output_path), segment_tensor, sr)
                
                segments_created += 1
                print(f"      ✅ Created: {output_filename}")
                
                # Stop at target
                if self.processed_count >= 1000:  # Hard stop at 1000
                    break
            
            return segments_created
            
        except Exception as e:
            print(f"      Processing error: {e}")
            return 0

    def generate_simple_prompts(self):
        """Generate simple but effective piano prompts"""
        print(f"\n📝 Generating prompts for piano clips...")
        
        processed_files = list((self.output_dir / "processed_10s").glob("*.wav"))
        
        if len(processed_files) == 0:
            print("❌ No processed files found!")
            return {}
        
        # Simple but effective piano prompts
        prompt_templates = [
            "classical piano music, {tempo} BPM, {style}, {mood}",
            "piano solo, {tempo} BPM, {character}, {acoustic}",
            "solo piano, {tempo} BPM, {expression}, {quality}",
            "piano music, {tempo} BPM, {genre}, {atmosphere}"
        ]
        
        vocab = {
            "tempo": [72, 80, 88, 96, 104, 112, 120, 132, 144],
            "style": ["classical style", "romantic style", "baroque style", "contemporary style"],
            "mood": ["contemplative", "peaceful", "dramatic", "gentle", "expressive"],
            "character": ["lyrical melody", "flowing phrases", "expressive performance", "delicate touch"],
            "acoustic": ["concert grand piano", "warm acoustics", "clear articulation", "rich resonance"],
            "expression": ["cantabile", "legato phrasing", "dynamic expression", "musical phrasing"],
            "quality": ["high quality recording", "professional performance", "studio recording"],
            "genre": ["classical music", "art music", "concert music", "instrumental music"],
            "atmosphere": ["intimate setting", "concert hall", "peaceful atmosphere", "refined ambiance"]
        }
        
        prompts = {}
        
        for file_path in processed_files:
            template = random.choice(prompt_templates)
            prompt_vars = {}
            
            # Fill variables
            template_vars = re.findall(r'{(\w+)}', template)
            for var in template_vars:
                if var in vocab:
                    prompt_vars[var] = random.choice(vocab[var])
            
            prompt = template.format(**prompt_vars)
            prompts[file_path.name] = prompt
        
        # Save prompts
        prompts_file = self.output_dir / "classical_piano_prompts.json"
        with open(prompts_file, 'w') as f:
            json.dump(prompts, f, indent=2)
        
        print(f"✅ Generated {len(prompts)} prompts")
        print(f"📁 Saved to: {prompts_file}")
        
        return prompts

    def create_configs(self):
        """Create training configuration files"""
        print(f"\n⚙️ Creating training configs...")
        
        processed_files = list((self.output_dir / "processed_10s").glob("*.wav"))
        
        # Dataset config
        dataset_config = {
            "dataset_type": "audio_dir",
            "datasets": [{
                "id": "piano_10s",
                "path": str(self.output_dir / "processed_10s") + "/",
                "custom_metadata_module": str(self.output_dir / "custom_metadata.py")
            }],
            "random_crop": True,
            "sample_rate": 44100,
            "sample_size": 441000,
            "channels": 2
        }
        
        config_file = self.output_dir / "dataset_config.json"
        with open(config_file, 'w') as f:
            json.dump(dataset_config, f, indent=2)
        
        # Custom metadata
        metadata_code = f'''import json
from pathlib import Path

def get_custom_metadata(info):
    """Return custom metadata for piano training"""
    
    prompts_file = Path("{self.output_dir}") / "classical_piano_prompts.json"
    if prompts_file.exists():
        with open(prompts_file, 'r') as f:
            prompts = json.load(f)
        
        filename = Path(info['path']).name
        prompt = prompts.get(filename, "classical piano music, 96 BPM, expressive performance")
        
        return {{
            "text": prompt,
            "seconds_start": 0,
            "seconds_total": 10
        }}
    
    return {{
        "text": "classical piano music, 96 BPM, expressive performance",
        "seconds_start": 0,
        "seconds_total": 10
    }}
'''
        
        metadata_file = self.output_dir / "custom_metadata.py"
        with open(metadata_file, 'w') as f:
            f.write(metadata_code)
        
        # Summary
        summary = {
            'dataset_name': 'Piano Fine-tuning Dataset',
            'total_clips': len(processed_files),
            'duration_per_clip': '10 seconds',
            'format': '44.1kHz WAV, stereo',
            'files': {
                'dataset_config': str(config_file),
                'custom_metadata': str(metadata_file),
                'prompts': str(self.output_dir / "classical_piano_prompts.json"),
                'audio_dir': str(self.output_dir / "processed_10s")
            }
        }
        
        summary_file = self.output_dir / "dataset_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Dataset config: {config_file}")
        print(f"✅ Custom metadata: {metadata_file}")
        print(f"✅ Summary: {summary_file}")
        
        return summary

def main():
    print("🎹 SIMPLE PIANO COLLECTOR")
    print("="*50)
    print("🎯 Goal: Collect piano audio and create training dataset")
    print("🔧 Simplified approach for better reliability")
    print("="*50)
    
    collector = SimplePianoCollector()
    
    # Check existing
    existing_files = list((collector.output_dir / "processed_10s").glob("*.wav"))
    if existing_files:
        collector.processed_count = len(existing_files)
        print(f"📁 Found {len(existing_files)} existing files")
    
    # Collect samples
    print(f"\n🔍 PHASE 1: Collecting Audio")
    clips_collected = collector.collect_piano_samples(target_clips=1000)  # Target 1000 clips
    
    if clips_collected == 0:
        print("❌ No clips collected. Check your internet connection.")
        return
    
    # Generate prompts
    print(f"\n📝 PHASE 2: Generating Prompts")
    prompts = collector.generate_simple_prompts()
    
    # Create configs
    print(f"\n⚙️ PHASE 3: Creating Configs")
    summary = collector.create_configs()
    
    # Final report
    print(f"\n🎉 COLLECTION COMPLETE!")
    print("="*40)
    print(f"📊 Total clips: {summary['total_clips']}")
    print(f"📁 Audio directory: {summary['files']['audio_dir']}")
    print(f"⚙️  Dataset config: {summary['files']['dataset_config']}")
    
    if summary['total_clips'] > 0:
        print(f"\n🚀 READY FOR TRAINING!")
        print(f"💡 Training command:")
        print(f"""python train.py \\
  --dataset-config {summary['files']['dataset_config']} \\
  --model-config ./stable-audio-open-1.0/model_config.json \\
  --pretrained-ckpt-path ./stable-audio-open-1.0/model.safetensors \\
  --name piano_{summary['total_clips']}_clips \\
  --batch-size 2 \\
  --accum-batches 2 \\
  --precision 16 \\
  --checkpoint-every 200 \\
  --save-dir ./checkpoints""")

if __name__ == "__main__":
    main()
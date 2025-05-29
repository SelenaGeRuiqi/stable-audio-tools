import requests
import json
import os
from pathlib import Path
import librosa
import numpy as np
import torchaudio
import torch
from bs4 import BeautifulSoup
import time
import random
import re
from urllib.parse import urljoin, urlparse
import subprocess

class FixedDeepHouseCollector:
    def __init__(self, output_dir="deep_house_dataset"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Expanded and more flexible search strategy
        self.search_strategies = [
            {
                "name": "Internet Archive - Electronic",
                "method": "internet_archive",
                "terms": ["electronic", "house", "techno", "dance", "club"],
                "target_files": 15
            },
            {
                "name": "Internet Archive - Music Collections", 
                "method": "internet_archive_broad",
                "collections": ["etree", "audio_music", "opensource_audio"],
                "target_files": 10
            },
            {
                "name": "Freesound - House Samples",
                "method": "freesound_search", 
                "terms": ["house", "deep", "bass", "groove", "electronic"],
                "target_files": 5
            }
        ]
        
        # Relaxed quality criteria to get more files
        self.quality_criteria = {
            "min_sample_rate": 22050,  # Lowered from 44100
            "target_duration": 20,
            "min_duration": 10,        # Lowered from 15
            "max_duration": 600,       # Increased
            "min_file_size_mb": 0.1,   # Lowered
            "max_file_size_mb": 100,   # Increased
            "preferred_formats": [".wav", ".flac", ".mp3", ".ogg"],
            "min_bpm": 100,            # Broader range
            "max_bpm": 150
        }
        
        self.collected_files = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

    def run_comprehensive_search(self, target_files=30):
        """Run comprehensive search across multiple strategies"""
        print(f"🎯 TARGET: {target_files} Deep House tracks")
        print("="*50)
        
        for strategy in self.search_strategies:
            if len(self.collected_files) >= target_files:
                break
                
            print(f"\n🔍 {strategy['name']}")
            print("-" * 30)
            
            remaining_files = min(strategy['target_files'], target_files - len(self.collected_files))
            
            if strategy['method'] == 'internet_archive':
                self.search_internet_archive_electronic(strategy['terms'], remaining_files)
            elif strategy['method'] == 'internet_archive_broad':
                self.search_internet_archive_broad(strategy['collections'], remaining_files)
            elif strategy['method'] == 'freesound_search':
                self.search_freesound_samples(strategy['terms'], remaining_files)
            
            print(f"📊 Progress: {len(self.collected_files)}/{target_files} files collected")
        
        return self.collected_files

    def search_internet_archive_electronic(self, search_terms, max_files):
        """Enhanced Internet Archive search for electronic music"""
        base_url = "https://archive.org/advancedsearch.php"
        
        # Broader search queries
        search_queries = [
            'collection:opensource_audio AND subject:"electronic"',
            'collection:opensource_audio AND title:"house"',
            'collection:opensource_audio AND title:"deep"',
            'collection:etree AND subject:"electronic"',
            'collection:audio_music AND description:"dance"',
            'format:"VBR MP3" AND subject:"music"',  # Very broad
        ]
        
        for query in search_queries:
            if len(self.collected_files) >= max_files:
                break
                
            print(f"🔍 Query: {query[:50]}...")
            
            params = {
                'q': query,
                'fl': 'identifier,title,creator,description,downloads,format',
                'rows': 20,  # More results per query
                'page': 1,
                'output': 'json'
            }
            
            try:
                response = self.session.get(base_url, params=params, timeout=30)
                data = response.json()
                
                for doc in data.get('response', {}).get('docs', []):
                    if len(self.collected_files) >= max_files:
                        break
                    
                    identifier = doc.get('identifier')
                    title = doc.get('title', 'Unknown')
                    creator = doc.get('creator', 'Unknown')
                    description = doc.get('description', '')
                    
                    if identifier and self._might_be_electronic_music(title, description):
                        success = self._download_from_internet_archive_fixed(identifier, title, creator)
                        if success:
                            print(f"  ✅ Downloaded: {title[:50]}...")
                        
                        time.sleep(1)  # Be nice to the server
                
            except Exception as e:
                print(f"  ❌ Query error: {e}")
            
            time.sleep(2)

    def search_internet_archive_broad(self, collections, max_files):
        """Broad search across multiple IA collections"""
        base_url = "https://archive.org/advancedsearch.php"
        
        for collection in collections:
            if len(self.collected_files) >= max_files:
                break
                
            print(f"🔍 Collection: {collection}")
            
            # Very broad queries for each collection
            queries = [
                f'collection:{collection} AND format:"VBR MP3"',
                f'collection:{collection} AND format:"FLAC"',
                f'collection:{collection} AND subject:"music"'
            ]
            
            for query in queries:
                if len(self.collected_files) >= max_files:
                    break
                
                params = {
                    'q': query,
                    'fl': 'identifier,title,creator,description',
                    'rows': 15,
                    'page': 1,
                    'output': 'json'
                }
                
                try:
                    response = self.session.get(base_url, params=params, timeout=30)
                    data = response.json()
                    
                    for doc in data.get('response', {}).get('docs', []):
                        if len(self.collected_files) >= max_files:
                            break
                        
                        identifier = doc.get('identifier')
                        title = doc.get('title', 'Unknown')
                        creator = doc.get('creator', 'Unknown')
                        
                        if identifier:
                            success = self._download_from_internet_archive_fixed(identifier, title, creator)
                            if success:
                                print(f"  ✅ Found usable track: {title[:40]}...")
                
                except Exception as e:
                    print(f"  ❌ Collection {collection} error: {e}")
                
                time.sleep(1)

    def search_freesound_samples(self, search_terms, max_files):
        """Search Freesound for electronic music samples"""
        print("🔍 Searching Freesound (sample only - no API key)")
        
        # Note: This would require Freesound API key
        # For now, we'll use Internet Archive more extensively
        
        # Alternative: Search for longer form music that might work
        self.search_internet_archive_music_specific(max_files)

    def search_internet_archive_music_specific(self, max_files):
        """Specific music-focused IA search"""
        base_url = "https://archive.org/advancedsearch.php"
        
        # Music-specific queries
        music_queries = [
            'mediatype:audio AND format:"VBR MP3"',
            'mediatype:audio AND format:"FLAC"', 
            'subject:"Music" AND format:"128Kbps MP3"',
            'collection:opensource_media AND mediatype:audio',
            'title:"mix" AND mediatype:audio',
            'title:"set" AND mediatype:audio'
        ]
        
        for query in music_queries:
            if len(self.collected_files) >= max_files:
                break
                
            print(f"🎵 Music query: {query[:40]}...")
            
            params = {
                'q': query,
                'fl': 'identifier,title,creator,description',
                'rows': 10,
                'page': random.randint(1, 5),  # Random page for variety
                'output': 'json'
            }
            
            try:
                response = self.session.get(base_url, params=params, timeout=30)
                data = response.json()
                
                for doc in data.get('response', {}).get('docs', []):
                    if len(self.collected_files) >= max_files:
                        break
                    
                    identifier = doc.get('identifier')
                    title = doc.get('title', 'Unknown')
                    creator = doc.get('creator', 'Unknown')
                    
                    if identifier:
                        success = self._download_from_internet_archive_fixed(identifier, title, creator)
                        if success:
                            print(f"  ✅ Music track: {title[:40]}...")
            
            except Exception as e:
                print(f"  ❌ Music query error: {e}")
            
            time.sleep(2)

    def _might_be_electronic_music(self, title, description):
        """Very permissive check for potentially electronic music"""
        text = (title + " " + description).lower()
        
        # Electronic indicators (very broad)
        positive_indicators = [
            'electronic', 'house', 'techno', 'dance', 'club', 'beat', 'groove',
            'mix', 'dj', 'remix', 'synth', 'bass', 'track', 'music', 'sound',
            'ambient', 'chill', 'downtempo', 'minimal'
        ]
        
        # Things to definitely avoid
        avoid_indicators = [
            'speech', 'talk', 'podcast', 'radio show', 'news', 'interview',
            'book', 'reading', 'audiobook', 'lecture', 'sermon'
        ]
        
        has_positive = any(indicator in text for indicator in positive_indicators)
        has_negative = any(indicator in text for indicator in avoid_indicators)
        
        return has_positive and not has_negative

    def _download_from_internet_archive_fixed(self, identifier, title, creator):
        """Fixed version with better error handling"""
        try:
            # Get metadata
            metadata_url = f"https://archive.org/metadata/{identifier}"
            response = self.session.get(metadata_url, timeout=30)
            metadata = response.json()
            
            # Find audio files with more flexible criteria
            audio_files = []
            for file_info in metadata.get('files', []):
                format_name = file_info.get('format', '')
                filename = file_info.get('name', '')
                
                # More flexible format matching
                if any(fmt in format_name for fmt in ['MP3', 'FLAC', 'Ogg', 'VBR']):
                    size = file_info.get('size', '0')
                    try:
                        size_mb = int(size) / (1024 * 1024)
                        if (self.quality_criteria["min_file_size_mb"] <= size_mb <= 
                            self.quality_criteria["max_file_size_mb"]):
                            audio_files.append(file_info)
                    except:
                        # If size parsing fails, still try the file
                        audio_files.append(file_info)
            
            if not audio_files:
                return False
            
            # Take the first suitable file (or random selection)
            selected_file = random.choice(audio_files)
            filename = selected_file.get('name')
            
            if filename:
                download_url = f"https://archive.org/download/{identifier}/{filename}"
                
                # Fixed filename creation
                safe_filename = self._create_safe_filename_fixed(f"{creator}_{title}_{filename}")
                local_path = self.output_dir / "raw" / safe_filename
                local_path.parent.mkdir(exist_ok=True, parents=True)
                
                # Download with better error handling
                try:
                    response = self.session.get(download_url, stream=True, timeout=120)
                    response.raise_for_status()
                    
                    with open(local_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Quick verification
                    if local_path.exists() and local_path.stat().st_size > 1024:  # At least 1KB
                        # Try to verify it's audio (basic check)
                        if self._basic_audio_check(local_path):
                            file_info = {
                                'path': str(local_path),
                                'original_title': title,
                                'creator': creator,
                                'source': 'Internet Archive',
                                'identifier': identifier,
                                'license': 'Creative Commons'
                            }
                            
                            self.collected_files.append(file_info)
                            return True
                        else:
                            # Remove non-audio file
                            os.remove(local_path)
                            return False
                    else:
                        return False
                        
                except Exception as e:
                    print(f"    ❌ Download failed: {e}")
                    return False
                
        except Exception as e:
            print(f"    ❌ Metadata error: {e}")
            return False

    def _create_safe_filename_fixed(self, filename):
        """Fixed safe filename creation"""
        # Remove problematic characters more safely
        safe_name = re.sub(r'[<>:"/\\|?*]', '', filename)  # Remove filesystem-unsafe chars
        safe_name = re.sub(r'[^\w\s.-]', '_', safe_name)   # Replace other special chars
        safe_name = re.sub(r'\s+', '_', safe_name)         # Replace spaces with underscores
        safe_name = safe_name[:100]  # Limit length
        
        # Ensure it has an extension
        if '.' not in safe_name[-5:]:
            safe_name += '.mp3'
        
        return safe_name

    def _basic_audio_check(self, file_path):
        """Basic check if file is likely audio"""
        try:
            # Try to load with librosa (basic test)
            y, sr = librosa.load(str(file_path), duration=1.0)  # Just load 1 second
            
            if len(y) > 1000 and sr > 8000:  # Very basic sanity check
                return True
            else:
                return False
                
        except Exception as e:
            # If librosa fails, try file extension
            ext = file_path.suffix.lower()
            return ext in ['.mp3', '.wav', '.flac', '.ogg', '.m4a']

    def process_to_20s_segments_enhanced(self):
        """Enhanced processing with better error handling"""
        print(f"\n🔄 Processing {len(self.collected_files)} files into 20-second segments...")
        
        processed_dir = self.output_dir / "processed_20s"
        processed_dir.mkdir(exist_ok=True, parents=True)
        
        processed_files = []
        
        for i, file_info in enumerate(self.collected_files):
            try:
                print(f"\n📂 Processing file {i+1}/{len(self.collected_files)}: {Path(file_info['path']).name[:50]}...")
                
                # Load audio with error handling
                try:
                    audio, sr = librosa.load(file_info['path'], sr=44100)
                except Exception as e:
                    print(f"   ❌ Could not load audio: {e}")
                    continue
                
                duration = len(audio) / sr
                print(f"   📊 Duration: {duration:.1f}s, Sample rate: {sr}Hz")
                
                if duration < 15:
                    print(f"   ⚠️  Too short, skipping")
                    continue
                
                # Extract segments more intelligently
                segment_length = 20 * sr
                
                if duration >= 20:
                    # Extract up to 3 segments from longer tracks
                    max_segments = min(3, int(duration // 15))  # Every 15 seconds
                    
                    for seg_idx in range(max_segments):
                        # Smart segment positioning
                        if max_segments == 1:
                            # Single segment from middle
                            start_sample = max(0, int((len(audio) - segment_length) / 2))
                        else:
                            # Distribute segments across track
                            start_sample = int(seg_idx * (len(audio) - segment_length) / (max_segments - 1))
                        
                        start_sample = max(0, min(start_sample, len(audio) - segment_length))
                        end_sample = start_sample + segment_length
                        
                        segment = audio[start_sample:end_sample]
                        
                        # Pad if necessary
                        if len(segment) < segment_length:
                            padding = segment_length - len(segment)
                            segment = np.pad(segment, (0, padding), mode='constant')
                        
                        # Convert to stereo tensor
                        if len(segment.shape) == 1:
                            segment_stereo = np.stack([segment, segment])
                        else:
                            segment_stereo = segment.T if segment.shape[1] == 2 else np.stack([segment, segment])
                        
                        # Save segment
                        output_filename = f"deep_house_{i+1:03d}_{seg_idx+1:02d}.wav"
                        output_path = processed_dir / output_filename
                        
                        segment_tensor = torch.from_numpy(segment_stereo).float()
                        torchaudio.save(str(output_path), segment_tensor, 44100)
                        
                        processed_files.append({
                            'path': str(output_path),
                            'filename': output_filename,
                            'original_info': file_info,
                            'segment_index': seg_idx,
                            'duration': 20.0
                        })
                        
                        print(f"   ✅ Created: {output_filename}")
                
            except Exception as e:
                print(f"   ❌ Processing error: {e}")
                continue
        
        print(f"\n🎉 Successfully created {len(processed_files)} 20-second segments!")
        return processed_files

    def generate_enhanced_prompts(self, processed_files):
        """Enhanced prompt generation with more variety"""
        print("\n📝 Generating enhanced Deep House prompts...")
        
        # More diverse prompt templates
        templates = [
            "deep house music, {bpm} BPM, {bass_type}, {rhythm_style}, {atmosphere}",
            "deep house track, {bpm} BPM, {production_style}, {harmonic_content}, {energy}",
            "house music, {bpm} BPM, {groove_type}, {sonic_character}, {club_vibe}",
            "deep house, {bpm} BPM, {instrument_focus}, {texture}, professional quality",
            "electronic house music, {bpm} BPM, {mood}, {arrangement}, {mix_style}"
        ]
        
        # Expanded vocabulary
        vocab = {
            "bass_type": ["warm rolling bassline", "deep sub bass", "organic bass groove", "smooth bass foundation", "punchy bass", "melodic bassline"],
            "rhythm_style": ["four-on-the-floor pattern", "steady groove", "hypnotic rhythm", "driving pulse", "syncopated beats", "minimal percussion"],
            "atmosphere": ["lush atmospheric pads", "spacious soundscape", "intimate atmosphere", "ethereal ambience", "warm textures", "dreamy pads"],
            "production_style": ["crisp production", "analog warmth", "modern clarity", "vintage-inspired", "polished sound", "organic production"],
            "harmonic_content": ["jazzy chord progressions", "sophisticated harmonies", "rich chord work", "melodic elements", "harmonic layers", "tonal sophistication"],
            "energy": ["moderate energy flow", "building intensity", "controlled dynamics", "steady drive", "euphoric lift", "contemplative mood"],
            "groove_type": ["infectious groove", "subtle swing", "tight rhythm", "flowing pulse", "locked groove", "rhythmic foundation"],
            "sonic_character": ["warm analog character", "digital precision", "organic textures", "smooth transitions", "layered soundscape", "refined tones"],
            "club_vibe": ["underground atmosphere", "late night energy", "dancefloor groove", "intimate club setting", "sophisticated nightlife", "after hours vibe"],
            "instrument_focus": ["synthesizer pads", "electric piano touches", "string elements", "percussion layers", "filtered vocals", "ambient textures"],
            "texture": ["layered composition", "dynamic arrangement", "evolving textures", "spatial depth", "sonic richness", "detailed production"],
            "mood": ["uplifting", "contemplative", "soulful", "euphoric", "introspective", "positive"],
            "arrangement": ["carefully structured", "flowing arrangement", "balanced mix", "thoughtful composition", "dynamic build", "seamless transitions"],
            "mix_style": ["wide stereo field", "balanced frequencies", "clear separation", "cohesive blend", "professional mixdown", "spatial imaging"]
        }
        
        prompts_data = []
        
        for i, file_info in enumerate(processed_files):
            # Generate BPM in Deep House range
            bpm = random.randint(120, 128)
            
            # Select template and fill
            template = random.choice(templates)
            prompt_vars = {"bpm": bpm}
            
            # Fill template variables
            import re
            template_vars = re.findall(r'{(\w+)}', template)
            for var in template_vars:
                if var in vocab:
                    prompt_vars[var] = random.choice(vocab[var])
                elif var == "bpm":
                    prompt_vars[var] = bpm
            
            prompt = template.format(**prompt_vars)
            
            prompt_data = {
                'filename': file_info['filename'],
                'path': file_info['path'],
                'prompt': prompt,
                'target_bpm': bpm,
                'original_source': file_info['original_info'].get('source', 'Internet Archive'),
                'duration': 20.0
            }
            
            prompts_data.append(prompt_data)
            print(f"✅ {file_info['filename']}: {prompt}")
        
        return prompts_data

    def save_complete_dataset(self, prompts_data):
        """Save the complete dataset"""
        print(f"\n💾 Saving complete dataset with {len(prompts_data)} tracks...")
        
        # Save prompts for training
        prompts_json = {item['filename']: item['prompt'] for item in prompts_data}
        
        prompts_file = self.output_dir / "deep_house_prompts.json"
        with open(prompts_file, 'w') as f:
            json.dump(prompts_json, f, indent=2)
        
        # Save metadata
        metadata_file = self.output_dir / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(prompts_data, f, indent=2)
        
        # Create summary
        summary = {
            'dataset_name': 'Deep House Fine-tuning Dataset',
            'total_files': len(prompts_data),
            'duration_per_file': '20 seconds',
            'total_duration_minutes': (len(prompts_data) * 20) / 60,
            'format': '44.1kHz WAV, stereo',
            'genre': 'Deep House',
            'bpm_range': '120-128 BPM',
            'files': {
                'prompts': str(prompts_file),
                'metadata': str(metadata_file),
                'audio_dir': str(self.output_dir / "processed_20s")
            }
        }
        
        summary_file = self.output_dir / "dataset_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"📁 Dataset ready at: {self.output_dir}")
        print(f"📄 Prompts: {prompts_file}")
        print(f"🎵 Audio files: {self.output_dir / 'processed_20s'}")
        
        return summary

def main():
    print("🎵 FIXED DEEP HOUSE DATA COLLECTOR")
    print("="*45)
    
    collector = FixedDeepHouseCollector()
    
    # Step 1: Comprehensive search
    collected_files = collector.run_comprehensive_search(target_files=30)
    
    if len(collected_files) == 0:
        print("❌ No files collected. Internet connection or archive issues.")
        return
    
    print(f"\n✅ Collected {len(collected_files)} raw audio files")
    
    # Step 2: Process to 20s segments
    processed_files = collector.process_to_20s_segments_enhanced()
    
    if len(processed_files) == 0:
        print("❌ No files processed successfully.")
        return
    
    # Step 3: Generate prompts
    prompts_data = collector.generate_enhanced_prompts(processed_files)
    
    # Step 4: Save dataset
    summary = collector.save_complete_dataset(prompts_data)
    
    print("\n🎉 ENHANCED DATASET COMPLETE!")
    print("="*40)
    print(f"📊 Total segments: {summary['total_files']}")
    print(f"⏱️  Total duration: {summary['total_duration_minutes']:.1f} minutes")
    print(f"🎯 Target achieved: {len(prompts_data) >= 25} ({'✅' if len(prompts_data) >= 25 else '❌'})")
    
    if len(prompts_data) < 25:
        print(f"\n💡 TIPS TO GET MORE FILES:")
        print("1. Run the script again (different random selections)")
        print("2. Manually add some tracks to 'raw' folder")
        print("3. Try different time of day (archive availability varies)")

if __name__ == "__main__":
    main()
import requests
import json
import os
from pathlib import Path
import time
import urllib.parse
from bs4 import BeautifulSoup
import yt_dlp

class MusicDownloader:
    def __init__(self, output_dir="raw_audio"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.downloaded_files = []
        
    def download_from_archive_org(self, search_terms=["electronic", "ambient", "instrumental"], max_files=20):
        """
        Download from Internet Archive - high quality, legally clear
        """
        print("🏛️  Downloading from Internet Archive...")
        
        base_url = "https://archive.org/advancedsearch.php"
        
        for term in search_terms[:3]:  # Limit search terms
            print(f"\n🔍 Searching for: {term}")
            
            params = {
                'q': f'collection:opensource_audio AND subject:"{term}" AND format:VBR MP3',
                'fl': 'identifier,title,creator,format',
                'rows': min(max_files // len(search_terms), 10),
                'page': 1,
                'output': 'json'
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=30)
                data = response.json()
                
                for doc in data.get('response', {}).get('docs', []):
                    identifier = doc.get('identifier')
                    title = doc.get('title', 'Unknown')
                    creator = doc.get('creator', 'Unknown')
                    
                    if identifier:
                        download_url = f"https://archive.org/download/{identifier}"
                        self._download_archive_item(identifier, title, creator, download_url)
                        
                        if len(self.downloaded_files) >= max_files:
                            break
                            
                time.sleep(2)  # Be respectful to the server
                
            except Exception as e:
                print(f"❌ Error searching Internet Archive: {e}")
        
        return self.downloaded_files
    
    def _download_archive_item(self, identifier, title, creator, base_url):
        """Download a specific item from Internet Archive"""
        try:
            # Get file list
            files_url = f"https://archive.org/metadata/{identifier}"
            response = requests.get(files_url, timeout=30)
            metadata = response.json()
            
            # Look for best quality audio file
            audio_files = []
            for file_info in metadata.get('files', []):
                if file_info.get('format') in ['VBR MP3', '320Kbps MP3', 'FLAC', 'Ogg Vorbis']:
                    audio_files.append(file_info)
            
            if not audio_files:
                print(f"⚠️  No suitable audio files found for {title}")
                return
            
            # Choose best quality file
            best_file = max(audio_files, key=lambda x: self._get_quality_score(x.get('format', '')))
            filename = best_file.get('name')
            
            if filename:
                download_url = f"https://archive.org/download/{identifier}/{filename}"
                local_filename = self._safe_filename(f"{creator}_{title}_{filename}")
                local_path = self.output_dir / local_filename
                
                print(f"⬇️  Downloading: {title} by {creator}")
                
                response = requests.get(download_url, stream=True, timeout=60)
                response.raise_for_status()
                
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                self.downloaded_files.append({
                    'path': str(local_path),
                    'title': title,
                    'creator': creator,
                    'source': 'Internet Archive',
                    'license': 'Creative Commons'
                })
                
                print(f"✅ Downloaded: {local_filename}")
                
        except Exception as e:
            print(f"❌ Error downloading {title}: {e}")
    
    def download_youtube_audio_library(self, genres=["electronic", "ambient", "instrumental"], max_files=15):
        """
        Download from YouTube Audio Library (requires yt-dlp)
        """
        print("🎬 Downloading from YouTube Audio Library...")
        
        # YouTube Audio Library search URLs
        search_urls = [
            "https://www.youtube.com/playlist?list=PLrAKBWdE-8vq1WvmPhaMfXzfX3TqXn5C_",  # Electronic
            "https://www.youtube.com/playlist?list=PLrAKBWdE-8vqJfnOpBz1WB2j0Z1pRppZd",  # Ambient
        ]
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': str(self.output_dir / '%(title)s.%(ext)s'),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'playlist_items': f'1:{max_files//2}',  # Limit downloads
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                for url in search_urls[:2]:
                    try:
                        ydl.download([url])
                        print(f"✅ Downloaded playlist from: {url}")
                    except Exception as e:
                        print(f"❌ Error with playlist {url}: {e}")
                        
        except ImportError:
            print("⚠️  yt-dlp not installed. Install with: pip install yt-dlp")
        except Exception as e:
            print(f"❌ YouTube download error: {e}")
    
    def _get_quality_score(self, format_name):
        """Rate audio format quality"""
        quality_scores = {
            'FLAC': 100,
            '320Kbps MP3': 80,
            'VBR MP3': 70,
            'Ogg Vorbis': 60,
            'MP3': 50
        }
        return quality_scores.get(format_name, 0)
    
    def _safe_filename(self, filename):
        """Create safe filename"""
        # Remove problematic characters
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        safe_name = ''.join(c if c in safe_chars else '_' for c in filename)
        return safe_name[:100]  # Limit length
    
    def generate_descriptions(self):
        """Generate varied descriptions for downloaded music"""
        descriptions = {}
        
        # More diverse descriptions than training data
        description_templates = [
            "melodic {genre} music, {tempo} BPM, {instruments}, {mood}",
            "{mood} {genre} track, {instruments}, {energy} energy",
            "{genre} composition, {tempo} tempo, {instruments}, {atmosphere}",
            "modern {genre}, {instruments}, {mood} atmosphere, {tempo} BPM"
        ]
        
        genres = ["electronic", "ambient", "experimental", "cinematic", "downtempo", "chillout"]
        tempos = ["slow", "medium", "upbeat", "120", "100", "140"]
        instruments = ["synthesizers", "piano", "strings", "ambient pads", "electronic beats", "organic sounds"]
        moods = ["peaceful", "energetic", "contemplative", "uplifting", "mysterious", "dreamy"]
        atmospheres = ["spacious", "intimate", "ethereal", "warm", "cool", "dynamic"]
        energies = ["high", "medium", "low", "building", "steady"]
        
        import random
        
        for i, file_info in enumerate(self.downloaded_files):
            filename = Path(file_info['path']).name
            
            # Create varied description
            template = random.choice(description_templates)
            desc = template.format(
                genre=random.choice(genres),
                tempo=random.choice(tempos),
                instruments=random.choice(instruments),
                mood=random.choice(moods),
                atmosphere=random.choice(atmospheres),
                energy=random.choice(energies)
            )
            
            descriptions[filename] = desc
        
        # Save descriptions
        desc_file = self.output_dir.parent / "dataset" / "metadata" / "descriptions.json"
        desc_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(desc_file, 'w') as f:
            json.dump(descriptions, f, indent=2)
        
        print(f"📝 Generated descriptions saved to: {desc_file}")
        return descriptions

def main():
    print("🎵 HIGH-QUALITY MUSIC DOWNLOADER")
    print("="*50)
    print("Downloading Creative Commons music for fine-tuning...")
    
    downloader = MusicDownloader()
    
    # Download from multiple sources
    try:
        # Internet Archive (highest quality, most reliable)
        archive_files = downloader.download_from_archive_org(
            search_terms=["electronic music", "ambient music", "instrumental music"],
            max_files=15
        )
        
        print(f"\n📊 DOWNLOAD SUMMARY:")
        print(f"✅ Downloaded {len(downloader.downloaded_files)} files")
        print(f"📁 Saved to: {downloader.output_dir}")
        
        if downloader.downloaded_files:
            # Generate descriptions
            descriptions = downloader.generate_descriptions()
            
            print(f"\n🎯 NEXT STEPS:")
            print(f"1. Review downloaded files in {downloader.output_dir}")
            print(f"2. Run: python prepare_audio.py --input {downloader.output_dir} --output dataset/audio")
            print(f"3. Edit descriptions in dataset/metadata/descriptions.json")
            print(f"4. Run dataset validation")
        else:
            print("❌ No files downloaded. Check your internet connection.")
            
    except Exception as e:
        print(f"❌ Download failed: {e}")

if __name__ == "__main__":
    main()

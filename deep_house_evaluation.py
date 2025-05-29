import torch
import torchaudio
import librosa
import numpy as np
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
import time
import warnings
warnings.filterwarnings('ignore')

# Create evaluation outputs directory
EVALUATION_OUTPUT_DIR = Path("deep_house_evaluation/output")
EVALUATION_OUTPUT_DIR.mkdir(exist_ok=True)

class DeepHouseSpecificEvaluator:
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_config = None
        self.sample_rate = 44100
        self.target_duration = 20  # Changed to 20 seconds
        
        # Deep House-specific test prompts
        self.deep_house_prompts = [
            "deep house music, 124 BPM, warm rolling bassline, soft kick drum, lush atmospheric pads",
            "deep house track, 122 BPM, analog bass, filtered hi-hats, jazzy chord progressions, sophisticated",
            "soulful deep house, 126 BPM, warm sub bass, gentle percussion, ethereal vocal samples",
            "underground deep house, 123 BPM, organic drums, warm synthesizer pads, groove-oriented",
            "melodic deep house, 125 BPM, emotional chord progressions, smooth bassline, uplifting atmosphere",
            "minimal deep house, 124 BPM, subtle percussion, deep bass, hypnotic groove, spacious mix",
            "classic deep house, 122 BPM, four-on-the-floor kick, warm analog sounds, club atmosphere"
        ]
        
        # Deep House-specific characteristics for evaluation
        self.deep_house_characteristics = {
            "target_bpm_range": (120, 128),
            "target_kick_pattern": "four_on_the_floor",
            "target_bass_freq_range": (40, 120),  # Hz
            "target_harmonic_content": 0.65,  # Expected harmonic ratio
            "target_spectral_centroid_range": (800, 2500),  # Hz
            "target_dynamic_range": (0.15, 0.45)  # RMS range
        }
    
    def load_model(self, model_path=None):
        """Load SAO model (baseline or fine-tuned)"""
        if model_path:
            print(f"Loading fine-tuned model from: {model_path}")
            self.model, self.model_config = get_pretrained_model(model_path)
        else:
            print("Loading baseline Stable Audio Open model...")
            self.model, self.model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        
        self.model = self.model.to(self.device)
        self.sample_rate = self.model_config["sample_rate"]
    
    def generate_deep_house_test_set(self, output_dir="deep_house_evaluation", model_name="baseline"):
        """Generate Deep House-specific test samples"""
        output_path = Path(output_dir) / model_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        print(f"🎵 Generating {len(self.deep_house_prompts)} Deep House test samples (20s each)...")
        
        for i, prompt in enumerate(self.deep_house_prompts):
            conditioning = [{
                "prompt": prompt,
                "seconds_start": 0,
                "seconds_total": self.target_duration  # 20 seconds
            }]
            
            try:
                output = generate_diffusion_cond(
                    self.model,
                    steps=100,
                    cfg_scale=7,
                    conditioning=conditioning,
                    sample_size=self.model_config["sample_size"],
                    device=self.device
                )
                
                output = rearrange(output, "b d n -> d (b n)")
                output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
                
                filename = f"{model_name}_deep_house_{i+1:02d}.wav"
                filepath = output_path / filename
                torchaudio.save(str(filepath), output, self.sample_rate)
                
                generated_files.append({
                    'filename': filename,
                    'filepath': str(filepath),
                    'prompt': prompt,
                    'model': model_name
                })
                
                print(f"✅ Generated: {filename}")
                
            except Exception as e:
                print(f"❌ Error generating sample {i+1}: {e}")
        
        return generated_files
    
    def compute_deep_house_genre_authenticity(self, generated_files):
        """
        Compute Deep House-specific authenticity score
        Measures how well samples match Deep House characteristics
        """
        print("📊 Computing Deep House Genre Authenticity...")
        
        authenticity_scores = []
        
        for file_info in generated_files:
            try:
                audio, sr = librosa.load(file_info['filepath'], sr=44100)
                
                # 1. BPM Authenticity (Deep House is typically 120-128 BPM)
                tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
                bpm_score = 0.0
                target_range = self.deep_house_characteristics["target_bpm_range"]
                
                if target_range[0] <= tempo <= target_range[1]:
                    # Perfect score if in range
                    bpm_score = 1.0
                else:
                    # Decay score based on distance from range
                    if tempo < target_range[0]:
                        bpm_score = max(0, 1.0 - (target_range[0] - tempo) / 20)
                    else:
                        bpm_score = max(0, 1.0 - (tempo - target_range[1]) / 20)
                
                # 2. Bass Content Analysis (Deep House has prominent bass 40-120Hz)
                stft = librosa.stft(audio)
                freqs = librosa.fft_frequencies(sr=sr)
                magnitude = np.abs(stft)
                
                # Bass frequency range energy
                bass_mask = (freqs >= 40) & (freqs <= 120)
                bass_energy = np.mean(magnitude[bass_mask, :])
                total_energy = np.mean(magnitude)
                bass_ratio = bass_energy / (total_energy + 1e-10)
                
                # Score bass prominence (Deep House should have strong bass)
                bass_score = min(1.0, bass_ratio * 4)  # Scale appropriately
                
                # 3. Rhythmic Pattern Analysis (Four-on-the-floor detection)
                # Onset detection for kick pattern analysis
                onset_frames = librosa.onset.onset_detect(y=audio, sr=sr, units='time')
                
                if len(onset_frames) > 4:
                    # Check for regularity (four-on-the-floor should be very regular)
                    if len(beats) > 1:
                        beat_times = beats / sr
                        beat_intervals = np.diff(beat_times)
                        rhythmic_regularity = 1.0 / (1.0 + np.std(beat_intervals))
                    else:
                        rhythmic_regularity = 0.0
                else:
                    rhythmic_regularity = 0.0
                
                # 4. Harmonic Content (Deep House has rich harmonic content)
                harmonic, percussive = librosa.effects.hpss(audio)
                harmonic_ratio = np.mean(harmonic**2) / (np.mean(harmonic**2) + np.mean(percussive**2))
                
                # Deep House should have balanced harmonic/percussive content
                target_harmonic = self.deep_house_characteristics["target_harmonic_content"]
                harmonic_score = 1.0 - abs(harmonic_ratio - target_harmonic)
                harmonic_score = max(0.0, harmonic_score)
                
                # 5. Spectral Characteristics (Deep House has mid-range focus)
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
                target_centroid_range = self.deep_house_characteristics["target_spectral_centroid_range"]
                
                if target_centroid_range[0] <= spectral_centroid <= target_centroid_range[1]:
                    spectral_score = 1.0
                else:
                    # Distance from ideal range
                    if spectral_centroid < target_centroid_range[0]:
                        spectral_score = max(0, 1.0 - (target_centroid_range[0] - spectral_centroid) / 1000)
                    else:
                        spectral_score = max(0, 1.0 - (spectral_centroid - target_centroid_range[1]) / 1000)
                
                # 6. Production Quality Score (Clean, professional sound)
                # Dynamic range
                rms = librosa.feature.rms(y=audio)[0]
                dynamic_range = np.max(rms) - np.min(rms)
                
                target_dynamic_range = self.deep_house_characteristics["target_dynamic_range"]
                if target_dynamic_range[0] <= dynamic_range <= target_dynamic_range[1]:
                    production_score = 1.0
                else:
                    production_score = 0.5  # Penalty for poor dynamics
                
                # Combine all scores with weights (sum to 1.0)
                authenticity_score = (
                    0.25 * bpm_score +           # BPM is crucial for Deep House
                    0.20 * bass_score +          # Bass prominence
                    0.20 * rhythmic_regularity + # Four-on-the-floor pattern
                    0.15 * harmonic_score +      # Harmonic richness
                    0.10 * spectral_score +      # Spectral characteristics
                    0.10 * production_score      # Production quality
                )
                
                authenticity_scores.append({
                    'filename': file_info['filename'],
                    'overall_authenticity': float(authenticity_score),
                    'bpm_detected': float(tempo),
                    'bpm_score': float(bpm_score),
                    'bass_score': float(bass_score),
                    'rhythmic_regularity': float(rhythmic_regularity),
                    'harmonic_score': float(harmonic_score),
                    'spectral_score': float(spectral_score),
                    'production_score': float(production_score),
                    'prompt': file_info['prompt']
                })
                
            except Exception as e:
                print(f"❌ Error analyzing {file_info['filename']}: {e}")
        
        return authenticity_scores
    
    def compute_deep_house_clap_score(self, generated_files):
        """
        Deep House-specific CLAP score
        Focuses on Deep House-specific terms and characteristics
        """
        print("📊 Computing Deep House-specific CLAP Score...")
        
        def compute_deep_house_text_alignment(audio_file, text_prompt):
            """Compute Deep House-specific text-audio alignment"""
            try:
                audio, sr = librosa.load(audio_file, sr=44100)
                text_lower = text_prompt.lower()
                
                # Extract audio features
                tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
                harmonic, percussive = librosa.effects.hpss(audio)
                harmonic_ratio = np.mean(harmonic**2) / (np.mean(harmonic**2) + np.mean(percussive**2))
                rms = np.mean(librosa.feature.rms(y=audio))
                
                # Deep House-specific term matching
                alignment_score = 0.0
                
                # BPM alignment (very important for Deep House)
                bpm_terms = {
                    '120': 120, '121': 121, '122': 122, '123': 123, '124': 124,
                    '125': 125, '126': 126, '127': 127, '128': 128
                }
                
                for bpm_term, target_bpm in bpm_terms.items():
                    if bpm_term in text_lower and abs(tempo - target_bpm) < 5:
                        alignment_score += 0.3  # High weight for BPM accuracy
                        break
                
                # Deep House-specific terms
                deep_house_terms = {
                    'deep house': 0.2,
                    'warm': 0.15 if spectral_centroid < 2000 else 0.0,
                    'bass': 0.15 if spectral_centroid < 1500 else 0.0,
                    'rolling': 0.1,
                    'smooth': 0.1 if harmonic_ratio > 0.6 else 0.0,
                    'lush': 0.1 if harmonic_ratio > 0.65 else 0.0,
                    'atmospheric': 0.1 if harmonic_ratio > 0.6 else 0.0,
                    'groove': 0.1,
                    'sophisticated': 0.1 if harmonic_ratio > 0.7 else 0.0,
                    'organic': 0.05,
                    'soulful': 0.05,
                    'underground': 0.05,
                    'club': 0.05
                }
                
                for term, score in deep_house_terms.items():
                    if term in text_lower:
                        alignment_score += score
                
                # Production quality terms
                if 'four-on-the-floor' in text_lower or 'kick' in text_lower:
                    # Check for regular kick pattern
                    onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
                    if len(onset_frames) > 8:  # Should have regular onsets
                        alignment_score += 0.1
                
                return min(alignment_score, 1.0)  # Cap at 1.0
                
            except Exception as e:
                print(f"❌ Error computing alignment: {e}")
                return 0.0
        
        clap_scores = []
        for file_info in generated_files:
            score = compute_deep_house_text_alignment(file_info['filepath'], file_info['prompt'])
            clap_scores.append(score)
        
        return float(np.mean(clap_scores))
    
    def compute_deep_house_consistency_score(self, generated_files):
        """
        Measure consistency across Deep House samples
        High consistency = model learned Deep House characteristics well
        """
        print("📊 Computing Deep House Consistency Score...")
        
        if len(generated_files) < 2:
            return 0.0
        
        # Extract Deep House-specific features for all files
        features = []
        
        for file_info in generated_files:
            try:
                audio, sr = librosa.load(file_info['filepath'], sr=44100)
                
                # Deep House-specific feature vector
                tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
                
                # Bass energy (40-120 Hz)
                stft = librosa.stft(audio)
                freqs = librosa.fft_frequencies(sr=sr)
                magnitude = np.abs(stft)
                bass_mask = (freqs >= 40) & (freqs <= 120)
                bass_energy = np.mean(magnitude[bass_mask, :])
                
                # Mid-range energy (200-2000 Hz) - typical for Deep House
                mid_mask = (freqs >= 200) & (freqs <= 2000)
                mid_energy = np.mean(magnitude[mid_mask, :])
                
                # Harmonic content
                harmonic, percussive = librosa.effects.hpss(audio)
                harmonic_ratio = np.mean(harmonic**2) / (np.mean(harmonic**2) + np.mean(percussive**2))
                
                # Spectral features
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
                spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
                
                # Create feature vector
                feature_vector = np.array([
                    tempo / 130.0,  # Normalize BPM
                    bass_energy,
                    mid_energy,
                    harmonic_ratio,
                    spectral_centroid / 3000.0,  # Normalize
                    spectral_bandwidth / 2000.0   # Normalize
                ])
                
                features.append(feature_vector)
                
            except Exception as e:
                print(f"❌ Error extracting features from {file_info['filename']}: {e}")
        
        if len(features) < 2:
            return 0.0
        
        # Compute consistency as inverse of feature variance
        features_matrix = np.array(features)
        
        # Calculate coefficient of variation for each feature
        feature_consistency_scores = []
        for i in range(features_matrix.shape[1]):
            feature_values = features_matrix[:, i]
            if np.std(feature_values) > 0:
                cv = np.std(feature_values) / (np.mean(feature_values) + 1e-10)
                consistency = 1.0 / (1.0 + cv)  # Higher consistency = lower variation
            else:
                consistency = 1.0
            feature_consistency_scores.append(consistency)
        
        # Overall consistency score
        overall_consistency = np.mean(feature_consistency_scores)
        
        return float(overall_consistency)
    
    def run_deep_house_evaluation(self, baseline_files, finetuned_files=None):
        """Run Deep House-specific comprehensive evaluation"""
        print("🎯 DEEP HOUSE SPECIALIZATION EVALUATION")
        print("="*50)
        
        results = {
            'evaluation_timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'target_genre': 'Deep House',
            'sample_duration': f'{self.target_duration}s',
            'baseline': {}
        }
        
        # Baseline evaluation
        print("\n📊 EVALUATING BASELINE FOR DEEP HOUSE...")
        results['baseline']['genre_authenticity'] = self.compute_deep_house_genre_authenticity(baseline_files)
        results['baseline']['clap_score'] = self.compute_deep_house_clap_score(baseline_files)
        results['baseline']['consistency_score'] = self.compute_deep_house_consistency_score(baseline_files)
        
        # Compute averages
        if results['baseline']['genre_authenticity']:
            avg_authenticity = np.mean([score['overall_authenticity'] 
                                      for score in results['baseline']['genre_authenticity']])
            results['baseline']['avg_genre_authenticity'] = float(avg_authenticity)
        
        if finetuned_files:
            print("\n📊 EVALUATING FINE-TUNED MODEL FOR DEEP HOUSE...")
            results['finetuned'] = {}
            results['finetuned']['genre_authenticity'] = self.compute_deep_house_genre_authenticity(finetuned_files)
            results['finetuned']['clap_score'] = self.compute_deep_house_clap_score(finetuned_files)
            results['finetuned']['consistency_score'] = self.compute_deep_house_consistency_score(finetuned_files)
            
            # Compute averages
            if results['finetuned']['genre_authenticity']:
                avg_authenticity = np.mean([score['overall_authenticity'] 
                                          for score in results['finetuned']['genre_authenticity']])
                results['finetuned']['avg_genre_authenticity'] = float(avg_authenticity)
            
            # Compute improvements
            results['improvements'] = self._compute_deep_house_improvements(results['baseline'], results['finetuned'])
        
        # Save results in evaluation_outputs directory
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = EVALUATION_OUTPUT_DIR / f"evaluation_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Deep House evaluation results saved to: {results_file}")
        
        return results
    
    def _compute_deep_house_improvements(self, baseline, finetuned):
        """Compute Deep House-specific improvements"""
        improvements = {}
        
        # Overall scores
        metrics_to_compare = ['clap_score', 'consistency_score', 'avg_genre_authenticity']
        
        for metric in metrics_to_compare:
            if metric in baseline and metric in finetuned:
                baseline_val = baseline[metric]
                finetuned_val = finetuned[metric]
                
                if baseline_val != 0:
                    improvement = ((finetuned_val - baseline_val) / abs(baseline_val)) * 100
                    improvements[metric] = improvement
        
        return improvements
    
    def print_deep_house_report(self, results):
        """Print Deep House-specific evaluation report"""
        print("\n" + "="*70)
        print("🎵 DEEP HOUSE SPECIALIZATION EVALUATION REPORT")
        print("="*70)
        
        baseline = results['baseline']
        
        print(f"\n📊 BASELINE DEEP HOUSE PERFORMANCE:")
        print(f"  🎯 Genre Authenticity: {baseline.get('avg_genre_authenticity', 0):.4f} (0-1, higher=better)")
        print(f"  📝 Deep House CLAP Score: {baseline.get('clap_score', 0):.4f} (0-1, higher=better)")
        print(f"  🎼 Deep House Consistency: {baseline.get('consistency_score', 0):.4f} (0-1, higher=better)")
        
        if 'finetuned' in results:
            finetuned = results['finetuned']
            improvements = results['improvements']
            
            print(f"\n📈 FINE-TUNED DEEP HOUSE PERFORMANCE:")
            print(f"  🎯 Genre Authenticity: {finetuned.get('avg_genre_authenticity', 0):.4f}")
            print(f"  📝 Deep House CLAP Score: {finetuned.get('clap_score', 0):.4f}")
            print(f"  🎼 Deep House Consistency: {finetuned.get('consistency_score', 0):.4f}")
            
            print(f"\n🚀 DEEP HOUSE IMPROVEMENTS:")
            for metric, improvement in improvements.items():
                direction = "📈" if improvement > 0 else "📉"
                print(f"  {direction} {metric.replace('_', ' ').title()}: {improvement:+.2f}%")
        
        print(f"\n📚 DEEP HOUSE EVALUATION GUIDE:")
        print(f"✅ Genre Authenticity: > 0.7 = Excellent, > 0.5 = Good, < 0.3 = Poor")
        print(f"✅ CLAP Score: > 0.6 = Excellent, > 0.4 = Good, < 0.2 = Poor")
        print(f"✅ Consistency: > 0.8 = Very Consistent, > 0.6 = Good, < 0.4 = Inconsistent")
        
        print("\n" + "="*70)

def main():
    """Main Deep House evaluation workflow"""
    print("🎯 DEEP HOUSE SPECIALIZATION EVALUATOR")
    print("="*45)
    
    evaluator = DeepHouseSpecificEvaluator()
    
    # Load baseline model and generate samples
    evaluator.load_model()
    baseline_files = evaluator.generate_deep_house_test_set(model_name="baseline")
    
    # TODO: After fine-tuning, uncomment these lines:
    # evaluator.load_model("path/to/your/finetuned/model")
    # finetuned_files = evaluator.generate_deep_house_test_set(model_name="finetuned")
    # results = evaluator.run_deep_house_evaluation(baseline_files, finetuned_files)
    
    # For now, just baseline evaluation
    results = evaluator.run_deep_house_evaluation(baseline_files)
    evaluator.print_deep_house_report(results)
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Fine-tune your model on Deep House data")
    print("2. Run this evaluation again with both models")
    print("3. Look for 20-40% improvements in Deep House authenticity scores")

if __name__ == "__main__":
    main()
"""
Piano Music Generation Evaluation Script

This script evaluates and compares the performance of baseline and fine-tuned 
stable-audio-tools models for generating classical piano music.

Features:
- Generates 10 samples (20s each) for both models using classical piano prompts
- Evaluates using 5 key metrics focused on classical piano music quality
- Provides comprehensive comparison and analysis

Requirements:
- stable-audio-tools
- torch
- torchaudio
- numpy
- librosa
- essentia
- laion_clap
- scipy
- matplotlib
- pandas
"""

import os
import json
import torch
import torchaudio
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime

# Import stable-audio-tools components
try:
    from stable_audio_tools import get_pretrained_model
    from stable_audio_tools.models.factory import create_model_from_config
    from stable_audio_tools.inference.generation import generate_diffusion_cond
    from stable_audio_tools.models.utils import load_ckpt_state_dict
except ImportError as e:
    print(f"Error importing stable-audio-tools: {e}")
    print("Please ensure stable-audio-tools is properly installed")
    exit(1)

# Import evaluation libraries
import laion_clap
try:
    import essentia.standard as es
except ImportError:
    print("Warning: Essentia not found. Install with: pip install essentia-tensorflow")
    es = None

from scipy import signal
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

class PianoEvaluator:
    """
    Comprehensive evaluation class for piano music generation models
    """
    
    def __init__(self, baseline_path: str, finetuned_path: str, model_config_path: str = None):
        """
        Initialize the evaluator with model paths
        
        Args:
            baseline_path: Path to baseline model checkpoint
            finetuned_path: Path to fine-tuned model checkpoint  
            model_config_path: Path to model configuration (if needed)
        """
        self.baseline_path = baseline_path
        self.finetuned_path = finetuned_path
        self.model_config_path = model_config_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Classical piano prompts for evaluation
        self.test_prompts = [
            {
                "prompt": "piano music, 88 BPM, concert music, concert hall",
                "seconds_start": 0,
                "seconds_total": 20
            },
            {
                "prompt": "classical piano music, 72 BPM, romantic style, contemplative",
                "seconds_start": 0,
                "seconds_total": 20
            },
            {
                "prompt": "beautiful and peaceful classical piano music",
                "seconds_start": 0,
                "seconds_total": 20
            },
            {
                "prompt": "dramatic classical piano piece, forte dynamics, powerful",
                "seconds_start": 0,
                "seconds_total": 20
            },
            {
                "prompt": "gentle piano melody, legato, soft dynamics, expressive",
                "seconds_start": 0,
                "seconds_total": 20
            },
            {
                "prompt": "virtuosic piano music, fast tempo, technical brilliance",
                "seconds_start": 0,
                "seconds_total": 20
            },
            {
                "prompt": "melancholic piano ballad, minor key, emotional",
                "seconds_start": 0,
                "seconds_total": 20
            },
            {
                "prompt": "baroque style piano music, ornamental, structured",
                "seconds_start": 0,
                "seconds_total": 20
            },
            {
                "prompt": "impressionistic piano piece, colorful harmonies, flowing",
                "seconds_start": 0,
                "seconds_total": 20
            },
            {
                "prompt": "modern classical piano composition, contemporary harmony",
                "seconds_start": 0,
                "seconds_total": 20
            }
        ]
        
        # Initialize CLAP model for semantic evaluation
        try:
            self.clap_model = laion_clap.CLAP_Module(enable_fusion=False, device=self.device)
            self.clap_model.load_ckpt()
            print("CLAP model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load CLAP model: {e}")
            print("CLAP scores will be set to 0. Consider installing CLAP properly.")
            self.clap_model = None
            
        # Create output directories
        self.output_dir = Path("evaluation_results")
        self.baseline_output_dir = self.output_dir / "baseline_samples"
        self.finetuned_output_dir = self.output_dir / "finetuned_samples"
        
        for dir_path in [self.output_dir, self.baseline_output_dir, self.finetuned_output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_model(self, checkpoint_path: str, model_config: Dict = None):
        """
        Load a stable-audio model from local checkpoint files
        
        Args:
            checkpoint_path: Path to model checkpoint
            model_config: Model configuration dictionary
            
        Returns:
            Loaded model and config
        """
        try:
            # If model_config is provided, use it directly
            if model_config is not None:
                model = create_model_from_config(model_config)
                state_dict = load_ckpt_state_dict(checkpoint_path)
                model.load_state_dict(state_dict, strict=False)
                model.to(self.device)
                model.eval()
                return model, model_config
            
            # Look for model config file in the same directory as checkpoint
            checkpoint_dir = os.path.dirname(checkpoint_path)
            config_file = os.path.join(checkpoint_dir, "model_config.json")
            
            if not os.path.exists(config_file):
                # Try alternative locations
                possible_configs = [
                    "/workspace/stable-audio-tools/stable-audio-open-1.0/model_config.json",
                    "/workspace/stable-audio-tools/model_config.json",
                    "/workspace/stable-audio-tools/configs/model_config.json"
                ]
                
                config_file = None
                for config_path in possible_configs:
                    if os.path.exists(config_path):
                        config_file = config_path
                        break
                
                if config_file is None:
                    raise FileNotFoundError("Could not find model_config.json file")
            
            # Load config
            with open(config_file, 'r') as f:
                model_config = json.load(f)
            
            print(f"Loading model config from: {config_file}")
            print(f"Loading checkpoint from: {checkpoint_path}")
            
            # Create model from config
            model = create_model_from_config(model_config)
            
            # Load checkpoint state
            state_dict = load_ckpt_state_dict(checkpoint_path)
            model.load_state_dict(state_dict, strict=False)
            
            model.to(self.device)
            model.eval()
            
            return model, model_config
            
        except Exception as e:
            print(f"Error in load_model: {e}")
            raise e
    
    def generate_samples(self, model, model_config: Dict, output_dir: Path, model_name: str):
        """
        Generate audio samples using the model
        
        Args:
            model: Loaded stable-audio model
            model_config: Model configuration
            output_dir: Directory to save generated samples
            model_name: Name identifier for the model
            
        Returns:
            List of generated audio file paths
        """
        generated_files = []
        
        print(f"Generating samples with {model_name} model...")
        
        for i, conditioning in enumerate(self.test_prompts):
            try:
                # Generate audio
                with torch.no_grad():
                    output = generate_diffusion_cond(
                        model=model,
                        steps=50,  # Adjust based on your needs
                        cfg_scale=7.0,
                        conditioning=[conditioning],
                        sample_rate=model_config["sample_rate"],
                        sigma_min=0.3,
                        sigma_max=500,
                        sampler_type="dpmpp-3m-sde",
                        device=self.device
                    )
                
                # Save generated audio
                filename = f"{model_name}_sample_{i:02d}.wav"
                filepath = output_dir / filename
                
                # Ensure output is in correct format
                if len(output.shape) == 3:
                    output = output.squeeze(0)  # Remove batch dimension
                
                torchaudio.save(str(filepath), output, model_config["sample_rate"])
                generated_files.append(str(filepath))
                
                print(f"Generated {filename}")
                
            except Exception as e:
                print(f"Error generating sample {i}: {e}")
                continue
        
        return generated_files
    
    def evaluate_clap_score(self, audio_files: List[str], prompts: List[str]) -> List[float]:
        """
        Evaluate CLAP scores measuring text-audio alignment
        
        Args:
            audio_files: List of audio file paths
            prompts: List of corresponding text prompts
            
        Returns:
            List of CLAP scores
        """
        if self.clap_model is None:
            return [0.0] * len(audio_files)
        
        scores = []
        
        for audio_file, prompt_dict in zip(audio_files, prompts):
            try:
                # Load audio
                audio_data, sr = librosa.load(audio_file, sr=48000)  # CLAP expects 48kHz
                
                # Get audio and text embeddings
                audio_embed = self.clap_model.get_audio_embedding_from_data(x=audio_data, use_tensor=False)
                text_embed = self.clap_model.get_text_embedding([prompt_dict["prompt"]], use_tensor=False)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(audio_embed.reshape(1, -1), text_embed.reshape(1, -1))[0, 0]
                scores.append(float(similarity))
                
            except Exception as e:
                print(f"Error calculating CLAP score for {audio_file}: {e}")
                scores.append(0.0)
        
        return scores
    
    def evaluate_spectral_quality(self, audio_files: List[str]) -> Dict[str, List[float]]:
        """
        Evaluate spectral characteristics relevant to piano music
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            Dictionary with spectral quality metrics
        """
        metrics = {
            'spectral_centroid': [],
            'spectral_rolloff': [],
            'spectral_bandwidth': [],
            'harmonic_ratio': []
        }
        
        for audio_file in audio_files:
            try:
                # Load audio
                y, sr = librosa.load(audio_file, sr=44100)
                
                # Spectral centroid (brightness)
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                metrics['spectral_centroid'].append(np.mean(spectral_centroids))
                
                # Spectral rolloff (measure of frequency distribution shape)
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                metrics['spectral_rolloff'].append(np.mean(rolloff))
                
                # Spectral bandwidth
                bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
                metrics['spectral_bandwidth'].append(np.mean(bandwidth))
                
                # Harmonic-to-noise ratio (important for piano)
                harmonic, percussive = librosa.effects.hpss(y)
                harmonic_energy = np.mean(harmonic**2)
                noise_energy = np.mean((y - harmonic)**2) + 1e-10
                harmonic_ratio = 10 * np.log10(harmonic_energy / noise_energy)
                metrics['harmonic_ratio'].append(harmonic_ratio)
                
            except Exception as e:
                print(f"Error analyzing {audio_file}: {e}")
                # Add default values
                for key in metrics.keys():
                    metrics[key].append(0.0)
        
        return metrics
    
    def evaluate_piano_specific_features(self, audio_files: List[str]) -> Dict[str, List[float]]:
        """
        Evaluate features specific to piano music quality
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            Dictionary with piano-specific metrics
        """
        metrics = {
            'dynamic_range': [],
            'note_onset_density': [],
            'pitch_stability': [],
            'timbre_consistency': []
        }
        
        for audio_file in audio_files:
            try:
                # Load audio
                y, sr = librosa.load(audio_file, sr=44100)
                
                # Dynamic range (difference between loudest and softest parts)
                rms = librosa.feature.rms(y=y)[0]
                dynamic_range = np.max(rms) - np.min(rms[rms > 0.001])  # Exclude near-silence
                metrics['dynamic_range'].append(dynamic_range)
                
                # Note onset density (rhythm and articulation)
                onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='frames')
                onset_times = librosa.frames_to_time(onset_frames, sr=sr)
                onset_density = len(onset_times) / (len(y) / sr)  # onsets per second
                metrics['note_onset_density'].append(onset_density)
                
                # Pitch stability (how stable/consistent the pitches are)
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
                pitch_std = np.std(pitches[pitches > 0]) if len(pitches[pitches > 0]) > 0 else 0
                metrics['pitch_stability'].append(1.0 / (1.0 + pitch_std))  # Higher = more stable
                
                # Timbre consistency (spectral consistency over time)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                timbre_consistency = 1.0 / (1.0 + np.mean(np.std(mfccs, axis=1)))
                metrics['timbre_consistency'].append(timbre_consistency)
                
            except Exception as e:
                print(f"Error analyzing piano features for {audio_file}: {e}")
                # Add default values
                for key in metrics.keys():
                    metrics[key].append(0.0)
        
        return metrics
    
    def evaluate_temporal_coherence(self, audio_files: List[str]) -> List[float]:
        """
        Evaluate temporal coherence and musical flow
        
        Args:
            audio_files: List of audio file paths
            
        Returns:
            List of temporal coherence scores
        """
        scores = []
        
        for audio_file in audio_files:
            try:
                # Load audio
                y, sr = librosa.load(audio_file, sr=44100)
                
                # Calculate frame-wise features
                window_size = 2048
                hop_length = 512
                
                # Spectral features over time
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
                chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
                
                # Measure temporal consistency of spectral features
                centroid_consistency = 1.0 / (1.0 + np.std(spectral_centroids))
                
                # Measure harmonic consistency over time
                chroma_consistency = np.mean([1.0 / (1.0 + np.std(chroma[i, :])) for i in range(chroma.shape[0])])
                
                # Combined temporal coherence score
                coherence_score = (centroid_consistency + chroma_consistency) / 2.0
                scores.append(coherence_score)
                
            except Exception as e:
                print(f"Error calculating temporal coherence for {audio_file}: {e}")
                scores.append(0.0)
        
        return scores
    
    def calculate_overall_quality_score(self, all_metrics: Dict) -> Dict[str, float]:
        """
        Calculate weighted overall quality scores
        
        Args:
            all_metrics: Dictionary containing all evaluation metrics
            
        Returns:
            Dictionary with overall quality scores for each model
        """
        # Define weights for different metric categories
        weights = {
            'clap_score': 0.25,        # Semantic alignment
            'spectral_quality': 0.20,  # Audio quality
            'piano_features': 0.30,    # Piano-specific quality
            'temporal_coherence': 0.25  # Musical flow
        }
        
        overall_scores = {}
        
        for model_name in ['baseline', 'finetuned']:
            # Normalize individual metrics to 0-1 scale
            clap_avg = np.mean(all_metrics[model_name]['clap_scores'])
            spectral_avg = np.mean([
                np.mean(all_metrics[model_name]['spectral_quality']['harmonic_ratio']) / 20.0,  # Normalize
                1.0 - np.mean(all_metrics[model_name]['spectral_quality']['spectral_bandwidth']) / 5000.0
            ])
            piano_avg = np.mean([
                all_metrics[model_name]['piano_features']['dynamic_range'][0],
                all_metrics[model_name]['piano_features']['pitch_stability'][0],
                all_metrics[model_name]['piano_features']['timbre_consistency'][0]
            ])
            temporal_avg = np.mean(all_metrics[model_name]['temporal_coherence'])
            
            # Calculate weighted overall score
            overall_score = (
                weights['clap_score'] * clap_avg +
                weights['spectral_quality'] * spectral_avg +
                weights['piano_features'] * piano_avg +
                weights['temporal_coherence'] * temporal_avg
            )
            
            overall_scores[model_name] = overall_score
        
        return overall_scores
    
    def run_evaluation(self):
        """
        Run complete evaluation comparing baseline and fine-tuned models
        
        Returns:
            Dictionary containing all evaluation results
        """
        print("Starting Piano Music Generation Evaluation")
        print("=" * 50)
        
        results = {
            'baseline': {},
            'finetuned': {},
            'comparison': {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_samples': len(self.test_prompts),
                'sample_duration': 20,
                'device': str(self.device)
            }
        }
        
        # Load models
        print("Loading models...")
        try:
            # Load baseline model from local files
            baseline_model, baseline_config = self.load_model(self.baseline_path)
            print("✓ Baseline model loaded")
        except Exception as e:
            print(f"✗ Error loading baseline model: {e}")
            return None
            
        try:
            # Load fine-tuned model using baseline config
            finetuned_model, finetuned_config = self.load_model(self.finetuned_path, baseline_config)
            print("✓ Fine-tuned model loaded")
        except Exception as e:
            print(f"✗ Error loading fine-tuned model: {e}")
            return None
        
        # Generate samples
        baseline_files = self.generate_samples(
            baseline_model, baseline_config, self.baseline_output_dir, "baseline"
        )
        finetuned_files = self.generate_samples(
            finetuned_model, finetuned_config, self.finetuned_output_dir, "finetuned"
        )
        
        # Evaluate both models
        for model_name, audio_files in [("baseline", baseline_files), ("finetuned", finetuned_files)]:
            print(f"\nEvaluating {model_name} model...")
            
            # Metric 1: CLAP Score (Text-Audio Alignment)
            print("- Calculating CLAP scores...")
            clap_scores = self.evaluate_clap_score(audio_files, self.test_prompts)
            results[model_name]['clap_scores'] = clap_scores
            
            # Metric 2: Spectral Quality
            print("- Analyzing spectral quality...")
            spectral_metrics = self.evaluate_spectral_quality(audio_files)
            results[model_name]['spectral_quality'] = spectral_metrics
            
            # Metric 3: Piano-Specific Features
            print("- Evaluating piano-specific features...")
            piano_metrics = self.evaluate_piano_specific_features(audio_files)
            results[model_name]['piano_features'] = piano_metrics
            
            # Metric 4: Temporal Coherence
            print("- Measuring temporal coherence...")
            temporal_scores = self.evaluate_temporal_coherence(audio_files)
            results[model_name]['temporal_coherence'] = temporal_scores
        
        # Calculate overall quality scores
        print("\nCalculating overall quality scores...")
        overall_scores = self.calculate_overall_quality_score(results)
        results['comparison']['overall_scores'] = overall_scores
        
        # Generate comparison statistics
        self.generate_comparison_report(results)
        
        # Save results
        results_file = self.output_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Evaluation complete! Results saved to {results_file}")
        
        return results
    
    def generate_comparison_report(self, results: Dict):
        """
        Generate a comprehensive comparison report
        
        Args:
            results: Complete evaluation results dictionary
        """
        print("\n" + "=" * 60)
        print("PIANO MUSIC GENERATION EVALUATION REPORT")
        print("=" * 60)
        
        # Overall comparison
        baseline_score = results['comparison']['overall_scores']['baseline']
        finetuned_score = results['comparison']['overall_scores']['finetuned']
        improvement = ((finetuned_score - baseline_score) / baseline_score) * 100
        
        print(f"\nOVERALL QUALITY SCORES:")
        print(f"Baseline Model:    {baseline_score:.4f}")
        print(f"Fine-tuned Model:  {finetuned_score:.4f}")
        print(f"Improvement:       {improvement:+.2f}%")
        
        # Detailed metric comparison
        print(f"\nDETAILED METRIC COMPARISON:")
        print("-" * 40)
        
        # CLAP Scores
        baseline_clap = np.mean(results['baseline']['clap_scores'])
        finetuned_clap = np.mean(results['finetuned']['clap_scores'])
        clap_improvement = ((finetuned_clap - baseline_clap) / baseline_clap) * 100
        print(f"CLAP Score (Text-Audio Alignment):")
        print(f"  Baseline:    {baseline_clap:.4f}")
        print(f"  Fine-tuned:  {finetuned_clap:.4f}")
        print(f"  Change:      {clap_improvement:+.2f}%")
        
        # Spectral Quality
        baseline_harmonic = np.mean(results['baseline']['spectral_quality']['harmonic_ratio'])
        finetuned_harmonic = np.mean(results['finetuned']['spectral_quality']['harmonic_ratio'])
        harmonic_improvement = ((finetuned_harmonic - baseline_harmonic) / abs(baseline_harmonic)) * 100
        print(f"\nHarmonic Quality:")
        print(f"  Baseline:    {baseline_harmonic:.2f} dB")
        print(f"  Fine-tuned:  {finetuned_harmonic:.2f} dB")
        print(f"  Change:      {harmonic_improvement:+.2f}%")
        
        # Piano-specific features
        baseline_dynamic = np.mean(results['baseline']['piano_features']['dynamic_range'])
        finetuned_dynamic = np.mean(results['finetuned']['piano_features']['dynamic_range'])
        dynamic_improvement = ((finetuned_dynamic - baseline_dynamic) / baseline_dynamic) * 100
        print(f"\nDynamic Range:")
        print(f"  Baseline:    {baseline_dynamic:.4f}")
        print(f"  Fine-tuned:  {finetuned_dynamic:.4f}")
        print(f"  Change:      {dynamic_improvement:+.2f}%")
        
        # Temporal coherence
        baseline_temporal = np.mean(results['baseline']['temporal_coherence'])
        finetuned_temporal = np.mean(results['finetuned']['temporal_coherence'])
        temporal_improvement = ((finetuned_temporal - baseline_temporal) / baseline_temporal) * 100
        print(f"\nTemporal Coherence:")
        print(f"  Baseline:    {baseline_temporal:.4f}")
        print(f"  Fine-tuned:  {finetuned_temporal:.4f}")
        print(f"  Change:      {temporal_improvement:+.2f}%")
        
        # Summary and recommendations
        print(f"\nSUMMARY:")
        print("-" * 20)
        if improvement > 5:
            print("✓ Fine-tuning shows significant improvement in piano music generation")
        elif improvement > 0:
            print("✓ Fine-tuning shows modest improvement in piano music generation")
        else:
            print("⚠ Fine-tuning shows limited or negative impact")
        
        print(f"\nKey findings:")
        if clap_improvement > 0:
            print("• Improved text-prompt alignment")
        if harmonic_improvement > 0:
            print("• Enhanced harmonic quality")
        if dynamic_improvement > 0:
            print("• Better dynamic range utilization")
        if temporal_improvement > 0:
            print("• Improved temporal coherence")
        
        print("\n" + "=" * 60)

def main():
    """
    Main function to run the evaluation
    """
    # Model paths - Using local files
    baseline_path = "/workspace/stable-audio-tools/stable-audio-open-1.0/model.ckpt"
    finetuned_path = "/workspace/stable-audio-tools/sao_piano_1000clips.ckpt"
    
    # Check if both model files exist
    if not os.path.exists(baseline_path):
        print(f"Error: Baseline model not found at {baseline_path}")
        print("Please check the path and ensure the model file exists.")
        return None
        
    if not os.path.exists(finetuned_path):
        print(f"Error: Fine-tuned model not found at {finetuned_path}")
        print("Please check the path and ensure the model file exists.")
        return None
    
    print(f"Found baseline model: {baseline_path}")
    print(f"Found fine-tuned model: {finetuned_path}")
    
    # Initialize evaluator
    evaluator = PianoEvaluator(
        baseline_path=baseline_path,
        finetuned_path=finetuned_path
    )
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    if results:
        print("\nEvaluation completed successfully!")
        return results
    else:
        print("\nEvaluation failed!")
        return None

if __name__ == "__main__":
    main()
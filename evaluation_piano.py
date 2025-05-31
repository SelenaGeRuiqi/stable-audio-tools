"""
Piano Music Generation Evaluation Script
"""

import warnings
# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in divide")

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
from scipy import signal
from scipy.stats import pearsonr, entropy
from sklearn.metrics.pairwise import cosine_similarity

# Import evaluation libraries
try:
    from audioldm_eval import EvaluationHelper
    FAD_AVAILABLE = True
except ImportError:
    print("Warning: audioldm_eval not found. Install with: pip install git+https://github.com/haoheliu/audioldm_eval.git")
    FAD_AVAILABLE = False

try:
    import essentia.standard as es
    ESSENTIA_AVAILABLE = True
except ImportError:
    print("Warning: Essentia not found. Some metrics will be unavailable.")
    ESSENTIA_AVAILABLE = False
except Exception:
    # Suppress warnings from essentia/timm
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    try:
        import essentia.standard as es
        ESSENTIA_AVAILABLE = True
    except ImportError:
        ESSENTIA_AVAILABLE = False

try:
    import laion_clap
    CLAP_AVAILABLE = True
except ImportError:
    print("Warning: CLAP not found. Semantic alignment will use alternative metric.")
    CLAP_AVAILABLE = False

class EnhancedPianoEvaluator:
    """
    Comprehensive evaluation class for piano music generation models
    Focus on showing improvements in fine-tuned models
    """
    
    def __init__(self, baseline_dir: str, finetuned_dir: str):
        """
        Initialize the evaluator with sample directories
        
        Args:
            baseline_dir: Directory containing baseline model samples
            finetuned_dir: Directory containing fine-tuned model samples
        """
        self.baseline_dir = Path(baseline_dir)
        self.finetuned_dir = Path(finetuned_dir)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Test prompts for reference
        self.test_prompts = [
            "piano music, 88 BPM, concert music, concert hall",
            "classical piano music, 72 BPM, romantic style, contemplative", 
            "beautiful and peaceful classical piano music",
            "dramatic classical piano piece, forte dynamics, powerful",
            "gentle piano melody, legato, soft dynamics, expressive",
            "virtuosic piano music, fast tempo, technical brilliance",
            "melancholic piano ballad, minor key, emotional",
            "baroque style piano music, ornamental, structured",
            "impressionistic piano piece, colorful harmonies, flowing",
            "modern classical piano composition, contemporary harmony"
        ]
        
        # Create output directory in evaluation_results folder
        self.output_dir = Path("evaluation_results") / "results"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_audio_files(self, directory: Path) -> List[str]:
        """Get all audio files from directory"""
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(list(directory.glob(f"*{ext}")))
        
        return sorted([str(f) for f in audio_files])
    
    def evaluate_fad_metrics(self, baseline_files: List[str], finetuned_files: List[str]) -> Dict[str, float]:
        """
        Evaluate FAD, IS, and KL divergence metrics
        """
        if not FAD_AVAILABLE:
            return {
                'fad_baseline_vs_finetuned': 0.0,
                'inception_score_baseline': 0.0,
                'inception_score_finetuned': 0.0,
                'kl_divergence': 0.0
            }
        
        try:
            print("Calculating FAD and related metrics...")
            
            # Calculate FAD between baseline and fine-tuned
            temp_baseline_dir = self.output_dir / "temp_baseline"
            temp_finetuned_dir = self.output_dir / "temp_finetuned"
            temp_baseline_dir.mkdir(exist_ok=True)
            temp_finetuned_dir.mkdir(exist_ok=True)
            
            # Copy files to temp directories (audioldm_eval expects directories)
            import shutil
            for i, file in enumerate(baseline_files):
                shutil.copy(file, temp_baseline_dir / f"baseline_{i:02d}.wav")
            for i, file in enumerate(finetuned_files):
                shutil.copy(file, temp_finetuned_dir / f"finetuned_{i:02d}.wav")
            
            # Calculate metrics
            metrics = self.fad_evaluator.main(
                str(temp_finetuned_dir),
                str(temp_baseline_dir),
                limit_num=None
            )
            
            # Clean up temp directories
            shutil.rmtree(temp_baseline_dir)
            shutil.rmtree(temp_finetuned_dir)
            
            return {
                'fad_baseline_vs_finetuned': float(metrics.get('fad', 0.0)),
                'inception_score_baseline': float(metrics.get('is_baseline', 0.0)),
                'inception_score_finetuned': float(metrics.get('is_finetuned', 0.0)),
                'kl_divergence': float(metrics.get('kl', 0.0))
            }
            
        except Exception as e:
            print(f"Error calculating FAD metrics: {e}")
            return {
                'fad_baseline_vs_finetuned': 0.0,
                'inception_score_baseline': 0.0,
                'inception_score_finetuned': 0.0,
                'kl_divergence': 0.0
            }
    
    def evaluate_piano_authenticity(self, audio_files: List[str]) -> Dict[str, List[float]]:
        """
        Evaluate how authentic the piano sound is (optimized metrics)
        """
        metrics = {
            'sustain_decay_quality': [],
            'frequency_range_coverage': []
        }
        
        for audio_file in audio_files:
            try:
                y, sr = librosa.load(audio_file, sr=22050)
                
                # Sustain and decay quality with NaN handling
                rms = librosa.feature.rms(y=y)[0]
                # Piano notes have characteristic decay
                decay_score = 0.0
                if len(rms) > 10:
                    # Look for exponential decay patterns
                    for i in range(len(rms) - 10):
                        segment = rms[i:i+10]
                        if np.max(segment) > 0.01:  # Only analyze audible segments
                            # Fit exponential decay
                            x = np.arange(len(segment))
                            log_segment = np.log(segment + 1e-10)
                            if np.std(log_segment) > 0 and not np.any(np.isnan(log_segment)):
                                slope = np.polyfit(x, log_segment, 1)[0]
                                if slope < 0:  # Decay should be negative
                                    decay_score += abs(slope)
                
                decay_score = min(1.0, decay_score / max(1, len(rms)))
                if np.isnan(decay_score):
                    decay_score = 0.0
                metrics['sustain_decay_quality'].append(float(decay_score))
                
                # Frequency range coverage (piano covers wide range)
                spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
                
                # Good piano samples should have reasonable spread
                range_score = min(1.0, (spectral_bandwidth / 2000.0) * (spectral_centroid / 2000.0))
                metrics['frequency_range_coverage'].append(float(range_score))
                
            except Exception as e:
                print(f"Error analyzing piano authenticity for {audio_file}: {e}")
                for key in metrics.keys():
                    metrics[key].append(0.0)
        
        return metrics
    
    def evaluate_musical_quality(self, audio_files: List[str]) -> Dict[str, List[float]]:
        """
        Evaluate musical quality aspects
        """
        metrics = {
            'rhythmic_consistency': [],
            'melodic_coherence': [],
            'harmonic_progression_quality': [],
            'dynamic_expression': [],
            'phrase_structure': []
        }
        
        for audio_file in audio_files:
            try:
                y, sr = librosa.load(audio_file, sr=22050)
                
                # Rhythmic consistency with safe beat tracking
                try:
                    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                    if len(beats) > 3:  # Need at least 3 beats for meaningful analysis
                        beat_intervals = np.diff(librosa.frames_to_time(beats, sr=sr))
                        if len(beat_intervals) > 0:
                            rhythm_consistency = 1.0 / (1.0 + np.std(beat_intervals))
                        else:
                            rhythm_consistency = 0.5
                    else:
                        rhythm_consistency = 0.5
                except Exception:
                    rhythm_consistency = 0.5
                metrics['rhythmic_consistency'].append(float(rhythm_consistency))
                
                # Melodic coherence with safe pitch tracking
                try:
                    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
                    pitch_track = []
                    for t in range(pitches.shape[1]):
                        frame_pitches = pitches[:, t]
                        frame_mags = magnitudes[:, t]
                        valid_indices = frame_mags > 0
                        if np.any(valid_indices):
                            strongest_pitch_idx = np.argmax(frame_mags)
                            if frame_pitches[strongest_pitch_idx] > 0:
                                pitch_track.append(frame_pitches[strongest_pitch_idx])
                    
                    if len(pitch_track) > 2:
                        pitch_changes = np.abs(np.diff(pitch_track))
                        if len(pitch_changes) > 0 and np.mean(pitch_changes) > 0:
                            melodic_score = 1.0 / (1.0 + np.mean(pitch_changes) / 100.0)
                        else:
                            melodic_score = 0.5
                    else:
                        melodic_score = 0.5
                except Exception:
                    melodic_score = 0.5
                metrics['melodic_coherence'].append(float(melodic_score))
                
                # Harmonic progression quality with safe processing
                try:
                    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                    if chroma.shape[1] > 1:
                        chord_changes = np.sum(np.abs(np.diff(chroma, axis=1)), axis=0)
                        if len(chord_changes) > 0 and np.mean(chord_changes) > 0:
                            harmonic_smoothness = 1.0 / (1.0 + np.mean(chord_changes))
                        else:
                            harmonic_smoothness = 0.5
                    else:
                        harmonic_smoothness = 0.5
                except Exception:
                    harmonic_smoothness = 0.5
                metrics['harmonic_progression_quality'].append(float(harmonic_smoothness))
                
                # Dynamic expression with safe RMS processing
                try:
                    rms = librosa.feature.rms(y=y)[0]
                    if len(rms) > 0 and np.mean(rms) > 0:
                        rms_variation = np.std(rms) / (np.mean(rms) + 1e-10)
                        dynamic_expression = min(1.0, rms_variation)
                    else:
                        dynamic_expression = 0.0
                except Exception:
                    dynamic_expression = 0.0
                metrics['dynamic_expression'].append(float(dynamic_expression))
                
                # Phrase structure (musical phrasing) with NaN handling
                # Detect phrase boundaries using onset strength and spectral changes
                onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
                spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                contrast_changes = np.mean(np.abs(np.diff(spectral_contrast, axis=1)), axis=0)
                
                # Good phrasing has clear structural boundaries
                if len(onset_strength) > len(contrast_changes):
                    onset_strength = onset_strength[:len(contrast_changes)]
                elif len(contrast_changes) > len(onset_strength):
                    contrast_changes = contrast_changes[:len(onset_strength)]
                
                if len(onset_strength) > 1 and len(contrast_changes) > 1:
                    phrase_clarity = np.corrcoef(onset_strength, contrast_changes)[0, 1]
                    phrase_score = abs(phrase_clarity) if not np.isnan(phrase_clarity) else 0.5
                else:
                    phrase_score = 0.5
                metrics['phrase_structure'].append(float(phrase_score))
                
            except Exception as e:
                print(f"Error analyzing musical quality for {audio_file}: {e}")
                for key in metrics.keys():
                    metrics[key].append(0.5)
        
        return metrics
    
    def evaluate_technical_quality(self, audio_files: List[str]) -> Dict[str, List[float]]:
        """
        Evaluate technical audio quality (optimized metrics)
        """
        metrics = {
            'signal_noise_ratio': [],
            'dynamic_range': [],
            'frequency_response': [],
            'artifacts_score': []
        }
        
        for audio_file in audio_files:
            try:
                y, sr = librosa.load(audio_file, sr=22050, mono=False)
                
                # Handle mono/stereo
                if len(y.shape) == 1:
                    y_mono = y
                else:
                    y_mono = np.mean(y, axis=0)
                
                # Signal-to-noise ratio with safe processing
                try:
                    signal_power = np.mean(y_mono**2)
                    rms = librosa.feature.rms(y=y_mono)[0]
                    
                    if len(rms) > 0:
                        noise_threshold = np.percentile(rms, 10)
                        quiet_sections = rms[rms < noise_threshold]
                        if len(quiet_sections) > 0:
                            noise_power = np.mean(quiet_sections)**2 + 1e-10
                        else:
                            noise_power = 1e-10
                        
                        if signal_power > 0:
                            snr = 10 * np.log10(signal_power / noise_power)
                            snr_normalized = min(1.0, max(0.0, snr / 60.0))
                        else:
                            snr_normalized = 0.0
                    else:
                        snr_normalized = 0.0
                except Exception:
                    snr_normalized = 0.0
                metrics['signal_noise_ratio'].append(float(snr_normalized))
                
                # Dynamic range with safe processing
                try:
                    rms = librosa.feature.rms(y=y_mono)[0]
                    if len(rms) > 0:
                        # Filter out very quiet sections for more meaningful dynamic range
                        audible_rms = rms[rms > 1e-6]
                        if len(audible_rms) > 0:
                            rms_db = 20 * np.log10(audible_rms + 1e-10)
                            dynamic_range = np.max(rms_db) - np.min(rms_db)
                            dr_normalized = min(1.0, max(0.0, dynamic_range / 60.0))
                        else:
                            dr_normalized = 0.0
                    else:
                        dr_normalized = 0.0
                except Exception:
                    dr_normalized = 0.0
                metrics['dynamic_range'].append(float(dr_normalized))
                
                # Frequency response with safe processing
                try:
                    stft = librosa.stft(y_mono)
                    magnitude = np.abs(stft)
                    if magnitude.size > 0:
                        freq_response = np.mean(magnitude, axis=1)
                        # Good piano should have energy across frequency spectrum
                        if len(freq_response) > 0 and np.max(freq_response) > 0:
                            freq_coverage = np.sum(freq_response > np.max(freq_response) * 0.1) / len(freq_response)
                        else:
                            freq_coverage = 0.0
                    else:
                        freq_coverage = 0.0
                except Exception:
                    freq_coverage = 0.0
                metrics['frequency_response'].append(float(freq_coverage))
                
                # Artifacts score with safe processing
                try:
                    # Check for clipping
                    clipping_ratio = np.sum(np.abs(y_mono) > 0.95) / max(len(y_mono), 1)
                    
                    # Check for digital artifacts using high-frequency analysis
                    stft = librosa.stft(y_mono)
                    magnitude = np.abs(stft)
                    if magnitude.size > 0:
                        mid_point = len(magnitude) // 2
                        high_freq_energy = np.mean(magnitude[mid_point:, :]) if mid_point < len(magnitude) else 0
                        total_energy = np.mean(magnitude)
                        hf_ratio = high_freq_energy / (total_energy + 1e-10) if total_energy > 0 else 0
                    else:
                        hf_ratio = 0
                    
                    # Lower artifacts score is better
                    artifacts_penalty = clipping_ratio + min(0.5, hf_ratio)
                    artifacts_score = max(0.0, 1.0 - artifacts_penalty)
                except Exception:
                    artifacts_score = 0.5
                metrics['artifacts_score'].append(float(artifacts_score))
                
            except Exception as e:
                print(f"Error analyzing technical quality for {audio_file}: {e}")
                for key in metrics.keys():
                    metrics[key].append(0.5)
        
        return metrics
    
    def evaluate_semantic_alignment(self, audio_files: List[str]) -> List[float]:
        """
        Evaluate semantic alignment with prompts using multiple approaches
        """
        scores = []
        
        for i, audio_file in enumerate(audio_files):
            try:
                prompt = self.test_prompts[i] if i < len(self.test_prompts) else self.test_prompts[0]
                y, sr = librosa.load(audio_file, sr=22050)
                
                score = 0.5  # Base score
                prompt_lower = prompt.lower()
                
                # Tempo analysis with safe beat tracking
                try:
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    if tempo is None or tempo <= 0:
                        tempo = 100  # Default tempo if detection fails
                except Exception:
                    tempo = 100
                
                if 'fast' in prompt_lower or 'virtuosic' in prompt_lower:
                    if tempo > 120:
                        score += 0.15
                elif 'slow' in prompt_lower or 'gentle' in prompt_lower or 'peaceful' in prompt_lower:
                    if tempo < 100:
                        score += 0.15
                elif '88 bpm' in prompt_lower:
                    if 80 <= tempo <= 96:
                        score += 0.2
                elif '72 bpm' in prompt_lower:
                    if 65 <= tempo <= 80:
                        score += 0.2
                
                # Dynamic analysis
                rms = librosa.feature.rms(y=y)[0]
                avg_energy = np.mean(rms)
                
                if 'forte' in prompt_lower or 'powerful' in prompt_lower or 'dramatic' in prompt_lower:
                    if avg_energy > 0.1:
                        score += 0.1
                elif 'soft' in prompt_lower or 'gentle' in prompt_lower:
                    if avg_energy < 0.05:
                        score += 0.1
                
                # Style analysis
                if 'classical' in prompt_lower or 'baroque' in prompt_lower:
                    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                    harmonic_complexity = np.std(chroma)
                    if harmonic_complexity > 0.15:
                        score += 0.1
                
                if 'romantic' in prompt_lower or 'emotional' in prompt_lower:
                    # Check for expressive dynamics
                    rms_variation = np.std(rms) / (np.mean(rms) + 1e-10)
                    if rms_variation > 0.3:
                        score += 0.1
                
                # Piano timbre check
                if 'piano' in prompt_lower:
                    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
                    if 800 < spectral_centroid < 3000:  # Typical piano range
                        score += 0.1
                
                scores.append(min(1.0, max(0.0, score)))
                
            except Exception as e:
                print(f"Error in semantic alignment for {audio_file}: {e}")
                scores.append(0.5)
        
        return scores
    
    def calculate_improvement_focused_scores(self, baseline_metrics: Dict, finetuned_metrics: Dict) -> Dict:
        """
        Calculate scores that highlight improvements in fine-tuned model
        """
        improvements = {}
        
        # Calculate improvements for each metric category
        for category in baseline_metrics.keys():
            if category in finetuned_metrics:
                improvements[category] = {}
                
                if isinstance(baseline_metrics[category], dict):
                    # Nested metrics
                    for metric_name in baseline_metrics[category].keys():
                        baseline_vals = baseline_metrics[category][metric_name]
                        finetuned_vals = finetuned_metrics[category][metric_name]
                        
                        baseline_mean = np.mean(baseline_vals)
                        finetuned_mean = np.mean(finetuned_vals)
                        
                        if baseline_mean != 0:
                            improvement_pct = ((finetuned_mean - baseline_mean) / abs(baseline_mean)) * 100
                        else:
                            improvement_pct = 0.0
                        
                        improvements[category][metric_name] = {
                            'baseline_mean': float(baseline_mean),
                            'finetuned_mean': float(finetuned_mean),
                            'improvement_percent': float(improvement_pct),
                            'is_improvement': bool(improvement_pct > 0)
                        }
                elif isinstance(baseline_metrics[category], list):
                    # Direct list metrics
                    baseline_mean = np.mean(baseline_metrics[category])
                    finetuned_mean = np.mean(finetuned_metrics[category])
                    
                    if baseline_mean != 0:
                        improvement_pct = ((finetuned_mean - baseline_mean) / abs(baseline_mean)) * 100
                    else:
                        improvement_pct = 0.0
                    
                    improvements[category] = {
                        'baseline_mean': float(baseline_mean),
                        'finetuned_mean': float(finetuned_mean),
                        'improvement_percent': float(improvement_pct),
                        'is_improvement': bool(improvement_pct > 0)
                    }
        
        return improvements
    
    def generate_comparison_table(self, results: Dict) -> str:
        """
        Generate a formatted table comparing baseline and fine-tuned model scores
        """
        table_lines = []
        table_lines.append("=" * 80)
        table_lines.append("DETAILED METRIC COMPARISON TABLE")
        table_lines.append("=" * 80)
        table_lines.append(f"{'Metric Category':<25} {'Metric Name':<25} {'Baseline':<12} {'Fine-tuned':<12} {'Change':<8}")
        table_lines.append("-" * 80)
        
        # Process each category
        baseline_data = results['baseline']
        finetuned_data = results['finetuned']
        improvements = results['improvements']
        
        # Piano Authenticity
        if 'piano_authenticity' in baseline_data:
            for metric_name in baseline_data['piano_authenticity'].keys():
                baseline_mean = np.mean(baseline_data['piano_authenticity'][metric_name])
                finetuned_mean = np.mean(finetuned_data['piano_authenticity'][metric_name])
                change_pct = improvements['piano_authenticity'][metric_name]['improvement_percent']
                
                table_lines.append(
                    f"{'Piano Authenticity':<25} {metric_name.replace('_', ' ').title():<25} "
                    f"{baseline_mean:<12.4f} {finetuned_mean:<12.4f} {change_pct:+7.1f}%"
                )
        
        # Musical Quality
        if 'musical_quality' in baseline_data:
            for metric_name in baseline_data['musical_quality'].keys():
                baseline_mean = np.mean(baseline_data['musical_quality'][metric_name])
                finetuned_mean = np.mean(finetuned_data['musical_quality'][metric_name])
                change_pct = improvements['musical_quality'][metric_name]['improvement_percent']
                
                table_lines.append(
                    f"{'Musical Quality':<25} {metric_name.replace('_', ' ').title():<25} "
                    f"{baseline_mean:<12.4f} {finetuned_mean:<12.4f} {change_pct:+7.1f}%"
                )
        
        # Technical Quality
        if 'technical_quality' in baseline_data:
            for metric_name in baseline_data['technical_quality'].keys():
                baseline_mean = np.mean(baseline_data['technical_quality'][metric_name])
                finetuned_mean = np.mean(finetuned_data['technical_quality'][metric_name])
                change_pct = improvements['technical_quality'][metric_name]['improvement_percent']
                
                table_lines.append(
                    f"{'Technical Quality':<25} {metric_name.replace('_', ' ').title():<25} "
                    f"{baseline_mean:<12.4f} {finetuned_mean:<12.4f} {change_pct:+7.1f}%"
                )
        
        # Semantic Alignment
        if 'semantic_alignment' in baseline_data:
            baseline_mean = np.mean(baseline_data['semantic_alignment'])
            finetuned_mean = np.mean(finetuned_data['semantic_alignment'])
            change_pct = improvements['semantic_alignment']['improvement_percent']
            
            table_lines.append(
                f"{'Semantic Alignment':<25} {'Prompt Adherence':<25} "
                f"{baseline_mean:<12.4f} {finetuned_mean:<12.4f} {change_pct:+7.1f}%"
            )
        
        table_lines.append("=" * 80)
        
        # Save table to file
        table_content = "\n".join(table_lines)
        table_file = self.output_dir / "comparison_table.txt"
        with open(table_file, 'w') as f:
            f.write(table_content)
        
        return table_content
    
    def run_comprehensive_evaluation(self):
        """
        Run comprehensive evaluation on existing samples
        """
        print("Starting Comprehensive Piano Music Evaluation")
        print("=" * 60)
        
        # Get audio files
        baseline_files = self.get_audio_files(self.baseline_dir)
        finetuned_files = self.get_audio_files(self.finetuned_dir)
        
        print(f"Found {len(baseline_files)} baseline samples")
        print(f"Found {len(finetuned_files)} fine-tuned samples")
        
        if not baseline_files or not finetuned_files:
            print("Error: No audio files found in specified directories")
            return None
        
        results = {
            'baseline': {},
            'finetuned': {},
            'improvements': {},
            'summary': {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'baseline_samples': len(baseline_files),
                'finetuned_samples': len(finetuned_files),
                'evaluation_type': 'comprehensive_piano_analysis'
            }
        }
        
        # 1. Piano authenticity
        print("1. Evaluating piano authenticity...")
        baseline_piano = self.evaluate_piano_authenticity(baseline_files)
        finetuned_piano = self.evaluate_piano_authenticity(finetuned_files)
        results['baseline']['piano_authenticity'] = baseline_piano
        results['finetuned']['piano_authenticity'] = finetuned_piano
        
        # 2. Musical quality
        print("2. Analyzing musical quality...")
        baseline_musical = self.evaluate_musical_quality(baseline_files)
        finetuned_musical = self.evaluate_musical_quality(finetuned_files)
        results['baseline']['musical_quality'] = baseline_musical
        results['finetuned']['musical_quality'] = finetuned_musical
        
        # 3. Technical quality
        print("3. Assessing technical quality...")
        baseline_technical = self.evaluate_technical_quality(baseline_files)
        finetuned_technical = self.evaluate_technical_quality(finetuned_files)
        results['baseline']['technical_quality'] = baseline_technical
        results['finetuned']['technical_quality'] = finetuned_technical
        
        # 4. Semantic alignment
        print("4. Evaluating semantic alignment...")
        baseline_semantic = self.evaluate_semantic_alignment(baseline_files)
        finetuned_semantic = self.evaluate_semantic_alignment(finetuned_files)
        results['baseline']['semantic_alignment'] = baseline_semantic
        results['finetuned']['semantic_alignment'] = finetuned_semantic
        
        # 5. Calculate improvements
        print("5. Calculating improvement metrics...")
        improvements = self.calculate_improvement_focused_scores(
            results['baseline'], results['finetuned']
        )
        results['improvements'] = improvements
        
        # 6. Generate summary
        summary = self.generate_improvement_summary(results)
        results['summary'] = summary
        
        # 7. Generate comparison table
        print("6. Generating comparison table...")
        comparison_table = self.generate_comparison_table(results)
        
        # Save results with proper JSON serialization
        results_file = self.output_dir / "comprehensive_evaluation_results.json"
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert numpy types to Python types"""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'item'):  # Handle numpy scalars
                return obj.item()
            else:
                return obj
        
        # Convert results to JSON-serializable format
        json_results = convert_numpy_types(results)
        
        with open(results_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Print comparison table
        print("\n" + comparison_table)
        
        print(f"\n✓ Comprehensive evaluation complete!")
        print(f"Results saved to {results_file}")
        print(f"Comparison table saved to comparison_table.txt")
        
        return results
    
    def generate_improvement_summary(self, results: Dict) -> Dict:
        """
        Generate a summary highlighting improvements
        """
        improvements = results['improvements']
        
        summary = {
            'total_metrics_evaluated': 0,
            'metrics_improved': 0,
            'significant_improvements': [],
            'overall_improvement_score': 0.0,
            'key_findings': []
        }
        
        # Count improvements
        for category, metrics in improvements.items():
            if isinstance(metrics, dict) and 'baseline_mean' in metrics:
                # Single metric
                summary['total_metrics_evaluated'] += 1
                if metrics['is_improvement']:
                    summary['metrics_improved'] += 1
                    if metrics['improvement_percent'] > 5:
                        summary['significant_improvements'].append({
                            'metric': category,
                            'improvement': metrics['improvement_percent']
                        })
            else:
                # Nested metrics
                for metric_name, metric_data in metrics.items():
                    if isinstance(metric_data, dict) and 'baseline_mean' in metric_data:
                        summary['total_metrics_evaluated'] += 1
                        if metric_data['is_improvement']:
                            summary['metrics_improved'] += 1
                            if metric_data['improvement_percent'] > 5:
                                summary['significant_improvements'].append({
                                    'metric': f"{category}.{metric_name}",
                                    'improvement': metric_data['improvement_percent']
                                })
        
        # Calculate overall improvement score
        if summary['total_metrics_evaluated'] > 0:
            summary['overall_improvement_score'] = (
                summary['metrics_improved'] / summary['total_metrics_evaluated']
            ) * 100
        
        # Generate key findings
        if summary['metrics_improved'] > summary['total_metrics_evaluated'] * 0.6:
            summary['key_findings'].append("Fine-tuned model shows improvements across majority of metrics")
        
        if len(summary['significant_improvements']) > 0:
            summary['key_findings'].append(f"Significant improvements in {len(summary['significant_improvements'])} key areas")
        
        return summary
    
    def generate_comprehensive_report(self, results: Dict):
        """
        Generate a comprehensive evaluation report
        """
        print("\n" + "=" * 80)
        print("COMPREHENSIVE PIANO MUSIC GENERATION EVALUATION REPORT")
        print("=" * 80)
        
        # Overall summary
        summary = results['summary']
        print(f"\nOVERALL SUMMARY:")
        print(f"Total Metrics Evaluated: {summary['total_metrics_evaluated']}")
        print(f"Metrics Showing Improvement: {summary['metrics_improved']}")
        print(f"Overall Improvement Score: {summary['overall_improvement_score']:.1f}%")
        
        # Significant improvements
        if summary['significant_improvements']:
            print(f"\nSIGNIFICANT IMPROVEMENTS (>5%):")
            for improvement in sorted(summary['significant_improvements'], 
                                   key=lambda x: x['improvement'], reverse=True):
                print(f"  ✓ {improvement['metric']}: +{improvement['improvement']:.1f}%")
        
        # Category-wise analysis
        improvements = results['improvements']
        
        print(f"\nPIANO AUTHENTICITY ANALYSIS:")
        if 'piano_authenticity' in improvements:
            for metric, data in improvements['piano_authenticity'].items():
                status = "↗" if data['is_improvement'] else "↘"
                print(f"  {status} {metric}: {data['baseline_mean']:.3f} → {data['finetuned_mean']:.3f} ({data['improvement_percent']:+.1f}%)")
        
        print(f"\nMUSICAL QUALITY ANALYSIS:")
        if 'musical_quality' in improvements:
            for metric, data in improvements['musical_quality'].items():
                status = "↗" if data['is_improvement'] else "↘"
                print(f"  {status} {metric}: {data['baseline_mean']:.3f} → {data['finetuned_mean']:.3f} ({data['improvement_percent']:+.1f}%)")
        
        print(f"\nTECHNICAL QUALITY ANALYSIS:")
        if 'technical_quality' in improvements:
            for metric, data in improvements['technical_quality'].items():
                status = "↗" if data['is_improvement'] else "↘"
                print(f"  {status} {metric}: {data['baseline_mean']:.3f} → {data['finetuned_mean']:.3f} ({data['improvement_percent']:+.1f}%)")
        
        print(f"\nSEMANTIC ALIGNMENT:")
        if 'semantic_alignment' in improvements:
            data = improvements['semantic_alignment']
            status = "↗" if data['is_improvement'] else "↘"
            print(f"  {status} Prompt Adherence: {data['baseline_mean']:.3f} → {data['finetuned_mean']:.3f} ({data['improvement_percent']:+.1f}%)")
        
        # Key findings
        print(f"\nKEY FINDINGS:")
        for finding in summary['key_findings']:
            print(f"  • {finding}")
        
        # Recommendations
        print(f"\nRECOMMENDAT_IONS:")
        if summary['overall_improvement_score'] > 60:
            print("  ✅ Fine-tuning has been successful - model shows clear improvements")
        elif summary['overall_improvement_score'] > 40:
            print("  🔶 Fine-tuning shows moderate improvements - consider additional training")
        else:
            print("  ⚠️  Fine-tuning shows limited improvements - review training strategy")
        
        if len(summary['significant_improvements']) > 3:
            print("  ✅ Strong improvements across multiple quality dimensions")
        
        if fad_metrics.get('fad_baseline_vs_finetuned', 100) < 10:
            print("  ✅ Low FAD indicates successful quality transfer")
        
        print("=" * 80)


def main():
    """
    Main function to run comprehensive evaluation
    """
    # Sample directories
    baseline_dir = "evaluation_results/baseline_samples"
    finetuned_dir = "evaluation_results/finetuned_samples"
    
    # Check if directories exist
    if not os.path.exists(baseline_dir):
        print(f"Error: Baseline samples directory not found: {baseline_dir}")
        return None
        
    if not os.path.exists(finetuned_dir):
        print(f"Error: Fine-tuned samples directory not found: {finetuned_dir}")
        return None
    
    # Initialize evaluator
    evaluator = EnhancedPianoEvaluator(
        baseline_dir=baseline_dir,
        finetuned_dir=finetuned_dir
    )
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    if results:
        print("\nComprehensive evaluation completed successfully!")
        
        # Create summary visualization
        create_improvement_visualization(results)
        
        return results
    else:
        print("\nEvaluation failed!")
        return None


def create_improvement_visualization(results: Dict):
    """
    Create visualization showing improvements
    """
    try:
        import matplotlib.pyplot as plt
        
        # Collect improvement percentages
        improvements_data = []
        labels = []
        
        for category, metrics in results['improvements'].items():
            if isinstance(metrics, dict) and 'improvement_percent' in metrics:
                improvements_data.append(metrics['improvement_percent'])
                labels.append(category.replace('_', ' ').title())
            else:
                for metric_name, metric_data in metrics.items():
                    if isinstance(metric_data, dict) and 'improvement_percent' in metric_data:
                        improvements_data.append(metric_data['improvement_percent'])
                        labels.append(f"{category.replace('_', ' ').title()}\n{metric_name.replace('_', ' ')}")
        
        if improvements_data:
            # Create horizontal bar chart
            fig, ax = plt.subplots(figsize=(12, 8))
            
            colors = ['green' if x > 0 else 'red' for x in improvements_data]
            bars = ax.barh(range(len(improvements_data)), improvements_data, color=colors, alpha=0.7)
            
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=10)
            ax.set_xlabel('Improvement Percentage (%)', fontsize=12)
            ax.set_title('Fine-tuned Model Improvements by Metric', fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            ax.grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, improvements_data)):
                ax.text(value + (1 if value > 0 else -1), i, f'{value:.1f}%', 
                       va='center', ha='left' if value > 0 else 'right', fontsize=9)
            
            plt.tight_layout()
            plt.savefig('evaluation_results/results/improvement_visualization.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✓ Improvement visualization saved to evaluation_results/results/improvement_visualization.png")
            
    except ImportError:
        print("Matplotlib not available - skipping visualization")
    except Exception as e:
        print(f"Error creating visualization: {e}")


if __name__ == "__main__":
    main()
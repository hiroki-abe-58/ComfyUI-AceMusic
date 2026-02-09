"""
ACE-Step API Handler for ComfyUI
Wraps the ACE-Step inference API for use in ComfyUI nodes.
"""

import os
import sys
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
import tempfile
import soundfile as sf
import numpy as np


def load_audio_sf(path):
    """Load audio using soundfile instead of torchaudio to avoid torchcodec dependency"""
    data, sample_rate = sf.read(path)
    # soundfile returns (samples, channels), convert to (channels, samples)
    if data.ndim == 1:
        data = data[np.newaxis, :]  # mono: (samples,) -> (1, samples)
    else:
        data = data.T  # stereo: (samples, channels) -> (channels, samples)
    waveform = torch.from_numpy(data).float()
    return waveform, sample_rate


# Try to import ACE-Step modules
ACESTEP_AVAILABLE = False
ACEStepPipeline = None

try:
    from acestep.pipeline_ace_step import ACEStepPipeline as _ACEStepPipeline
    ACEStepPipeline = _ACEStepPipeline
    ACESTEP_AVAILABLE = True
except ImportError:
    pass


@dataclass
class AceMusicModel:
    """Container for ACE-Step pipeline"""
    pipeline: Any = None
    device: str = "cuda"
    checkpoint_dir: str = ""
    initialized: bool = False
    cpu_offload: bool = False
    torch_compile: bool = False
    overlapped_decode: bool = False


class AceMusicHandler:
    """
    Handler class that wraps ACE-Step API for ComfyUI integration.
    """
    
    _instance = None
    _model_cache: Dict[str, AceMusicModel] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.checkpoint_dir = self._get_checkpoint_dir()
    
    def _get_checkpoint_dir(self) -> str:
        """Get the checkpoint directory for ACE-Step models"""
        try:
            import folder_paths
            # Try to find acemusic folder in ComfyUI models
            models_dir = folder_paths.models_dir
            acemusic_dir = os.path.join(models_dir, "acemusic")
            
            if os.path.exists(acemusic_dir):
                return acemusic_dir
        except ImportError:
            pass
        
        # Fallback to default ACE-Step checkpoints location
        default_cache = os.path.join(os.path.expanduser("~"), ".cache/ace-step/checkpoints")
        return default_cache
    
    def get_available_dit_models(self) -> List[str]:
        """Get list of available DiT models (ACE-Step uses a unified model)"""
        return ["ACE-Step-v1-3.5B"]
    
    def get_available_lm_models(self) -> List[str]:
        """Get list of available LM models (not applicable in current API)"""
        return ["none"]
    
    def load_model(
        self,
        dit_model: str = "ACE-Step-v1-3.5B",
        lm_model: str = "none",
        device: str = "cuda",
        offload_to_cpu: bool = False,
        torch_compile: bool = False,
        overlapped_decode: bool = False,
    ) -> AceMusicModel:
        """
        Load ACE-Step pipeline.
        
        Args:
            dit_model: Name of the model (currently only ACE-Step-v1-3.5B)
            lm_model: Not used in current API
            device: Device to use (cuda or cpu)
            offload_to_cpu: Whether to offload to CPU when not in use
            torch_compile: Whether to use torch.compile for optimization
            overlapped_decode: Use overlapped decoding for faster inference
            
        Returns:
            AceMusicModel containing the loaded pipeline
        """
        if not ACESTEP_AVAILABLE:
            raise ImportError(
                "ACE-Step is not installed. Please install it first:\n"
                "pip install git+https://github.com/ace-step/ACE-Step.git\n"
                "or clone from: https://github.com/ace-step/ACE-Step"
            )
        
        # Create cache key
        cache_key = f"{dit_model}_{device}_{offload_to_cpu}_{torch_compile}_{overlapped_decode}"
        
        # Check if already loaded
        if cache_key in self._model_cache:
            cached = self._model_cache[cache_key]
            if cached.initialized:
                return cached
        
        # Determine device_id from device string
        device_id = 0
        if device.startswith("cuda:"):
            device_id = int(device.split(":")[1])
        
        # Determine dtype
        dtype = "bfloat16" if device.startswith("cuda") else "float32"
        
        # Initialize pipeline
        pipeline = ACEStepPipeline(
            checkpoint_dir=self.checkpoint_dir if os.path.exists(self.checkpoint_dir) else None,
            device_id=device_id,
            dtype=dtype,
            torch_compile=torch_compile,
            cpu_offload=offload_to_cpu,
            overlapped_decode=overlapped_decode,
        )
        
        # Load checkpoint (will auto-download if needed)
        pipeline.load_checkpoint()
        
        # Create model container
        model = AceMusicModel(
            pipeline=pipeline,
            device=device,
            checkpoint_dir=self.checkpoint_dir,
            initialized=True,
            cpu_offload=offload_to_cpu,
            torch_compile=torch_compile,
            overlapped_decode=overlapped_decode,
        )
        
        # Cache the model
        self._model_cache[cache_key] = model
        
        return model
    
    def generate_music(
        self,
        model: AceMusicModel,
        caption: str = "",
        lyrics: str = "",
        duration: float = 30.0,
        bpm: Optional[int] = None,
        keyscale: str = "",
        timesignature: str = "",
        vocal_language: str = "unknown",
        instrumental: bool = False,
        inference_steps: int = 27,
        guidance_scale: float = 15.0,
        seed: int = -1,
        thinking: bool = True,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate music from text description.
        
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        if not ACESTEP_AVAILABLE:
            raise ImportError("ACE-Step is not installed")
        
        # Build prompt with metadata
        prompt_parts = [caption]
        if bpm and bpm > 0:
            prompt_parts.append(f"BPM: {bpm}")
        if keyscale:
            prompt_parts.append(f"Key: {keyscale}")
        if timesignature:
            prompt_parts.append(f"Time Signature: {timesignature}")
        if vocal_language and vocal_language != "unknown":
            prompt_parts.append(f"Language: {vocal_language}")
        if instrumental:
            prompt_parts.append("Instrumental")
        
        prompt = ", ".join(prompt_parts)
        
        # Handle seed
        manual_seeds = None if seed < 0 else [seed]
        
        # Generate music
        with tempfile.TemporaryDirectory() as temp_dir:
            results = model.pipeline(
                audio_duration=duration,
                prompt=prompt,
                lyrics=lyrics if lyrics else "",
                infer_step=inference_steps,
                guidance_scale=guidance_scale,
                scheduler_type="euler",
                cfg_type="apg",
                omega_scale=10.0,
                manual_seeds=manual_seeds,
                guidance_interval=0.5,
                guidance_interval_decay=0.0,
                min_guidance_scale=3.0,
                use_erg_tag=True,
                use_erg_lyric=True,
                use_erg_diffusion=True,
                oss_steps="",
                guidance_scale_text=0.0,
                guidance_scale_lyric=0.0,
                save_path=temp_dir,
                batch_size=batch_size,
                task="text2music",
            )
            
            # Get the audio file path (results contains paths + params dict)
            audio_paths = [r for r in results if isinstance(r, str) and r.endswith('.wav')]
            
            if not audio_paths:
                raise RuntimeError("No audio generated")
            
            # Load the generated audio
            waveform, sample_rate = load_audio_sf(audio_paths[0])
            
            # Ensure tensor has correct shape [B, C, S]
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)  # [C, S] -> [1, C, S]
            
            return waveform.cpu().float(), sample_rate
    
    def generate_cover(
        self,
        model: AceMusicModel,
        src_audio_path: str,
        caption: str = "",
        lyrics: str = "",
        cover_strength: float = 0.8,
        inference_steps: int = 27,
        seed: int = -1,
        thinking: bool = True,
    ) -> Tuple[torch.Tensor, int]:
        """
        Generate a cover version of existing audio using audio2audio.
        
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        if not ACESTEP_AVAILABLE:
            raise ImportError("ACE-Step is not installed")
        
        manual_seeds = None if seed < 0 else [seed]
        
        # Use audio2audio task with ref_audio_strength = 1 - cover_strength
        ref_audio_strength = 1.0 - cover_strength
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = model.pipeline(
                audio_duration=-1,  # Infer from input
                prompt=caption,
                lyrics=lyrics if lyrics else "",
                infer_step=inference_steps,
                guidance_scale=15.0,
                scheduler_type="euler",
                cfg_type="apg",
                omega_scale=10.0,
                manual_seeds=manual_seeds,
                guidance_interval=0.5,
                guidance_interval_decay=0.0,
                min_guidance_scale=3.0,
                use_erg_tag=True,
                use_erg_lyric=True,
                use_erg_diffusion=True,
                oss_steps="",
                guidance_scale_text=0.0,
                guidance_scale_lyric=0.0,
                save_path=temp_dir,
                batch_size=1,
                task="audio2audio",
                audio2audio_enable=True,
                ref_audio_strength=ref_audio_strength,
                ref_audio_input=src_audio_path,
            )
            
            audio_paths = [r for r in results if isinstance(r, str) and r.endswith('.wav')]
            
            if not audio_paths:
                raise RuntimeError("No audio generated")
            
            waveform, sample_rate = load_audio_sf(audio_paths[0])
            
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)
            
            return waveform.cpu().float(), sample_rate
    
    def repaint_audio(
        self,
        model: AceMusicModel,
        src_audio_path: str,
        start_time: float,
        end_time: float,
        caption: str = "",
        inference_steps: int = 27,
        seed: int = -1,
        retake_variance: float = 0.5,
        thinking: bool = True,
    ) -> Tuple[torch.Tensor, int]:
        """
        Repaint (regenerate) a specific section of audio.
        
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        if not ACESTEP_AVAILABLE:
            raise ImportError("ACE-Step is not installed")
        
        manual_seeds = None if seed < 0 else [seed]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = model.pipeline(
                audio_duration=-1,
                prompt=caption,
                lyrics="",
                infer_step=inference_steps,
                guidance_scale=15.0,
                scheduler_type="euler",
                cfg_type="apg",
                omega_scale=10.0,
                manual_seeds=manual_seeds,
                guidance_interval=0.5,
                guidance_interval_decay=0.0,
                min_guidance_scale=3.0,
                use_erg_tag=True,
                use_erg_lyric=True,
                use_erg_diffusion=True,
                oss_steps="",
                guidance_scale_text=0.0,
                guidance_scale_lyric=0.0,
                save_path=temp_dir,
                batch_size=1,
                task="repaint",
                retake_variance=retake_variance,
                repaint_start=start_time,
                repaint_end=end_time,
                src_audio_path=src_audio_path,
            )
            
            audio_paths = [r for r in results if isinstance(r, str) and r.endswith('.wav')]
            
            if not audio_paths:
                raise RuntimeError("No audio generated")
            
            waveform, sample_rate = load_audio_sf(audio_paths[0])
            
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)
            
            return waveform.cpu().float(), sample_rate
    
    def extend_audio(
        self,
        model: AceMusicModel,
        src_audio_path: str,
        extend_left: float = 0.0,
        extend_right: float = 0.0,
        caption: str = "",
        lyrics: str = "",
        inference_steps: int = 27,
        seed: int = -1,
        retake_variance: float = 0.5,
    ) -> Tuple[torch.Tensor, int]:
        """
        Extend audio at the beginning or end.
        
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        if not ACESTEP_AVAILABLE:
            raise ImportError("ACE-Step is not installed")
        
        manual_seeds = None if seed < 0 else [seed]
        
        # Load original audio to get duration
        original_waveform, original_sr = load_audio_sf(src_audio_path)
        original_duration = original_waveform.shape[-1] / original_sr
        
        # Calculate repaint region for extension
        repaint_start = -extend_left if extend_left > 0 else 0
        repaint_end = original_duration + extend_right if extend_right > 0 else original_duration
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = model.pipeline(
                audio_duration=-1,
                prompt=caption,
                lyrics=lyrics if lyrics else "",
                infer_step=inference_steps,
                guidance_scale=15.0,
                scheduler_type="euler",
                cfg_type="apg",
                omega_scale=10.0,
                manual_seeds=manual_seeds,
                guidance_interval=0.5,
                guidance_interval_decay=0.0,
                min_guidance_scale=3.0,
                use_erg_tag=True,
                use_erg_lyric=True,
                use_erg_diffusion=True,
                oss_steps="",
                guidance_scale_text=0.0,
                guidance_scale_lyric=0.0,
                save_path=temp_dir,
                batch_size=1,
                task="extend",
                retake_variance=retake_variance,
                repaint_start=repaint_start,
                repaint_end=repaint_end,
                src_audio_path=src_audio_path,
            )
            
            audio_paths = [r for r in results if isinstance(r, str) and r.endswith('.wav')]
            
            if not audio_paths:
                raise RuntimeError("No audio generated")
            
            waveform, sample_rate = load_audio_sf(audio_paths[0])
            
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)
            
            return waveform.cpu().float(), sample_rate
    
    def edit_audio(
        self,
        model: AceMusicModel,
        src_audio_path: str,
        original_caption: str,
        original_lyrics: str,
        target_caption: str,
        target_lyrics: str,
        inference_steps: int = 60,
        guidance_scale: float = 15.0,
        edit_n_min: float = 0.0,
        edit_n_max: float = 1.0,
        edit_n_avg: int = 1,
        seed: int = -1,
    ) -> Tuple[torch.Tensor, int]:
        """
        Edit audio using FlowEdit technique (change tags or lyrics while preserving structure).
        
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        if not ACESTEP_AVAILABLE:
            raise ImportError("ACE-Step is not installed")
        
        manual_seeds = None if seed < 0 else [seed]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = model.pipeline(
                audio_duration=-1,
                prompt=original_caption,
                lyrics=original_lyrics,
                infer_step=inference_steps,
                guidance_scale=guidance_scale,
                scheduler_type="euler",
                cfg_type="apg",
                omega_scale=10.0,
                manual_seeds=manual_seeds,
                guidance_interval=0.5,
                guidance_interval_decay=0.0,
                min_guidance_scale=3.0,
                use_erg_tag=True,
                use_erg_lyric=True,
                use_erg_diffusion=True,
                oss_steps="",
                guidance_scale_text=0.0,
                guidance_scale_lyric=0.0,
                save_path=temp_dir,
                batch_size=1,
                task="edit",
                src_audio_path=src_audio_path,
                edit_target_prompt=target_caption,
                edit_target_lyrics=target_lyrics,
                edit_n_min=edit_n_min,
                edit_n_max=edit_n_max,
                edit_n_avg=edit_n_avg,
            )
            
            audio_paths = [r for r in results if isinstance(r, str) and r.endswith('.wav')]
            
            if not audio_paths:
                raise RuntimeError("No audio generated")
            
            waveform, sample_rate = load_audio_sf(audio_paths[0])
            
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)
            
            return waveform.cpu().float(), sample_rate
    
    def retake_audio(
        self,
        model: AceMusicModel,
        src_audio_path: str,
        caption: str = "",
        lyrics: str = "",
        retake_variance: float = 0.5,
        inference_steps: int = 27,
        seed: int = -1,
    ) -> Tuple[torch.Tensor, int]:
        """
        Create a variation of existing audio.
        
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        if not ACESTEP_AVAILABLE:
            raise ImportError("ACE-Step is not installed")
        
        manual_seeds = None if seed < 0 else [seed]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            results = model.pipeline(
                audio_duration=-1,
                prompt=caption,
                lyrics=lyrics if lyrics else "",
                infer_step=inference_steps,
                guidance_scale=15.0,
                scheduler_type="euler",
                cfg_type="apg",
                omega_scale=10.0,
                manual_seeds=manual_seeds,
                guidance_interval=0.5,
                guidance_interval_decay=0.0,
                min_guidance_scale=3.0,
                use_erg_tag=True,
                use_erg_lyric=True,
                use_erg_diffusion=True,
                oss_steps="",
                guidance_scale_text=0.0,
                guidance_scale_lyric=0.0,
                save_path=temp_dir,
                batch_size=1,
                task="retake",
                retake_variance=retake_variance,
                src_audio_path=src_audio_path,
            )
            
            audio_paths = [r for r in results if isinstance(r, str) and r.endswith('.wav')]
            
            if not audio_paths:
                raise RuntimeError("No audio generated")
            
            waveform, sample_rate = load_audio_sf(audio_paths[0])
            
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(0)
            
            return waveform.cpu().float(), sample_rate

    def understand_audio(
        self,
        model: AceMusicModel,
        audio_path: str,
    ) -> Dict[str, Any]:
        """
        Analyze audio to extract metadata.
        Uses the LM model to understand the audio content.
        
        Returns:
            Dictionary with caption, lyrics, bpm, keyscale, duration, language
        """
        if not ACESTEP_AVAILABLE:
            raise ImportError("ACE-Step is not installed")
        
        # Load audio to get duration
        waveform, sample_rate = load_audio_sf(audio_path)
        duration = waveform.shape[-1] / sample_rate
        
        # Try to use understand_music API if available
        try:
            if hasattr(model.pipeline, 'understand_music'):
                result = model.pipeline.understand_music(audio_path)
                return {
                    "caption": result.get("caption", ""),
                    "lyrics": result.get("lyrics", ""),
                    "bpm": result.get("bpm", 120),
                    "keyscale": result.get("keyscale", ""),
                    "duration": duration,
                    "language": result.get("language", "unknown"),
                }
        except Exception:
            pass
        
        # Fallback: return basic info
        return {
            "caption": "Audio track",
            "lyrics": "",
            "bpm": 120,
            "keyscale": "C Major",
            "duration": duration,
            "language": "unknown",
        }

    def create_sample_from_query(
        self,
        model: AceMusicModel,
        query: str,
        instrumental: bool = False,
        vocal_language: Optional[str] = None,
        duration: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Generate complete music parameters from a natural language query.
        Uses the LM model to create caption, lyrics, and metadata.
        
        Returns:
            Dictionary with caption, lyrics, bpm, keyscale, duration, language, instrumental
        """
        if not ACESTEP_AVAILABLE:
            raise ImportError("ACE-Step is not installed")
        
        # Try to use create_sample API if available
        try:
            if hasattr(model.pipeline, 'create_sample'):
                result = model.pipeline.create_sample(
                    query=query,
                    instrumental=instrumental,
                    vocal_language=vocal_language,
                )
                return {
                    "caption": result.get("caption", query),
                    "lyrics": result.get("lyrics", "[Instrumental]" if instrumental else ""),
                    "bpm": result.get("bpm", 120),
                    "keyscale": result.get("keyscale", "C Major"),
                    "duration": result.get("duration", duration),
                    "language": result.get("language", vocal_language or "unknown"),
                    "instrumental": instrumental,
                }
        except Exception:
            pass
        
        # Fallback: generate based on query
        # Infer some basic metadata from the query
        bpm = 120  # Default BPM
        keyscale = "C Major"
        
        # Simple keyword-based inference
        query_lower = query.lower()
        if any(word in query_lower for word in ["fast", "energetic", "dance", "upbeat"]):
            bpm = 140
        elif any(word in query_lower for word in ["slow", "ballad", "calm", "peaceful"]):
            bpm = 80
        elif any(word in query_lower for word in ["rock", "punk"]):
            bpm = 130
        
        if any(word in query_lower for word in ["sad", "melancholic", "dark"]):
            keyscale = "A Minor"
        elif any(word in query_lower for word in ["happy", "cheerful", "bright"]):
            keyscale = "C Major"
        
        # Generate lyrics placeholder if not instrumental
        lyrics = "[Instrumental]" if instrumental else f"[Verse]\n{query}\n\n[Chorus]\n{query}"
        
        return {
            "caption": query,
            "lyrics": lyrics,
            "bpm": bpm,
            "keyscale": keyscale,
            "duration": duration,
            "language": vocal_language or "unknown",
            "instrumental": instrumental,
        }


# Global handler instance
_handler: Optional[AceMusicHandler] = None


def get_handler() -> AceMusicHandler:
    """Get the global AceMusicHandler instance"""
    global _handler
    if _handler is None:
        _handler = AceMusicHandler()
    return _handler

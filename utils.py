"""
Utility functions for ComfyUI-AceMusic
Handles audio type conversions, file I/O, and common operations.
"""

import os
import torch
import soundfile as sf
import tempfile
import numpy as np
from typing import Tuple, Dict, Any, Optional
from scipy import signal


def resample_waveform(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """
    Resample waveform using scipy.
    
    Args:
        waveform: Input tensor of shape (channels, samples)
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled tensor
    """
    if orig_sr == target_sr:
        return waveform
    
    # Calculate new number of samples
    num_samples = int(waveform.shape[-1] * target_sr / orig_sr)
    
    # Convert to numpy for scipy processing
    waveform_np = waveform.numpy()
    
    # Resample each channel
    if waveform_np.ndim == 1:
        resampled = signal.resample(waveform_np, num_samples)
    else:
        resampled = np.array([signal.resample(ch, num_samples) for ch in waveform_np])
    
    return torch.from_numpy(resampled).float()


def audio_to_tempfile(audio: Dict[str, Any], format: str = "wav") -> str:
    """
    Convert ComfyUI AUDIO type to a temporary file.
    
    Args:
        audio: ComfyUI AUDIO dict with "waveform" and "sample_rate"
        format: Output format ("wav" or "mp3")
        
    Returns:
        Path to temporary audio file
    """
    waveform = audio["waveform"]  # [B, C, S]
    sample_rate = audio["sample_rate"]
    
    # Use first batch item
    if waveform.dim() == 3:
        waveform = waveform[0]  # [C, S]
    
    # Ensure on CPU
    waveform = waveform.cpu()
    
    # Create temporary file
    suffix = f".{format}"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    
    # Save audio using soundfile
    # Convert from (channels, samples) to (samples, channels) for soundfile
    audio_np = waveform.numpy()
    if audio_np.ndim == 2:
        audio_np = audio_np.T
    sf.write(path, audio_np, sample_rate)
    
    return path


def result_to_audio(
    tensor: torch.Tensor,
    sample_rate: int = 48000,
) -> Dict[str, Any]:
    """
    Convert audio tensor to ComfyUI AUDIO type.
    
    Args:
        tensor: Audio tensor, can be [S], [C, S], or [B, C, S]
        sample_rate: Sample rate in Hz
        
    Returns:
        ComfyUI AUDIO dict
    """
    # Ensure correct dimensions [B, C, S]
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # [S] -> [1, 1, S]
    elif tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)  # [C, S] -> [1, C, S]
    
    # Ensure float and on CPU
    tensor = tensor.cpu().float()
    
    return {
        "waveform": tensor,
        "sample_rate": sample_rate,
    }


def load_audio_file(
    filepath: str,
    target_sample_rate: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Load audio file and convert to ComfyUI AUDIO type.
    
    Args:
        filepath: Path to audio file
        target_sample_rate: Resample to this rate if specified
        
    Returns:
        ComfyUI AUDIO dict
    """
    # Load using soundfile
    data, sample_rate = sf.read(filepath)
    # Convert from (samples, channels) to (channels, samples)
    if data.ndim == 1:
        waveform = torch.from_numpy(data[np.newaxis, :]).float()
    else:
        waveform = torch.from_numpy(data.T).float()
    
    # Resample if needed
    if target_sample_rate is not None and sample_rate != target_sample_rate:
        waveform = resample_waveform(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate
    
    return result_to_audio(waveform, sample_rate)


def normalize_audio(
    audio: Dict[str, Any],
    target_db: float = -3.0,
) -> Dict[str, Any]:
    """
    Normalize audio to target dB level.
    
    Args:
        audio: ComfyUI AUDIO dict
        target_db: Target level in dB (default -3dB)
        
    Returns:
        Normalized ComfyUI AUDIO dict
    """
    waveform = audio["waveform"].clone()
    
    # Calculate current peak
    peak = waveform.abs().max()
    
    if peak > 0:
        # Calculate target level
        target_linear = 10 ** (target_db / 20)
        # Normalize
        waveform = waveform * (target_linear / peak)
    
    return {
        "waveform": waveform,
        "sample_rate": audio["sample_rate"],
    }


def trim_audio(
    audio: Dict[str, Any],
    start_time: float = 0.0,
    end_time: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Trim audio to specified time range.
    
    Args:
        audio: ComfyUI AUDIO dict
        start_time: Start time in seconds
        end_time: End time in seconds (None for end of audio)
        
    Returns:
        Trimmed ComfyUI AUDIO dict
    """
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    
    start_sample = int(start_time * sample_rate)
    
    if end_time is not None:
        end_sample = int(end_time * sample_rate)
        waveform = waveform[:, :, start_sample:end_sample]
    else:
        waveform = waveform[:, :, start_sample:]
    
    return {
        "waveform": waveform,
        "sample_rate": sample_rate,
    }


def get_audio_duration(audio: Dict[str, Any]) -> float:
    """
    Get audio duration in seconds.
    
    Args:
        audio: ComfyUI AUDIO dict
        
    Returns:
        Duration in seconds
    """
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    
    # Get number of samples (last dimension)
    num_samples = waveform.shape[-1]
    
    return num_samples / sample_rate


def resample_audio(
    audio: Dict[str, Any],
    target_sample_rate: int,
) -> Dict[str, Any]:
    """
    Resample audio to target sample rate.
    
    Args:
        audio: ComfyUI AUDIO dict
        target_sample_rate: Target sample rate in Hz
        
    Returns:
        Resampled ComfyUI AUDIO dict
    """
    waveform = audio["waveform"]
    sample_rate = audio["sample_rate"]
    
    if sample_rate == target_sample_rate:
        return audio
    
    # Resample each batch item using scipy
    batch_size = waveform.shape[0]
    
    resampled = []
    for i in range(batch_size):
        resampled.append(resample_waveform(waveform[i], sample_rate, target_sample_rate))
    
    resampled_waveform = torch.stack(resampled, dim=0)
    
    return {
        "waveform": resampled_waveform,
        "sample_rate": target_sample_rate,
    }


def concat_audio(
    audio_list: list,
    crossfade_duration: float = 0.0,
) -> Dict[str, Any]:
    """
    Concatenate multiple audio clips.
    
    Args:
        audio_list: List of ComfyUI AUDIO dicts
        crossfade_duration: Duration of crossfade between clips in seconds
        
    Returns:
        Concatenated ComfyUI AUDIO dict
    """
    if not audio_list:
        raise ValueError("audio_list cannot be empty")
    
    if len(audio_list) == 1:
        return audio_list[0]
    
    # Use first audio's sample rate as reference
    target_sr = audio_list[0]["sample_rate"]
    
    # Resample all to same rate
    resampled = [resample_audio(a, target_sr) for a in audio_list]
    
    if crossfade_duration <= 0:
        # Simple concatenation
        waveforms = [a["waveform"] for a in resampled]
        combined = torch.cat(waveforms, dim=-1)
    else:
        # Concatenation with crossfade
        crossfade_samples = int(crossfade_duration * target_sr)
        
        result = resampled[0]["waveform"]
        
        for i in range(1, len(resampled)):
            next_wave = resampled[i]["waveform"]
            
            # Create crossfade
            fade_out = torch.linspace(1, 0, crossfade_samples)
            fade_in = torch.linspace(0, 1, crossfade_samples)
            
            # Apply fades
            result[..., -crossfade_samples:] *= fade_out
            next_wave[..., :crossfade_samples] *= fade_in
            
            # Overlap and add
            overlap = result[..., -crossfade_samples:] + next_wave[..., :crossfade_samples]
            
            # Combine: result (except overlap) + overlap + next (except overlap)
            result = torch.cat([
                result[..., :-crossfade_samples],
                overlap,
                next_wave[..., crossfade_samples:],
            ], dim=-1)
        
        combined = result
    
    return {
        "waveform": combined,
        "sample_rate": target_sr,
    }


def stereo_to_mono(audio: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert stereo audio to mono.
    
    Args:
        audio: ComfyUI AUDIO dict
        
    Returns:
        Mono ComfyUI AUDIO dict
    """
    waveform = audio["waveform"]
    
    if waveform.shape[1] == 1:
        return audio  # Already mono
    
    # Average channels
    mono = waveform.mean(dim=1, keepdim=True)
    
    return {
        "waveform": mono,
        "sample_rate": audio["sample_rate"],
    }


def mono_to_stereo(audio: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert mono audio to stereo by duplicating channel.
    
    Args:
        audio: ComfyUI AUDIO dict
        
    Returns:
        Stereo ComfyUI AUDIO dict
    """
    waveform = audio["waveform"]
    
    if waveform.shape[1] == 2:
        return audio  # Already stereo
    
    # Duplicate mono channel
    stereo = waveform.repeat(1, 2, 1)
    
    return {
        "waveform": stereo,
        "sample_rate": audio["sample_rate"],
    }


# Language codes supported by ACE-Step
SUPPORTED_LANGUAGES = [
    "unknown",
    "en",  # English
    "zh",  # Chinese
    "ja",  # Japanese
    "ko",  # Korean
    "es",  # Spanish
    "fr",  # French
    "de",  # German
    "it",  # Italian
    "pt",  # Portuguese
    "ru",  # Russian
]

# Musical key signatures
KEY_SIGNATURES = [
    "",  # No key specified
    "C Major",
    "C Minor",
    "C# Major",
    "C# Minor",
    "D Major",
    "D Minor",
    "D# Major",
    "D# Minor",
    "E Major",
    "E Minor",
    "F Major",
    "F Minor",
    "F# Major",
    "F# Minor",
    "G Major",
    "G Minor",
    "G# Major",
    "G# Minor",
    "A Major",
    "A Minor",
    "A# Major",
    "A# Minor",
    "B Major",
    "B Minor",
]

# Time signatures
TIME_SIGNATURES = [
    "",  # No time signature specified
    "4/4",
    "3/4",
    "2/4",
    "6/8",
    "5/4",
    "7/8",
    "12/8",
]

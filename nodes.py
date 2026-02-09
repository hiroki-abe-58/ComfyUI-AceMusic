"""
ComfyUI Nodes for AceMusic (ACE-Step)
Implements nodes for music generation, cover creation, repainting, and analysis.
"""

import torch
import os
import tempfile
from typing import Tuple, Dict, Any, Optional

from .acemusic_handler import get_handler, AceMusicModel, ACESTEP_AVAILABLE
from .utils import (
    audio_to_tempfile,
    result_to_audio,
    get_audio_duration,
    SUPPORTED_LANGUAGES,
    KEY_SIGNATURES,
    TIME_SIGNATURES,
)


class AceMusicModelLoader:
    """
    Load ACE-Step pipeline for music generation.
    Downloads models automatically from Hugging Face if not present.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "device": (["cuda", "cuda:0", "cuda:1", "cpu"], {"default": "cuda"}),
            },
            "optional": {
                "cpu_offload": ("BOOLEAN", {"default": False}),
                "torch_compile": ("BOOLEAN", {"default": False}),
                "overlapped_decode": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("ACEMUSIC_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "AceMusic/Loaders"
    
    def load_model(
        self,
        device: str,
        cpu_offload: bool = False,
        torch_compile: bool = False,
        overlapped_decode: bool = False,
    ):
        handler = get_handler()
        model = handler.load_model(
            device=device,
            offload_to_cpu=cpu_offload,
            torch_compile=torch_compile,
            overlapped_decode=overlapped_decode,
        )
        return (model,)


class AceMusicSettings:
    """
    Settings node for AceMusicGenerator.
    Outputs a settings dictionary to connect to AceMusicGenerator.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration": ("FLOAT", {
                    "default": 30.0,
                    "min": 5.0,
                    "max": 240.0,
                    "step": 1.0,
                }),
                "vocal_language": (SUPPORTED_LANGUAGES, {"default": "ja"}),
                "bpm": ("INT", {
                    "default": 120,
                    "min": 0,
                    "max": 300,
                    "step": 1,
                }),
                "timesignature": (TIME_SIGNATURES, {"default": "4/4"}),
                "keyscale": (KEY_SIGNATURES, {"default": ""}),
                "instrumental": ("BOOLEAN", {"default": False}),
                "inference_steps": ("INT", {
                    "default": 27,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 15.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.5,
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0x7FFFFFFF,
                }),
            },
        }
    
    RETURN_TYPES = ("ACEMUSIC_SETTINGS",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "create_settings"
    CATEGORY = "AceMusic/Settings"
    
    def create_settings(
        self,
        duration: float,
        vocal_language: str,
        bpm: int,
        timesignature: str,
        keyscale: str,
        instrumental: bool,
        inference_steps: int,
        guidance_scale: float,
        seed: int,
    ):
        return ({
            "duration": duration,
            "vocal_language": vocal_language,
            "bpm": bpm,
            "timesignature": timesignature,
            "keyscale": keyscale,
            "instrumental": instrumental,
            "inference_steps": inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
        },)


class AceMusicGenerator:
    """
    Generate music from text description (Text2Music).
    Connect AceMusicLyricsInput, AceMusicCaptionInput, and AceMusicSettings nodes.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ACEMUSIC_MODEL",),
                "caption": ("STRING", {"forceInput": True}),
                "lyrics": ("STRING", {"forceInput": True}),
                "settings": ("ACEMUSIC_SETTINGS",),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "AceMusic/Generation"
    
    def generate(
        self,
        model: AceMusicModel,
        caption: str,
        lyrics: str,
        settings: dict,
    ):
        handler = get_handler()
        
        # Extract settings with defaults
        duration = settings.get("duration", 30.0)
        inference_steps = settings.get("inference_steps", 27)
        seed = settings.get("seed", -1)
        bpm = settings.get("bpm", 120)
        keyscale = settings.get("keyscale", "")
        timesignature = settings.get("timesignature", "4/4")
        vocal_language = settings.get("vocal_language", "ja")
        instrumental = settings.get("instrumental", False)
        guidance_scale = settings.get("guidance_scale", 15.0)
        
        # Validate
        if duration < 5.0 or duration > 240.0:
            duration = 30.0
        if inference_steps < 1 or inference_steps > 100:
            inference_steps = 27
        
        actual_seed = seed if seed >= 0 else torch.randint(0, 0x7FFFFFFF, (1,)).item()
        actual_bpm = bpm if bpm > 0 else None
        
        caption = str(caption) if caption else "Pop song"
        lyrics = str(lyrics) if lyrics else ""
        
        print(f"[AceMusicGenerator] Running:")
        print(f"  caption={caption[:50]}..., lyrics_len={len(lyrics)}, duration={duration}")
        print(f"  steps={inference_steps}, seed={actual_seed}, bpm={actual_bpm}")
        print(f"  lang={vocal_language}, instrumental={instrumental}")
        
        tensor, sample_rate = handler.generate_music(
            model=model,
            caption=caption,
            lyrics=lyrics,
            duration=duration,
            bpm=actual_bpm,
            keyscale=keyscale,
            timesignature=timesignature,
            vocal_language=vocal_language,
            instrumental=instrumental,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            seed=actual_seed,
        )
        
        audio = result_to_audio(tensor, sample_rate)
        return (audio,)


class AceMusicCover:
    """
    Create a cover version of existing audio using Audio2Audio.
    Transforms the style while preserving structure.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ACEMUSIC_MODEL",),
                "audio": ("AUDIO",),
                "caption": ("STRING", {
                    "multiline": True,
                    "default": "Jazz cover with piano and saxophone",
                }),
                "cover_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
            },
            "optional": {
                "lyrics": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "inference_steps": ("INT", {
                    "default": 27,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0x7FFFFFFF,
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "create_cover"
    CATEGORY = "AceMusic/Generation"
    
    def create_cover(
        self,
        model: AceMusicModel,
        audio: Dict[str, Any],
        caption: str,
        cover_strength: float,
        lyrics: str = "",
        inference_steps: int = 27,
        seed: int = -1,
    ):
        handler = get_handler()
        
        # Save audio to temp file
        temp_path = audio_to_tempfile(audio)
        
        try:
            actual_seed = seed if seed >= 0 else torch.randint(0, 0x7FFFFFFF, (1,)).item()
            
            tensor, sample_rate = handler.generate_cover(
                model=model,
                src_audio_path=temp_path,
                caption=caption,
                lyrics=lyrics,
                cover_strength=cover_strength,
                inference_steps=inference_steps,
                seed=actual_seed,
            )
            
            result = result_to_audio(tensor, sample_rate)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return (result,)


class AceMusicRepaint:
    """
    Repaint (regenerate) a specific section of audio.
    Replaces a time range with newly generated content.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ACEMUSIC_MODEL",),
                "audio": ("AUDIO",),
                "start_time": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 240.0,
                    "step": 0.1,
                }),
                "end_time": ("FLOAT", {
                    "default": 10.0,
                    "min": 0.1,
                    "max": 240.0,
                    "step": 0.1,
                }),
            },
            "optional": {
                "caption": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "retake_variance": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "inference_steps": ("INT", {
                    "default": 27,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0x7FFFFFFF,
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "repaint"
    CATEGORY = "AceMusic/Generation"
    
    def repaint(
        self,
        model: AceMusicModel,
        audio: Dict[str, Any],
        start_time: float,
        end_time: float,
        caption: str = "",
        retake_variance: float = 0.5,
        inference_steps: int = 27,
        seed: int = -1,
    ):
        handler = get_handler()
        
        # Validate time range
        audio_duration = get_audio_duration(audio)
        if end_time > audio_duration:
            end_time = audio_duration
        if start_time >= end_time:
            raise ValueError(f"start_time ({start_time}) must be less than end_time ({end_time})")
        
        # Save audio to temp file
        temp_path = audio_to_tempfile(audio)
        
        try:
            actual_seed = seed if seed >= 0 else torch.randint(0, 0x7FFFFFFF, (1,)).item()
            
            tensor, sample_rate = handler.repaint_audio(
                model=model,
                src_audio_path=temp_path,
                start_time=start_time,
                end_time=end_time,
                caption=caption,
                inference_steps=inference_steps,
                seed=actual_seed,
                retake_variance=retake_variance,
            )
            
            result = result_to_audio(tensor, sample_rate)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return (result,)


class AceMusicRetake:
    """
    Create a variation of existing audio.
    Generates a new version while maintaining the overall structure.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ACEMUSIC_MODEL",),
                "audio": ("AUDIO",),
                "retake_variance": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
            },
            "optional": {
                "caption": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "lyrics": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "inference_steps": ("INT", {
                    "default": 27,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0x7FFFFFFF,
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "retake"
    CATEGORY = "AceMusic/Generation"
    
    def retake(
        self,
        model: AceMusicModel,
        audio: Dict[str, Any],
        retake_variance: float,
        caption: str = "",
        lyrics: str = "",
        inference_steps: int = 27,
        seed: int = -1,
    ):
        handler = get_handler()
        
        # Save audio to temp file
        temp_path = audio_to_tempfile(audio)
        
        try:
            actual_seed = seed if seed >= 0 else torch.randint(0, 0x7FFFFFFF, (1,)).item()
            
            tensor, sample_rate = handler.retake_audio(
                model=model,
                src_audio_path=temp_path,
                caption=caption,
                lyrics=lyrics,
                retake_variance=retake_variance,
                inference_steps=inference_steps,
                seed=actual_seed,
            )
            
            result = result_to_audio(tensor, sample_rate)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return (result,)


class AceMusicExtend:
    """
    Extend audio at the beginning or end.
    Adds new content that matches the style of the original.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ACEMUSIC_MODEL",),
                "audio": ("AUDIO",),
            },
            "optional": {
                "extend_left": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 120.0,
                    "step": 1.0,
                }),
                "extend_right": ("FLOAT", {
                    "default": 30.0,
                    "min": 0.0,
                    "max": 120.0,
                    "step": 1.0,
                }),
                "caption": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "lyrics": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "retake_variance": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "inference_steps": ("INT", {
                    "default": 27,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0x7FFFFFFF,
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "extend"
    CATEGORY = "AceMusic/Generation"
    
    def extend(
        self,
        model: AceMusicModel,
        audio: Dict[str, Any],
        extend_left: float = 0.0,
        extend_right: float = 30.0,
        caption: str = "",
        lyrics: str = "",
        retake_variance: float = 0.5,
        inference_steps: int = 27,
        seed: int = -1,
    ):
        handler = get_handler()
        
        if extend_left == 0.0 and extend_right == 0.0:
            raise ValueError("At least one of extend_left or extend_right must be greater than 0")
        
        # Save audio to temp file
        temp_path = audio_to_tempfile(audio)
        
        try:
            actual_seed = seed if seed >= 0 else torch.randint(0, 0x7FFFFFFF, (1,)).item()
            
            tensor, sample_rate = handler.extend_audio(
                model=model,
                src_audio_path=temp_path,
                extend_left=extend_left,
                extend_right=extend_right,
                caption=caption,
                lyrics=lyrics,
                inference_steps=inference_steps,
                seed=actual_seed,
                retake_variance=retake_variance,
            )
            
            result = result_to_audio(tensor, sample_rate)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return (result,)


class AceMusicEdit:
    """
    Edit audio using FlowEdit technique.
    Changes tags or lyrics while preserving melody and structure.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ACEMUSIC_MODEL",),
                "audio": ("AUDIO",),
                "original_caption": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "target_caption": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
            },
            "optional": {
                "original_lyrics": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "target_lyrics": ("STRING", {
                    "multiline": True,
                    "default": "",
                }),
                "edit_strength": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "inference_steps": ("INT", {
                    "default": 60,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 15.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.5,
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0x7FFFFFFF,
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "edit"
    CATEGORY = "AceMusic/Generation"
    
    def edit(
        self,
        model: AceMusicModel,
        audio: Dict[str, Any],
        original_caption: str,
        target_caption: str,
        original_lyrics: str = "",
        target_lyrics: str = "",
        edit_strength: float = 0.5,
        inference_steps: int = 60,
        guidance_scale: float = 15.0,
        seed: int = -1,
    ):
        handler = get_handler()
        
        # Save audio to temp file
        temp_path = audio_to_tempfile(audio)
        
        try:
            actual_seed = seed if seed >= 0 else torch.randint(0, 0x7FFFFFFF, (1,)).item()
            
            # edit_strength controls n_max (how much of the diffusion process to apply)
            edit_n_max = edit_strength
            
            tensor, sample_rate = handler.edit_audio(
                model=model,
                src_audio_path=temp_path,
                original_caption=original_caption,
                original_lyrics=original_lyrics,
                target_caption=target_caption,
                target_lyrics=target_lyrics,
                inference_steps=inference_steps,
                guidance_scale=guidance_scale,
                edit_n_min=0.0,
                edit_n_max=edit_n_max,
                edit_n_avg=1,
                seed=actual_seed,
            )
            
            result = result_to_audio(tensor, sample_rate)
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return (result,)


class AceMusicConditioning:
    """
    Combine music generation parameters into a conditioning object.
    Provides flexibility in connecting different parameter sources.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                }),
            },
            "optional": {
                "lyrics": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                }),
                "bpm": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 300,
                    "forceInput": True,
                }),
                "keyscale": ("STRING", {
                    "default": "",
                    "forceInput": True,
                }),
                "duration": ("FLOAT", {
                    "default": 30.0,
                    "min": 5.0,
                    "max": 240.0,
                    "forceInput": True,
                }),
                "vocal_language": ("STRING", {
                    "default": "unknown",
                    "forceInput": True,
                }),
            },
        }
    
    RETURN_TYPES = ("ACEMUSIC_COND",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "create_conditioning"
    CATEGORY = "AceMusic/Loaders"
    
    def create_conditioning(
        self,
        caption: str,
        lyrics: str = "",
        bpm: int = 0,
        keyscale: str = "",
        duration: float = 30.0,
        vocal_language: str = "unknown",
    ):
        conditioning = {
            "caption": caption,
            "lyrics": lyrics,
            "bpm": bpm if bpm > 0 else None,
            "keyscale": keyscale,
            "duration": duration,
            "vocal_language": vocal_language,
        }
        return (conditioning,)


class AceMusicGeneratorFromCond:
    """
    Generate music using a conditioning object.
    Alternative to AceMusicGenerator that accepts ACEMUSIC_COND input.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ACEMUSIC_MODEL",),
                "conditioning": ("ACEMUSIC_COND",),
                "inference_steps": ("INT", {
                    "default": 27,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0x7FFFFFFF,
                }),
            },
            "optional": {
                "instrumental": ("BOOLEAN", {"default": False}),
                "guidance_scale": ("FLOAT", {
                    "default": 15.0,
                    "min": 1.0,
                    "max": 30.0,
                    "step": 0.5,
                }),
            },
        }
    
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "AceMusic/Generation"
    
    def generate(
        self,
        model: AceMusicModel,
        conditioning: Dict[str, Any],
        inference_steps: int,
        seed: int,
        instrumental: bool = False,
        guidance_scale: float = 15.0,
    ):
        handler = get_handler()
        
        actual_seed = seed if seed >= 0 else torch.randint(0, 0x7FFFFFFF, (1,)).item()
        
        tensor, sample_rate = handler.generate_music(
            model=model,
            caption=conditioning.get("caption", ""),
            lyrics=conditioning.get("lyrics", ""),
            duration=conditioning.get("duration", 30.0),
            bpm=conditioning.get("bpm"),
            keyscale=conditioning.get("keyscale", ""),
            vocal_language=conditioning.get("vocal_language", "unknown"),
            instrumental=instrumental,
            inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            seed=actual_seed,
        )
        
        audio = result_to_audio(tensor, sample_rate)
        return (audio,)


class AceMusicLoadLora:
    """
    Load a LoRA adapter for the ACE-Step model.
    Allows for fine-tuned styles like RapMachine, Lyric2Vocal, etc.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ACEMUSIC_MODEL",),
                "lora_name_or_path": ("STRING", {
                    "default": "ACE-Step/ACE-Step-v1-chinese-rap-LoRA",
                }),
                "lora_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                }),
            },
        }
    
    RETURN_TYPES = ("ACEMUSIC_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora"
    CATEGORY = "AceMusic/Loaders"
    
    def load_lora(
        self,
        model: AceMusicModel,
        lora_name_or_path: str,
        lora_weight: float,
    ):
        if model.pipeline is not None:
            model.pipeline.load_lora(lora_name_or_path, lora_weight)
        return (model,)


class AceMusicUnderstand:
    """
    Analyze audio to extract metadata using LM model.
    Returns caption, lyrics, BPM, key signature, and duration.
    Requires LM model to be initialized.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ACEMUSIC_MODEL",),
                "audio": ("AUDIO",),
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("caption", "lyrics", "bpm", "keyscale", "duration", "language")
    FUNCTION = "analyze"
    CATEGORY = "AceMusic/Analysis"
    
    def analyze(
        self,
        model: AceMusicModel,
        audio: Dict[str, Any],
    ):
        handler = get_handler()
        
        # Save audio to temp file for analysis
        temp_path = audio_to_tempfile(audio)
        
        try:
            result = handler.understand_audio(
                model=model,
                audio_path=temp_path,
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        return (
            result.get("caption", ""),
            result.get("lyrics", ""),
            result.get("bpm", 120),
            result.get("keyscale", ""),
            result.get("duration", 0.0),
            result.get("language", "unknown"),
        )


class AceMusicCreateSample:
    """
    Generate complete music parameters from a natural language query.
    Uses LM model to create caption, lyrics, BPM, key signature, and suggested duration.
    Requires LM model to be initialized.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ACEMUSIC_MODEL",),
                "query": ("STRING", {
                    "multiline": True,
                    "default": "A cheerful pop song about summer",
                }),
            },
            "optional": {
                "instrumental": ("BOOLEAN", {"default": False}),
                "vocal_language": (SUPPORTED_LANGUAGES, {"default": "unknown"}),
                "duration": ("FLOAT", {
                    "default": 30.0,
                    "min": 5.0,
                    "max": 240.0,
                    "step": 1.0,
                }),
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING", "FLOAT")
    RETURN_NAMES = ("caption", "lyrics", "bpm", "keyscale", "duration")
    FUNCTION = "create_sample"
    CATEGORY = "AceMusic/Analysis"
    
    def create_sample(
        self,
        model: AceMusicModel,
        query: str,
        instrumental: bool = False,
        vocal_language: str = "unknown",
        duration: float = 30.0,
    ):
        handler = get_handler()
        
        result = handler.create_sample_from_query(
            model=model,
            query=query,
            instrumental=instrumental,
            vocal_language=vocal_language if vocal_language != "unknown" else None,
            duration=duration,
        )
        
        return (
            result.get("caption", query),
            result.get("lyrics", "[Instrumental]" if instrumental else ""),
            result.get("bpm", 120),
            result.get("keyscale", "C Major"),
            result.get("duration", duration),
        )


class AceMusicLyricsInput:
    """
    Simple text input node for lyrics.
    Provides a multiline text area for entering song lyrics.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lyrics": ("STRING", {
                    "multiline": True,
                    "default": "[Verse]\nYour lyrics here\n\n[Chorus]\nChorus lyrics here",
                }),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics",)
    FUNCTION = "output_lyrics"
    CATEGORY = "AceMusic/Input"
    
    def output_lyrics(self, lyrics: str):
        return (lyrics,)


class AceMusicCaptionInput:
    """
    Simple text input node for music caption/style description.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "caption": ("STRING", {
                    "multiline": True,
                    "default": "Pop song, female vocal, bright melody",
                }),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    FUNCTION = "output_caption"
    CATEGORY = "AceMusic/Input"
    
    def output_caption(self, caption: str):
        return (caption,)


# Node class mappings for registration
NODE_CLASS_MAPPINGS = {
    "AceMusicModelLoader": AceMusicModelLoader,
    "AceMusicSettings": AceMusicSettings,
    "AceMusicGenerator": AceMusicGenerator,
    "AceMusicCover": AceMusicCover,
    "AceMusicRepaint": AceMusicRepaint,
    "AceMusicRetake": AceMusicRetake,
    "AceMusicExtend": AceMusicExtend,
    "AceMusicEdit": AceMusicEdit,
    "AceMusicConditioning": AceMusicConditioning,
    "AceMusicGeneratorFromCond": AceMusicGeneratorFromCond,
    "AceMusicLoadLora": AceMusicLoadLora,
    "AceMusicUnderstand": AceMusicUnderstand,
    "AceMusicCreateSample": AceMusicCreateSample,
    "AceMusicLyricsInput": AceMusicLyricsInput,
    "AceMusicCaptionInput": AceMusicCaptionInput,
}

# Display names for the UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "AceMusicModelLoader": "AceMusic Model Loader",
    "AceMusicSettings": "AceMusic Settings",
    "AceMusicGenerator": "AceMusic Generator (Text2Music)",
    "AceMusicCover": "AceMusic Cover (Audio2Audio)",
    "AceMusicRepaint": "AceMusic Repaint",
    "AceMusicRetake": "AceMusic Retake (Variations)",
    "AceMusicExtend": "AceMusic Extend",
    "AceMusicEdit": "AceMusic Edit (FlowEdit)",
    "AceMusicConditioning": "AceMusic Conditioning",
    "AceMusicGeneratorFromCond": "AceMusic Generator (from Conditioning)",
    "AceMusicLoadLora": "AceMusic Load LoRA",
    "AceMusicUnderstand": "AceMusic Understand (Audio Analysis)",
    "AceMusicCreateSample": "AceMusic Create Sample (Query to Params)",
    "AceMusicLyricsInput": "AceMusic Lyrics Input",
    "AceMusicCaptionInput": "AceMusic Caption/Style Input",
}

"""
ComfyUI-AceMusic
Custom nodes for ACE-Step 1.5 music generation in ComfyUI.

Features:
- Text to Music generation
- Audio Cover creation
- Audio Repainting (partial regeneration)
- Audio Understanding (metadata extraction)
- Sample Creation from natural language

Requirements:
- ACE-Step 1.5 (https://github.com/ace-step/ACE-Step-1.5)
- torch >= 2.0.0
- torchaudio
- soundfile
"""

from .nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Version info
__version__ = "1.0.0"
__author__ = "hiroki-abe-58"
__description__ = "ACE-Step 1.5 music generation nodes for ComfyUI"
__url__ = "https://github.com/hiroki-abe-58/ComfyUI-AceMusic"

# Web extension (if needed in future)
WEB_DIRECTORY = None

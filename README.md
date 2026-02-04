# ComfyUI-AceMusic

Multilingual AI music generation nodes for ComfyUI powered by [ACE-Step 1.5](https://github.com/ace-step/ACE-Step). Generate full songs with lyrics in 19 languages including English, Chinese, Japanese, Korean, and more.

![Workflow Preview](assets/workflow_preview.png)

## Features

- **Multilingual Lyrics** - Generate music with vocals in 19 languages (English, Chinese, Japanese, Korean, Spanish, etc.)
- **Song Structure Control** - Use section markers like [Verse], [Chorus], [Bridge] to define song structure
- **Style Tags** - Control genre, vocal type, mood, tempo, and instruments
- **4-Minute Songs** - Generate up to 240 seconds of continuous audio
- **Audio Editing** - Cover, Repaint, Extend, Edit, and Retake capabilities
- **LoRA Support** - Load fine-tuned adapters for specialized styles
- **HeartMuLa Compatible** - Works seamlessly with HeartMuLa nodes

## Nodes

| Node | Description |
|------|-------------|
| **Model Loader** | Downloads and caches ACE-Step models |
| **Settings** | Configure generation parameters (duration, language, BPM, etc.) |
| **Generator** | Generate music from caption and lyrics (Text2Music) |
| **Lyrics Input** | Dedicated node for entering lyrics with section markers |
| **Caption Input** | Dedicated node for style/genre description |
| **Cover** | Transform existing audio into different styles (Audio2Audio) |
| **Repaint** | Regenerate specific sections of audio |
| **Retake** | Create variations of existing audio |
| **Extend** | Add new content to beginning or end of audio |
| **Edit** | Change tags/lyrics while preserving melody (FlowEdit) |
| **Conditioning** | Combine parameters into conditioning object |
| **Generator (from Cond)** | Generate from conditioning object |
| **Load LoRA** | Load fine-tuned LoRA adapters |
| **Understand** | Extract metadata from existing audio |
| **Create Sample** | Generate parameters from natural language query |

## Installation

### ComfyUI Manager (Recommended)

Search for "ComfyUI-AceMusic" and install.

### Manual

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hiroki-abe-58/ComfyUI-AceMusic.git
cd ComfyUI-AceMusic
pip install -r requirements.txt
```

### Install ACE-Step 1.5

```bash
pip install git+https://github.com/ace-step/ACE-Step.git
```

Models are automatically downloaded from Hugging Face on first use.

## Quick Start

1. Add **AceMusic Model Loader** node and select device (`cuda`)
2. Add **AceMusic Settings** node to configure parameters
3. Add **AceMusic Lyrics Input** node and enter lyrics:
   ```
   [Verse]
   Walking down the empty street
   Thinking about you and me
   
   [Chorus]
   We belong together
   Now and forever
   ```
4. Add **AceMusic Caption Input** with style tags: `pop, female vocal, energetic`
5. Connect all to **AceMusic Generator** -> **Preview Audio**

Load the example workflow from `workflow/AceMusic_Lyrics_v3.json`

## Section Markers

ACE-Step supports these section markers for song structure:

| Marker | Usage |
|--------|-------|
| [Intro] | Opening instrumental or vocal intro |
| [Verse] | Main verses |
| [Pre-Chorus] | Build-up before chorus |
| [Chorus] | Main hook/chorus |
| [Bridge] | Contrasting section |
| [Outro] | Ending section |
| [Instrumental] | Non-vocal sections |

## Style Tags

Combine tags in the caption to control output style:

- **Genre**: pop, rock, electronic, jazz, classical, hip-hop, r&b, country, folk, metal, indie, j-pop, k-pop
- **Vocal**: female vocal, male vocal, duet, choir, instrumental
- **Mood**: energetic, melancholic, uplifting, calm, aggressive, romantic, dreamy, dark
- **Tempo**: slow, medium, fast
- **Instruments**: piano, guitar, drums, synth, strings, bass, violin, saxophone

**Example**: `j-pop, female vocal, energetic, bright synthesizer, catchy melody`

## Models & Hardware

Models download automatically from Hugging Face to `~/.cache/ace-step/checkpoints/`

### Performance

| Device | RTF (27 steps) | Time for 1 min audio |
|--------|----------------|---------------------|
| RTX 5090 | ~50x | ~1.2s |
| RTX 4090 | 34.48x | 1.74s |
| A100 | 27.27x | 2.20s |
| RTX 3090 | 12.76x | 4.70s |
| M2 Max | 2.27x | 26.43s |

### VRAM Requirements

| Mode | VRAM | Notes |
|------|------|-------|
| Normal | 8GB+ | Full speed |
| CPU Offload | ~4GB | Slower but works on limited VRAM |

## Parameters

### Settings Node

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| duration | 30 | 5-240 | Audio length in seconds |
| vocal_language | ja | 19 languages | Language for vocals |
| bpm | 120 | 0-300 | Beats per minute (0 = auto) |
| timesignature | 4/4 | Various | Time signature |
| keyscale | (auto) | 24 keys | Musical key |
| instrumental | false | bool | Generate without vocals |
| inference_steps | 27 | 1-100 | Quality vs speed |
| guidance_scale | 15.0 | 1-30 | Prompt adherence |
| seed | -1 | int | Random seed (-1 = random) |

## Supported Languages

ACE-Step supports 19 languages. Top performers:

| Language | Code | Quality |
|----------|------|---------|
| English | en | Excellent |
| Chinese | zh | Excellent |
| Japanese | ja | Excellent |
| Korean | ko | Very Good |
| Spanish | es | Very Good |
| German | de | Good |
| French | fr | Good |
| Portuguese | pt | Good |
| Italian | it | Good |
| Russian | ru | Good |

## Integration with HeartMuLa

The AUDIO type is compatible with HeartMuLa outputs:

- Use HeartMuLa-generated audio as input to AceMusic Cover
- Use HeartMuLa-generated audio as input to AceMusic Repaint
- Chain HeartMuLa and AceMusic nodes together for advanced workflows

## Troubleshooting

### Models not loading / Download fails
- Check your internet connection
- Verify Hugging Face access
- Try manually downloading from https://huggingface.co/ACE-Step

### Out of VRAM
- Enable `cpu_offload` in Model Loader
- Reduce `duration`
- Close other GPU applications

### Slow generation
- Enable `torch_compile` (requires triton)
- Use lower `inference_steps` (10-15 for drafts)
- Use `overlapped_decode` for long audio (>48s)

### Audio quality issues
- Increase `inference_steps` (50-100 for best quality)
- Adjust `guidance_scale` (try 10-20)
- Provide more detailed captions
- Try different seeds

### Windows-specific issues
- For `torchaudio` errors, ensure `soundfile` is installed: `pip install soundfile`
- For torch.compile, install triton: `pip install triton-windows`

## Requirements

- Python >= 3.10
- PyTorch >= 2.0.0
- ComfyUI
- ACE-Step 1.5

## License

Apache 2.0

## Credits

- [ACE-Step](https://github.com/ace-step/ACE-Step) - Original music generation model by ACE Studio and StepFun
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Node-based UI framework
- [HeartMuLa](https://github.com/filliptm/ComfyUI_FL-HeartMuLa) - Inspiration for node design

## Links

- [ACE-Step GitHub](https://github.com/ace-step/ACE-Step)
- [ACE-Step Hugging Face](https://huggingface.co/ACE-Step)
- [ACE Studio](https://www.acestudio.ai/)

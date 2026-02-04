# ComfyUI-AceMusic

**[English](README.md)** | **[日本語](README_ja.md)** | **简体中文** | **[繁體中文](README_zh-TW.md)** | **[한국어](README_ko.md)** | **[Tiếng Việt](README_vi.md)**

基于 [ACE-Step 1.5](https://github.com/ace-step/ACE-Step) 的 ComfyUI 多语言 AI 音乐生成节点。支持19种语言生成带歌词的完整歌曲，包括中文、英语、日语、韩语等。

![工作流预览](assets/workflow_preview.png)

---

**如果这个项目对你有帮助，请给个 Star 支持一下！**

[![GitHub stars](https://img.shields.io/github/stars/hiroki-abe-58/ComfyUI-AceMusic?style=social)](https://github.com/hiroki-abe-58/ComfyUI-AceMusic)

---

## 亮点

- **全球首个 ComfyUI 完整 ACE-Step 集成** - 将 ACE-Step 1.5 的所有功能完整实现为 ComfyUI 节点（共15个节点）
- **模块化架构** - 分离 Settings/Lyrics/Caption 节点，消除组件顺序问题，提升工作流可读性
- **跨平台兼容** - 使用 soundfile/scipy 替代有问题的 torchaudio 后端，支持 Windows + Python 3.13+
- **HeartMuLa 互操作** - 与 HeartMuLa 节点无缝衔接，实现混合 AI 音乐工作流
- **生产环境就绪** - 强大的输入验证和自动回退机制，防止运行时错误

## 功能特性

- **多语言歌词** - 支持19种语言（中文、英语、日语、韩语、西班牙语等）生成带人声的音乐
- **歌曲结构控制** - 使用 [Verse]、[Chorus]、[Bridge] 等标记定义歌曲结构
- **风格标签** - 控制流派、人声类型、情绪、节奏和乐器
- **4分钟歌曲** - 生成最长240秒的连续音频
- **音频编辑** - Cover、Repaint、Extend、Edit、Retake 功能
- **LoRA 支持** - 加载微调适配器以实现特殊风格
- **HeartMuLa 兼容** - 与 HeartMuLa 节点无缝协作

## 节点列表

| 节点 | 描述 |
|------|------|
| **Model Loader** | 下载并缓存 ACE-Step 模型 |
| **Settings** | 配置生成参数（时长、语言、BPM等） |
| **Generator** | 从描述和歌词生成音乐（Text2Music） |
| **Lyrics Input** | 带段落标记的歌词输入专用节点 |
| **Caption Input** | 风格/流派描述输入专用节点 |
| **Cover** | 将现有音频转换为不同风格（Audio2Audio） |
| **Repaint** | 重新生成音频的特定部分 |
| **Retake** | 创建现有音频的变体 |
| **Extend** | 在音频开头或结尾添加新内容 |
| **Edit** | 保留旋律的同时更改标签/歌词（FlowEdit） |
| **Conditioning** | 将参数组合为 Conditioning 对象 |
| **Generator (from Cond)** | 从 Conditioning 对象生成 |
| **Load LoRA** | 加载微调的 LoRA 适配器 |
| **Understand** | 从现有音频提取元数据 |
| **Create Sample** | 从自然语言查询生成参数 |

## 安装

### ComfyUI Manager（推荐）

搜索 "ComfyUI-AceMusic" 并安装。

### 手动安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hiroki-abe-58/ComfyUI-AceMusic.git
cd ComfyUI-AceMusic
pip install -r requirements.txt
```

### 安装 ACE-Step 1.5

```bash
pip install git+https://github.com/ace-step/ACE-Step.git
```

模型会在首次使用时自动从 Hugging Face 下载。

## 快速开始

1. 添加 **AceMusic Model Loader** 节点并选择设备（`cuda`）
2. 添加 **AceMusic Settings** 节点配置参数
3. 添加 **AceMusic Lyrics Input** 节点并输入歌词：
   ```
   [Verse]
   走在空旷的街道上
   想着你和我的过往
   
   [Chorus]
   我们属于彼此
   从现在到永远
   ```
4. 在 **AceMusic Caption Input** 中输入风格标签：`pop, female vocal, energetic`
5. 将所有节点连接到 **AceMusic Generator** -> **Preview Audio**

示例工作流可从 `workflow/AceMusic_Lyrics_v3.json` 加载。

## 段落标记

ACE-Step 支持以下歌曲结构标记：

| 标记 | 用途 |
|------|------|
| [Intro] | 开场器乐或人声引子 |
| [Verse] | 主歌 |
| [Pre-Chorus] | 副歌前的过渡 |
| [Chorus] | 副歌/高潮 |
| [Bridge] | 对比段落 |
| [Outro] | 结尾段落 |
| [Instrumental] | 纯器乐段落 |

## 风格标签

在描述中组合标签来控制输出风格：

- **流派**: pop, rock, electronic, jazz, classical, hip-hop, r&b, country, folk, metal, indie, c-pop, mandopop
- **人声**: female vocal, male vocal, duet, choir, instrumental
- **情绪**: energetic, melancholic, uplifting, calm, aggressive, romantic, dreamy, dark
- **节奏**: slow, medium, fast
- **乐器**: piano, guitar, drums, synth, strings, bass, violin, erhu, pipa

**示例**: `mandopop, female vocal, romantic, piano, emotional ballad`

## 模型与硬件

模型从 Hugging Face 自动下载到 `~/.cache/ace-step/checkpoints/`

### 性能

| 设备 | RTF（27步） | 1分钟音频生成时间 |
|------|-------------|-------------------|
| RTX 5090 | ~50x | ~1.2秒 |
| RTX 4090 | 34.48x | 1.74秒 |
| A100 | 27.27x | 2.20秒 |
| RTX 3090 | 12.76x | 4.70秒 |
| M2 Max | 2.27x | 26.43秒 |

### 显存要求

| 模式 | 显存 | 备注 |
|------|------|------|
| 普通 | 8GB+ | 全速 |
| CPU Offload | ~4GB | 较慢但适用于显存有限的环境 |

## 支持的语言

ACE-Step 支持19种语言，品质排名：

| 语言 | 代码 | 品质 |
|------|------|------|
| 英语 | en | 优秀 |
| 中文 | zh | 优秀 |
| 日语 | ja | 优秀 |
| 韩语 | ko | 很好 |
| 西班牙语 | es | 很好 |

## 许可证

Apache 2.0

## 致谢

- [ACE-Step](https://github.com/ace-step/ACE-Step) - ACE Studio 和 StepFun 开发的原始音乐生成模型
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 节点式 UI 框架
- [HeartMuLa](https://github.com/filliptm/ComfyUI_FL-HeartMuLa) - 节点设计灵感来源

## 链接

- [ACE-Step GitHub](https://github.com/ace-step/ACE-Step)
- [ACE-Step Hugging Face](https://huggingface.co/ACE-Step)
- [ACE Studio](https://www.acestudio.ai/)

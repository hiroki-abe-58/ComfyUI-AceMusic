# ComfyUI-AceMusic

**[English](README.md)** | **[日本語](README_ja.md)** | **[简体中文](README_zh-CN.md)** | **繁體中文** | **[한국어](README_ko.md)** | **[Tiếng Việt](README_vi.md)**

基於 [ACE-Step](https://github.com/ace-step/ACE-Step) 的 ComfyUI 多語言 AI 音樂生成節點。支援19種語言生成帶歌詞的完整歌曲，包括中文、英語、日語、韓語等。

![工作流程預覽](assets/workflow_preview.png)

---

**如果這個專案對您有幫助，請給個 Star 支持一下！**

[![GitHub stars](https://img.shields.io/github/stars/hiroki-abe-58/ComfyUI-AceMusic?style=social)](https://github.com/hiroki-abe-58/ComfyUI-AceMusic)

---

## 亮點

- **全球首個 ComfyUI 完整 ACE-Step 整合** - 將 ACE-Step 的所有功能完整實現為 ComfyUI 節點（共15個節點）
- **模組化架構** - 分離 Settings/Lyrics/Caption 節點，消除元件順序問題，提升工作流程可讀性
- **跨平台相容** - 使用 soundfile/scipy 取代有問題的 torchaudio 後端，支援 Windows + Python 3.13+
- **HeartMuLa 互操作** - 與 HeartMuLa 節點無縫銜接，實現混合 AI 音樂工作流程
- **生產環境就緒** - 強大的輸入驗證和自動回退機制，防止執行時錯誤

## 功能特性

- **多語言歌詞** - 支援19種語言（中文、英語、日語、韓語、西班牙語等）生成帶人聲的音樂
- **歌曲結構控制** - 使用 [Verse]、[Chorus]、[Bridge] 等標記定義歌曲結構
- **風格標籤** - 控制曲風、人聲類型、情緒、節奏和樂器
- **4分鐘歌曲** - 生成最長240秒的連續音訊
- **音訊編輯** - Cover、Repaint、Extend、Edit、Retake 功能
- **LoRA 支援** - 載入微調適配器以實現特殊風格
- **HeartMuLa 相容** - 與 HeartMuLa 節點無縫協作

## 節點列表

| 節點 | 描述 |
|------|------|
| **Model Loader** | 下載並快取 ACE-Step 模型 |
| **Settings** | 設定生成參數（時長、語言、BPM等） |
| **Generator** | 從描述和歌詞生成音樂（Text2Music） |
| **Lyrics Input** | 帶段落標記的歌詞輸入專用節點 |
| **Caption Input** | 風格/曲風描述輸入專用節點 |
| **Cover** | 將現有音訊轉換為不同風格（Audio2Audio） |
| **Repaint** | 重新生成音訊的特定部分 |
| **Retake** | 建立現有音訊的變體 |
| **Extend** | 在音訊開頭或結尾新增內容 |
| **Edit** | 保留旋律的同時更改標籤/歌詞（FlowEdit） |
| **Conditioning** | 將參數組合為 Conditioning 物件 |
| **Generator (from Cond)** | 從 Conditioning 物件生成 |
| **Load LoRA** | 載入微調的 LoRA 適配器 |
| **Understand** | 從現有音訊擷取中繼資料 |
| **Create Sample** | 從自然語言查詢生成參數 |

## 安裝

### ComfyUI Manager（推薦）

搜尋 "ComfyUI-AceMusic" 並安裝。

### 手動安裝

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hiroki-abe-58/ComfyUI-AceMusic.git
cd ComfyUI-AceMusic
pip install -r requirements.txt
```

### 安裝 ACE-Step

```bash
pip install git+https://github.com/ace-step/ACE-Step.git
```

模型會在首次使用時自動從 Hugging Face 下載。

## 快速開始

1. 新增 **AceMusic Model Loader** 節點並選擇裝置（`cuda`）
2. 新增 **AceMusic Settings** 節點設定參數
3. 新增 **AceMusic Lyrics Input** 節點並輸入歌詞：
   ```
   [Verse]
   走在空曠的街道上
   想著你和我的過往
   
   [Chorus]
   我們屬於彼此
   從現在到永遠
   ```
4. 在 **AceMusic Caption Input** 中輸入風格標籤：`pop, female vocal, energetic`
5. 將所有節點連接到 **AceMusic Generator** -> **Preview Audio**

範例工作流程可從 `workflow/AceMusic_Lyrics_v3.json` 載入。

## 段落標記

ACE-Step 支援以下歌曲結構標記：

| 標記 | 用途 |
|------|------|
| [Intro] | 開場器樂或人聲引子 |
| [Verse] | 主歌 |
| [Pre-Chorus] | 副歌前的過渡 |
| [Chorus] | 副歌/高潮 |
| [Bridge] | 對比段落 |
| [Outro] | 結尾段落 |
| [Instrumental] | 純器樂段落 |

## 風格標籤

在描述中組合標籤來控制輸出風格：

- **曲風**: pop, rock, electronic, jazz, classical, hip-hop, r&b, country, folk, metal, indie, c-pop, mandopop
- **人聲**: female vocal, male vocal, duet, choir, instrumental
- **情緒**: energetic, melancholic, uplifting, calm, aggressive, romantic, dreamy, dark
- **節奏**: slow, medium, fast
- **樂器**: piano, guitar, drums, synth, strings, bass, violin, erhu, pipa

**範例**: `mandopop, female vocal, romantic, piano, emotional ballad`

## 模型與硬體

模型從 Hugging Face 自動下載到 `~/.cache/ace-step/checkpoints/`

### 效能

| 裝置 | RTF（27步） | 1分鐘音訊生成時間 |
|------|-------------|-------------------|
| RTX 5090 | ~50x | ~1.2秒 |
| RTX 4090 | 34.48x | 1.74秒 |
| A100 | 27.27x | 2.20秒 |
| RTX 3090 | 12.76x | 4.70秒 |
| M2 Max | 2.27x | 26.43秒 |

### 顯示記憶體需求

| 模式 | 顯存 | 備註 |
|------|------|------|
| 一般 | 8GB+ | 全速 |
| CPU Offload | ~4GB | 較慢但適用於顯存有限的環境 |

## 支援的語言

ACE-Step 支援19種語言，品質排名：

| 語言 | 代碼 | 品質 |
|------|------|------|
| 英語 | en | 優秀 |
| 中文 | zh | 優秀 |
| 日語 | ja | 優秀 |
| 韓語 | ko | 很好 |
| 西班牙語 | es | 很好 |

## 授權條款

Apache 2.0

## 致謝

- [ACE-Step](https://github.com/ace-step/ACE-Step) - ACE Studio 和 StepFun 開發的原始音樂生成模型
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 節點式 UI 框架
- [HeartMuLa](https://github.com/filliptm/ComfyUI_FL-HeartMuLa) - 節點設計靈感來源

## 連結

- [ACE-Step GitHub](https://github.com/ace-step/ACE-Step)
- [ACE-Step Hugging Face](https://huggingface.co/ACE-Step)
- [ACE Studio](https://www.acestudio.ai/)

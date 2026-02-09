# ComfyUI-AceMusic

**[English](README.md)** | **[日本語](README_ja.md)** | **[简体中文](README_zh-CN.md)** | **[繁體中文](README_zh-TW.md)** | **한국어** | **[Tiếng Việt](README_vi.md)**

[ACE-Step](https://github.com/ace-step/ACE-Step) 기반 ComfyUI용 다국어 AI 음악 생성 노드. 한국어, 영어, 중국어, 일본어 등 19개 언어로 가사가 있는 완전한 노래를 생성할 수 있습니다.

![워크플로우 미리보기](assets/workflow_preview.png)

---

**이 프로젝트가 도움이 되셨다면 Star를 눌러주세요!**

[![GitHub stars](https://img.shields.io/github/stars/hiroki-abe-58/ComfyUI-AceMusic?style=social)](https://github.com/hiroki-abe-58/ComfyUI-AceMusic)

---

## 하이라이트

- **세계 최초 ComfyUI용 완전한 ACE-Step 통합** - ACE-Step의 모든 기능을 ComfyUI 노드로 완전 구현 (총 15개 노드)
- **모듈식 아키텍처** - Settings/Lyrics/Caption 노드를 분리하여 위젯 순서 문제 해결 및 워크플로우 가독성 향상
- **크로스 플랫폼 호환** - 문제가 있는 torchaudio 백엔드 대신 soundfile/scipy 사용으로 Windows + Python 3.13+ 지원
- **HeartMuLa 상호 운용** - HeartMuLa 노드와 원활하게 연결하여 하이브리드 AI 음악 워크플로우 구현
- **프로덕션 환경 대응** - 강력한 입력 검증과 자동 폴백으로 런타임 오류 방지

## 기능

- **다국어 가사** - 19개 언어(한국어, 영어, 중국어, 일본어, 스페인어 등)로 보컬이 있는 음악 생성
- **곡 구조 제어** - [Verse], [Chorus], [Bridge] 등 섹션 마커로 곡 구조 정의
- **스타일 태그** - 장르, 보컬 유형, 분위기, 템포, 악기 제어
- **4분 노래** - 최대 240초의 연속 오디오 생성
- **오디오 편집** - Cover, Repaint, Extend, Edit, Retake 기능
- **LoRA 지원** - 특수 스타일용 파인튜닝 어댑터 로드
- **HeartMuLa 호환** - HeartMuLa 노드와 원활하게 연동

## 노드 목록

| 노드 | 설명 |
|------|------|
| **Model Loader** | ACE-Step 모델 다운로드 및 캐시 |
| **Settings** | 생성 매개변수 설정 (길이, 언어, BPM 등) |
| **Generator** | 설명과 가사로 음악 생성 (Text2Music) |
| **Lyrics Input** | 섹션 마커가 있는 가사 입력 전용 노드 |
| **Caption Input** | 스타일/장르 설명 입력 전용 노드 |
| **Cover** | 기존 오디오를 다른 스타일로 변환 (Audio2Audio) |
| **Repaint** | 오디오의 특정 부분 재생성 |
| **Retake** | 기존 오디오의 변형 생성 |
| **Extend** | 오디오 시작 또는 끝에 새 콘텐츠 추가 |
| **Edit** | 멜로디를 유지하면서 태그/가사 변경 (FlowEdit) |
| **Conditioning** | 매개변수를 Conditioning 객체로 결합 |
| **Generator (from Cond)** | Conditioning 객체에서 생성 |
| **Load LoRA** | 파인튜닝된 LoRA 어댑터 로드 |
| **Understand** | 기존 오디오에서 메타데이터 추출 |
| **Create Sample** | 자연어 쿼리에서 매개변수 생성 |

## 설치

### ComfyUI Manager (권장)

"ComfyUI-AceMusic"을 검색하여 설치하세요.

### 수동 설치

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hiroki-abe-58/ComfyUI-AceMusic.git
cd ComfyUI-AceMusic
pip install -r requirements.txt
```

### ACE-Step 설치

```bash
pip install git+https://github.com/ace-step/ACE-Step.git
```

모델은 처음 사용할 때 Hugging Face에서 자동으로 다운로드됩니다.

## 빠른 시작

1. **AceMusic Model Loader** 노드를 추가하고 장치 선택 (`cuda`)
2. **AceMusic Settings** 노드를 추가하여 매개변수 설정
3. **AceMusic Lyrics Input** 노드를 추가하고 가사 입력:
   ```
   [Verse]
   빈 거리를 걸어가며
   너와 나를 생각해
   
   [Chorus]
   우리는 하나야
   지금 그리고 영원히
   ```
4. **AceMusic Caption Input**에 스타일 태그 입력: `k-pop, female vocal, energetic`
5. 모든 노드를 **AceMusic Generator** -> **Preview Audio**에 연결

예제 워크플로우는 `workflow/AceMusic_Lyrics_v3.json`에서 로드할 수 있습니다.

## 섹션 마커

ACE-Step은 다음 곡 구조 마커를 지원합니다:

| 마커 | 용도 |
|------|------|
| [Intro] | 오프닝 인스트루멘탈 또는 보컬 인트로 |
| [Verse] | 메인 벌스 |
| [Pre-Chorus] | 코러스 전 빌드업 |
| [Chorus] | 메인 훅/코러스 |
| [Bridge] | 대조적인 섹션 |
| [Outro] | 엔딩 섹션 |
| [Instrumental] | 보컬 없는 섹션 |

## 스타일 태그

설명에서 태그를 조합하여 출력 스타일을 제어합니다:

- **장르**: pop, rock, electronic, jazz, classical, hip-hop, r&b, country, folk, metal, indie, k-pop, trot
- **보컬**: female vocal, male vocal, duet, choir, instrumental
- **분위기**: energetic, melancholic, uplifting, calm, aggressive, romantic, dreamy, dark
- **템포**: slow, medium, fast
- **악기**: piano, guitar, drums, synth, strings, bass, violin, gayageum

**예시**: `k-pop, female vocal, energetic, synth, catchy melody`

## 모델 및 하드웨어

모델은 Hugging Face에서 `~/.cache/ace-step/checkpoints/`로 자동 다운로드됩니다.

### 성능

| 장치 | RTF (27단계) | 1분 오디오 생성 시간 |
|------|--------------|---------------------|
| RTX 5090 | ~50x | ~1.2초 |
| RTX 4090 | 34.48x | 1.74초 |
| A100 | 27.27x | 2.20초 |
| RTX 3090 | 12.76x | 4.70초 |
| M2 Max | 2.27x | 26.43초 |

### VRAM 요구 사항

| 모드 | VRAM | 비고 |
|------|------|------|
| 일반 | 8GB+ | 최고 속도 |
| CPU Offload | ~4GB | 느리지만 VRAM이 제한된 환경에서 작동 |

## 지원 언어

ACE-Step은 19개 언어를 지원합니다. 품질 순위:

| 언어 | 코드 | 품질 |
|------|------|------|
| 영어 | en | 우수 |
| 중국어 | zh | 우수 |
| 일본어 | ja | 우수 |
| 한국어 | ko | 매우 좋음 |
| 스페인어 | es | 매우 좋음 |

## 라이선스

Apache 2.0

## 크레딧

- [ACE-Step](https://github.com/ace-step/ACE-Step) - ACE Studio와 StepFun이 개발한 원본 음악 생성 모델
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - 노드 기반 UI 프레임워크
- [HeartMuLa](https://github.com/filliptm/ComfyUI_FL-HeartMuLa) - 노드 디자인 영감

## 링크

- [ACE-Step GitHub](https://github.com/ace-step/ACE-Step)
- [ACE-Step Hugging Face](https://huggingface.co/ACE-Step)
- [ACE Studio](https://www.acestudio.ai/)

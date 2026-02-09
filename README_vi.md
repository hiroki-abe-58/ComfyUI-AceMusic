# ComfyUI-AceMusic

**[English](README.md)** | **[日本語](README_ja.md)** | **[简体中文](README_zh-CN.md)** | **[繁體中文](README_zh-TW.md)** | **[한국어](README_ko.md)** | **Tiếng Việt**

Các node tạo nhạc AI đa ngôn ngữ cho ComfyUI dựa trên [ACE-Step](https://github.com/ace-step/ACE-Step). Tạo các bài hát hoàn chỉnh có lời bằng 19 ngôn ngữ bao gồm tiếng Việt, tiếng Anh, tiếng Trung, tiếng Nhật, tiếng Hàn và nhiều ngôn ngữ khác.

![Xem trước Workflow](assets/workflow_preview.png)

---

**Nếu dự án này hữu ích với bạn, hãy cho một Star nhé!**

[![GitHub stars](https://img.shields.io/github/stars/hiroki-abe-58/ComfyUI-AceMusic?style=social)](https://github.com/hiroki-abe-58/ComfyUI-AceMusic)

---

## Điểm nổi bật

- **Tích hợp ACE-Step đầy đủ đầu tiên trên thế giới cho ComfyUI** - Triển khai hoàn chỉnh tất cả tính năng ACE-Step dưới dạng node ComfyUI (tổng cộng 15 node)
- **Kiến trúc module** - Tách biệt các node Settings/Lyrics/Caption, loại bỏ vấn đề thứ tự widget và cải thiện khả năng đọc workflow
- **Tương thích đa nền tảng** - Hoạt động trên Windows với Python 3.13+ bằng cách sử dụng soundfile/scipy thay vì backend torchaudio có vấn đề
- **Tương tác với HeartMuLa** - Kết nối liền mạch với các node HeartMuLa cho workflow nhạc AI kết hợp
- **Sẵn sàng cho production** - Xác thực đầu vào mạnh mẽ với fallback tự động ngăn ngừa lỗi runtime

## Tính năng

- **Lời bài hát đa ngôn ngữ** - Tạo nhạc có giọng hát bằng 19 ngôn ngữ (tiếng Việt, tiếng Anh, tiếng Trung, tiếng Nhật, tiếng Hàn, v.v.)
- **Kiểm soát cấu trúc bài hát** - Sử dụng các đánh dấu phần như [Verse], [Chorus], [Bridge] để định nghĩa cấu trúc
- **Thẻ phong cách** - Kiểm soát thể loại, loại giọng hát, tâm trạng, nhịp độ và nhạc cụ
- **Bài hát 4 phút** - Tạo audio liên tục lên đến 240 giây
- **Chỉnh sửa audio** - Các tính năng Cover, Repaint, Extend, Edit, Retake
- **Hỗ trợ LoRA** - Tải các adapter đã fine-tune cho phong cách đặc biệt
- **Tương thích HeartMuLa** - Hoạt động liền mạch với các node HeartMuLa

## Danh sách Node

| Node | Mô tả |
|------|-------|
| **Model Loader** | Tải và cache các model ACE-Step |
| **Settings** | Cấu hình tham số tạo (thời lượng, ngôn ngữ, BPM, v.v.) |
| **Generator** | Tạo nhạc từ mô tả và lời bài hát (Text2Music) |
| **Lyrics Input** | Node chuyên dụng để nhập lời với đánh dấu phần |
| **Caption Input** | Node chuyên dụng để nhập mô tả phong cách/thể loại |
| **Cover** | Chuyển đổi audio hiện có sang phong cách khác (Audio2Audio) |
| **Repaint** | Tạo lại các phần cụ thể của audio |
| **Retake** | Tạo biến thể của audio hiện có |
| **Extend** | Thêm nội dung mới vào đầu hoặc cuối audio |
| **Edit** | Thay đổi thẻ/lời trong khi giữ nguyên giai điệu (FlowEdit) |
| **Conditioning** | Kết hợp tham số thành đối tượng Conditioning |
| **Generator (from Cond)** | Tạo từ đối tượng Conditioning |
| **Load LoRA** | Tải adapter LoRA đã fine-tune |
| **Understand** | Do thoi luong audio (caption/BPM/key la placeholder*) |
| **Create Sample** | Tao tham so qua phuong phap heuristic tu khoa* |

> \* Phan tich audio va tao tham so bang AI can phien ban ACE-Step trong tuong lai. Hien tai cung cap do thoi luong chinh xac va suy luan dua tren tu khoa lam placeholder.

## Cài đặt

### ComfyUI Manager (Khuyến nghị)

Tìm kiếm "ComfyUI-AceMusic" và cài đặt.

### Cài đặt thủ công

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/hiroki-abe-58/ComfyUI-AceMusic.git
cd ComfyUI-AceMusic
pip install -r requirements.txt
```

### Cài đặt ACE-Step

```bash
pip install git+https://github.com/ace-step/ACE-Step.git
```

Các model sẽ tự động tải từ Hugging Face khi sử dụng lần đầu.

## Bắt đầu nhanh

1. Thêm node **AceMusic Model Loader** và chọn thiết bị (`cuda`)
2. Thêm node **AceMusic Settings** để cấu hình tham số
3. Thêm node **AceMusic Lyrics Input** và nhập lời:
   ```
   [Verse]
   Đi trên con đường vắng
   Nghĩ về anh và em
   
   [Chorus]
   Chúng ta thuộc về nhau
   Bây giờ và mãi mãi
   ```
4. Nhập thẻ phong cách vào **AceMusic Caption Input**: `v-pop, female vocal, romantic`
5. Kết nối tất cả vào **AceMusic Generator** -> **Preview Audio**

Workflow mẫu có thể tải từ `workflow/AceMusic_Lyrics_v3.json`.

## Đánh dấu phần

ACE-Step hỗ trợ các đánh dấu cấu trúc bài hát sau:

| Đánh dấu | Cách dùng |
|----------|-----------|
| [Intro] | Phần mở đầu nhạc cụ hoặc giọng hát |
| [Verse] | Các đoạn chính |
| [Pre-Chorus] | Phần dẫn trước điệp khúc |
| [Chorus] | Hook chính/điệp khúc |
| [Bridge] | Phần tương phản |
| [Outro] | Phần kết thúc |
| [Instrumental] | Các phần không có giọng hát |

## Thẻ phong cách

Kết hợp các thẻ trong mô tả để kiểm soát phong cách đầu ra:

- **Thể loại**: pop, rock, electronic, jazz, classical, hip-hop, r&b, country, folk, metal, indie, v-pop, ballad
- **Giọng hát**: female vocal, male vocal, duet, choir, instrumental
- **Tâm trạng**: energetic, melancholic, uplifting, calm, aggressive, romantic, dreamy, dark
- **Nhịp độ**: slow, medium, fast
- **Nhạc cụ**: piano, guitar, drums, synth, strings, bass, violin, dan bau, dan tranh

**Ví dụ**: `v-pop, female vocal, romantic, piano, emotional ballad`

## Model & Phần cứng

Các model tự động tải từ Hugging Face vào `~/.cache/ace-step/checkpoints/`

### Hiệu suất

| Thiết bị | RTF (27 bước) | Thời gian tạo 1 phút audio |
|----------|---------------|----------------------------|
| RTX 5090 | ~50x | ~1.2 giây |
| RTX 4090 | 34.48x | 1.74 giây |
| A100 | 27.27x | 2.20 giây |
| RTX 3090 | 12.76x | 4.70 giây |
| M2 Max | 2.27x | 26.43 giây |

### Yêu cầu VRAM

| Chế độ | VRAM | Ghi chú |
|--------|------|---------|
| Bình thường | 8GB+ | Tốc độ tối đa |
| CPU Offload | ~4GB | Chậm hơn nhưng hoạt động với VRAM hạn chế |

## Ngôn ngữ được hỗ trợ

ACE-Step hỗ trợ 19 ngôn ngữ. Xếp hạng chất lượng:

| Ngôn ngữ | Mã | Chất lượng |
|----------|-----|------------|
| Tiếng Anh | en | Xuất sắc |
| Tiếng Trung | zh | Xuất sắc |
| Tiếng Nhật | ja | Xuất sắc |
| Tiếng Hàn | ko | Rất tốt |
| Tiếng Tây Ban Nha | es | Rất tốt |

## Giấy phép

Apache 2.0

## Ghi nhận

- [ACE-Step](https://github.com/ace-step/ACE-Step) - Model tạo nhạc gốc bởi ACE Studio và StepFun
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - Framework UI dựa trên node
- [HeartMuLa](https://github.com/filliptm/ComfyUI_FL-HeartMuLa) - Nguồn cảm hứng thiết kế node

## Liên kết

- [ACE-Step GitHub](https://github.com/ace-step/ACE-Step)
- [ACE-Step Hugging Face](https://huggingface.co/ACE-Step)
- [ACE Studio](https://www.acestudio.ai/)

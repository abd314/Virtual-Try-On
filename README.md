# 👕 Virtual Try-On System with AI

![Virtual Try-On Demo](assets/demo.png)

A cutting-edge **AI-powered virtual fitting room** that lets users try on new clothes using text prompts. This system combines **object detection**, **instance segmentation**, and **diffusion-based inpainting** to realistically overlay clothing onto user-uploaded images.

Perfect for fashion tech demos, e-commerce applications, and AI research!

---

## 🎯 Features

- ✅ **Upload your photo** – Full-body image support
- ✅ **Select region**: Upper body (shirt, jacket) or Lower body (pants, skirt)
- ✅ **Text-to-clothing**: Describe any outfit (e.g., *"a blue denim jacket"*)
- ✅ **Automatic detection & segmentation** using YOLO + SAM 2.1
- ✅ **High-quality inpainting** via Segmind's SDXL Inpaint API
- ✅ **Gradio UI** – Interactive and easy-to-use web interface

---

## 🔧 How It Works

1. **Detection**: [YOLO model](https://universe.roboflow.com/bruuj/main-fashion-wmyfk) detects clothing items in the uploaded image.
2. **Segmentation**: [SAM 2.1](https://docs.ultralytics.com/models/sam-2/) segments the selected clothing region using bounding box prompts.
3. **Inpainting**: [Segmind SDXL Inpaint API](https://segmind.com) generates realistic clothing based on your text prompt.
4. **Output**: View the final try-on result alongside the original image and segmentation mask.


---

## 💻 Run Locally (VS Code / Local Machine)

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/virtual-tryon.git
cd virtual-tryon

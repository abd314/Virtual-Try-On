# 👕 Virtual Try-On System with AI

![Virtual Try-On Demo](assets/demo.png)

A powerful **AI-powered virtual fitting room** that lets users try on new clothes using just a photo and a text prompt. This system combines:

- **Clothing Detection** (YOLO via Roboflow)
- **Instance Segmentation** (SAM 2.1)
- **Text-to-Clothing Inpainting** (Segmind SDXL)
- **Interactive Gradio UI**

Perfect for fashion tech, e-commerce demos, or AI research!

---

## 🚀 Quick Start Guide

Follow these steps to **run this project locally on your machine** (e.g., in VS Code or any Python environment).

---

## 🔧 Step 1: Install Required Software

Make sure you have the following installed:

| Tool | Download Link |
|------|---------------|
| **Python 3.10 or 3.11** | [python.org](https://www.python.org/downloads/) |
| **Git** | [git-scm.com](https://git-scm.com/) |
| **VS Code (Optional)** | [code.visualstudio.com](https://code.visualstudio.com/) |

> 💡 SAM 2.1 and PyTorch require Python 3.10+ — **do not use 3.12 or higher** for compatibility.

---

## 📄 File Structure 

virtual-tryon/
├── app.py
├── inference_pipeline.py
├── requirements.txt
├── .env
└── assets/ (optional: for demo images)


---

## 📦 Step 3: Setup an Run

# 1. Clone the repository
git clone https://github.com/yourusername/virtual-tryon.git
cd virtual-tryon

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 4. Install all required packages
pip install -r requirements.txt

✅ This installs: ultralytics, inference-sdk, gradio, torch, opencv, and more. 

    
---

## 🗝️ Step 3: Get Your API Keys

ROBO_API_KEY=your_roboflow_api_key_here
SEG_API_KEY=your_segmind_api_key_here


## ▶️ Step 5: Run the App

'''bash 
python app.py






 





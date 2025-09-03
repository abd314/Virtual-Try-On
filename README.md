# 👕 Virtual Try-On System with AI


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

1. **Detection**: [YOLO model](https://universe.roboflow.com/bruuj/main-fashion-wmyfk) detects clothing items in the uploaded image. Or, From ultralytics YOLO("best.pt").
2. **Segmentation**: [SAM 2.1](https://docs.ultralytics.com/models/sam-2/) segments the selected clothing region using bounding box prompts.
3. **Inpainting**: [Segmind flux-kontext-dev API](https://segmind.com) generates realistic clothing based on your text prompt.
4. **Output**: View the final try-on result alongside the original image and segmentation mask.

---

## 📂 Project Structure


```
virtual-tryon/
├── app.py                  # Gradio frontend
├── inference_pipeline.py   # Core logic: detect → segment → inpaint
├── requirements.txt        # Python dependencies
├── .env                    # Your private API keys
├── .gitignore
├── README.md
└── assets/                 # Demo images and results

```
---


## 💻 Run Locally (VS Code / Local Machine)

### 1. Clone the Repository


git clone https://github.com/yourusername/virtual-tryon.git
cd virtual-tryon


### 2. Set Up Virtual Environment


```bash
python -m venv venv
source venv/bin/activate 
```

---

### 3. Install Dependencies

```txt
pip install -r requirements.txt
```


> 📌 Requires Python 3.10+ for full compatibility with PyTorch and Ultralytics.

### 4. Get Your API Keys

You'll need two free API keys:

| Service | Link | Purpose |
|--------|------|--------|
| **Roboflow** | [https://app.roboflow.com](https://app.roboflow.com) | Object detection (`main-fashion-wmyfk`) |
| **Segmind** | [https://segmind.com](https://segmind.com) | SDXL Inpainting API |

After signing up, copy your keys.



### 5. Create `.env` File

Create a `.env` file in the project root:

```env
ROBO_API_KEY=your_roboflow_api_key_here
SEG_API_KEY=your_segmind_api_key_here

```




> 🔐 Never commit this file! It's already in `.gitignore`.

### 6. Run the App

```bash
python app.py

```


Open the local URL (usually `http://127.0.0.1:7860`) in your browser.

---


---

## 🛠️ Model Details

| Component | Model Used |
|---------|------------|
| **Clothing Detection** | [`bruuj/main-fashion-wmyfk`](https://universe.roboflow.com/bruuj/main-fashion-wmyfk) (YOLOv8) |
| **Segmentation** | [`sam2.1_b.pt`](https://docs.ultralytics.com/models/sam-2/) (Segment Anything Model 2.1) |
| **Inpainting** | `SDXL Inpaint` via [Segmind API](https://api.segmind.com/v1/sdxl-inpaint) |

> ✅ All models are automatically downloaded on first run.

---

## 🧪 Example Prompts

Try these in the text box:

- `"A red Hawaiian shirt with floral print"`
- `"Black leather jacket with silver zippers"`
- `"Blue denim jeans with ripped knees"`
- `"A flowing white summer dress"`

💡 **Tip**: Be specific about color, texture, and style for best results.

---

## 📸 Sample Output

![Virtual Try-On Demo](assets/demo.png)

---








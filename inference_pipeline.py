# inference_pipeline.py
import os
import cv2
import numpy as np
from PIL import Image
import io
import requests
import base64
from typing import Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import after installing
from inference_sdk import InferenceHTTPClient
from ultralytics import SAM
import supervision as sv

# Load API keys
ROBO_API_KEY = os.getenv("ROBO_API_KEY")
SEG_API_KEY = os.getenv("SEG_API_KEY")

if not ROBO_API_KEY:
    raise ValueError("ROBO_API_KEY not found in .env file")
if not SEG_API_KEY:
    raise ValueError("SEG_API_KEY not found in .env file")

# Initialize clients
ROBO_CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBO_API_KEY
)

SAM_MODEL = SAM("sam2.1_b.pt")

# Class mapping based on your model: https://universe.roboflow.com/bruuj/main-fashion-wmyfk
# Confirm these class names in your Roboflow project
CLASS_NAMES = ["shirt", "jacket", "top", "pants", "skirt", "dress", "shoes"]
UPPER_CLASSES = [0, 1, 2]   # shirt, jacket, top
LOWER_CLASSES = [3, 4]      # pants, skirt
DRESS_CLASS = [5]           # dress (optional)

REGION_TO_CLASSES = {
    "upper": UPPER_CLASSES,
    "lower": LOWER_CLASSES
}


def image_to_base64(image: np.ndarray) -> str:
    """Convert RGB image (np array) to base64 string."""
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')


def detect_clothing(image_path: str, region: str) -> dict:
    """Run YOLO detection and filter by region."""
    result = ROBO_CLIENT.infer(image_path, model_id="main-fashion-wmyfk/1")
    
    target_ids = REGION_TO_CLASSES[region.lower()]
    filtered_preds = [
        pred for pred in result["predictions"]
        if pred["class_id"] in target_ids
    ]
    
    result["predictions"] = filtered_preds
    return result


def extract_bboxes(detection_results) -> list:
    """Extract bounding boxes in [x_min, y_min, x_max, y_max] format."""
    predictions = detection_results["predictions"]
    bboxes = [
        [
            int(pred["x"] - pred["width"] / 2),
            int(pred["y"] - pred["height"] / 2),
            int(pred["x"] + pred["width"] / 2),
            int(pred["y"] + pred["height"] / 2)
        ]
        for pred in predictions
    ]
    return bboxes


def segment_clothing(image_path: str, bboxes: list) -> np.ndarray:
    """Use SAM to generate binary mask from bounding box."""
    if not bboxes:
        return None

    results = SAM_MODEL(image_path, bboxes=bboxes[0])  # Use first detected item
    mask = results[0].masks.data[0].cpu().numpy()  # (H, W)
    binary_mask = (mask > 0.5).astype(np.uint8) * 255
    return binary_mask


def inpaint_with_prompt(image: np.ndarray, mask: np.ndarray, prompt: str) -> Image.Image:
    """Send image and mask to Segmind SDXL Inpaint API."""
    url = "https://api.segmind.com/v1/sdxl-inpaint"
    headers = {'x-api-key': SEG_API_KEY}

    # Convert images to base64
    img_b64 = image_to_base64(image)
    mask_b64 = base64.b64encode(cv2.imencode('.png', mask)[1]).decode('utf-8')

    data = {
        "image": img_b64,
        "mask": mask_b64,
        "prompt": prompt,
        "negative_prompt": "bad quality, blurry, cartoon, deformed, painting",
        "samples": 1,
        "scheduler": "DDIM",
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
        "seed": 42,
        "strength": 0.85,
        "base64": False
    }

    response = requests.post(url, json=data, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Inpainting failed: {response.text}")

    return Image.open(io.BytesIO(response.content))


def run_virtual_tryon(image_path: str, region: str, prompt: str) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """
    Full pipeline: detect → segment → inpaint
    Returns: original, mask, output
    """
    detection_result = detect_clothing(image_path, region)
    bboxes = extract_bboxes(detection_result)

    if not bboxes:
        raise ValueError(f"No {region} clothing detected. Try a clearer full-body image.")

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = segment_clothing(image_path, bboxes)

    if mask is None:
        raise ValueError("Segmentation failed.")

    result_image = inpaint_with_prompt(image_rgb, mask, prompt)

    original_pil = Image.fromarray(image_rgb)
    mask_pil = Image.fromarray(mask, mode="L")

    return original_pil, mask_pil, result_image

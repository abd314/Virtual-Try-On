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
from ultralytics import YOLO

# Load API keys
SEG_API_KEY = os.getenv("SEG_API_KEY")

if not SEG_API_KEY:
    raise ValueError("SEG_API_KEY not found in .env file")

# Initialize YOLO model
YOLO_MODEL = YOLO("best.pt")

# Initialize clients
ROBO_CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBO_API_KEY
)

SAM_MODEL = SAM("sam2.1_b.pt")



def image_to_base64(image: np.ndarray) -> str:
    """Convert RGB image (np array) to base64 string."""
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode('utf-8')


def detect_clothing(image_path: str, region: str) -> dict:
    """Run YOLO detection and filter based on spatial position (upper/lower)."""
    results = YOLO_MODEL.predict(source=image_path, conf=0.25, save=False, save_txt=False, verbose=False)
    results_obj = results[0]
    boxes = results_obj.boxes
    filtered_boxes = []
    if boxes is not None:
        boxes_data = boxes.data.cpu().numpy()  # shape: (num_boxes, 6)
        if len(boxes_data) == 0:
            return {"predictions": []}

        # Calculate average y position (top of bounding boxes)
        y_positions = boxes_data[:, 1]  # y1 coordinates
        avg_y = np.mean(y_positions)

        # Filter boxes based on position relative to average y
        for box in boxes_data:
            x1, y1, x2, y2, score, class_id = box
            box_center_y = (y1 + y2) / 2

            # Check if box belongs to selected region
            if (region.lower() == "upper" and box_center_y < avg_y) or \
               (region.lower() == "lower" and box_center_y > avg_y):
                filtered_boxes.append({
                    "x": float((x1 + x2) / 2),
                    "y": float((y1 + y2) / 2),
                    "width": float(x2 - x1),
                    "height": float(y2 - y1),
                    "score": float(score),
                    "class_id": int(class_id)
                })

    return {"predictions": filtered_boxes}


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
    """Send image and mask to Segmind Flux Kontext API."""
    url = "https://api.segmind.com/v1/flux-kontext-dev"
    headers = {'x-api-key': SEG_API_KEY}

    # Save image and mask to temporary files
    temp_image_path = "temp_input_image.png"
    temp_mask_path = "temp_mask.png"
    cv2.imwrite(temp_image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(temp_mask_path, mask)

    data = {
        "prompt": prompt,
        "guidance": 7,
        "num_inference_steps": 35,
        "aspect_ratio": "match_input_image",
        "output_format": "png",
        "output_quality": 90,
        "seed": 42,
        "disable_safety_checker": False
    }
    files = {
        "input_image": open(temp_image_path, "rb"),
        "mask": open(temp_mask_path, "rb")
    }

    response = requests.post(url, data=data, files=files, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Inpainting failed: {response.text}")

    return Image.open(io.BytesIO(response.content))


def run_virtual_tryon(image_path: str, region: str, prompt: str) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """
    Full pipeline: detect → segment → inpaint
    Returns: original, mask, output
    """
    detection_result = detect_clothing(image_path, region)
    all_bboxes = extract_bboxes(detection_result)

    if not all_bboxes:
        raise ValueError("No clothing detected. Try a clearer full-body image.")

    # Use the first detected bbox for the region
    selected_bbox = all_bboxes[0] if all_bboxes else None
    
    if selected_bbox is None:
        raise ValueError(f"No {region} clothing detected. Try a clearer full-body image.")

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = segment_clothing(image_path, [selected_bbox])

    if mask is None:
        raise ValueError("Segmentation failed.")

    # Compose prompt with region and clothing description
    full_prompt = f"{region} clothing: {prompt}"
    result_image = inpaint_with_prompt(image_rgb, mask, full_prompt)

    original_pil = Image.fromarray(image_rgb)
    mask_pil = Image.fromarray(mask, mode="L")

    return original_pil, mask_pil, result_image

# app.py
import gradio as gr
from inference_pipeline import run_virtual_tryon
import os

# Check environment
if not os.path.exists(".env"):
    gr.Warning("⚠️ .env file not found. Please create one with ROBO_API_KEY and SEG_API_KEY.")

def process_image(upload_image, region, prompt):
    if upload_image is None:
        raise gr.Error("Please upload an image.")
    if not prompt.strip():
        raise gr.Error("Please enter a clothing description.")

    temp_path = "uploaded_image.jpg"
    upload_image.save(temp_path)

    try:
        original, mask, result = run_virtual_tryon(temp_path, region, prompt)
        return original, mask, result
    except Exception as e:
        raise gr.Error(f"Error: {str(e)}")


# Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 👕 Virtual Try-On System")
    gr.Markdown("Upload your photo and try on new clothes using AI!")

    with gr.Row():
        with gr.Column():
            img_in = gr.Image(type="pil", label="Upload Image")
            region_sel = gr.Radio(["upper", "lower"], label="Clothing Region", value="upper")
            prompt_txt = gr.Textbox(label="Describe Clothing", placeholder="e.g., A red leather jacket")
            btn = gr.Button("Try On", variant="primary")

        with gr.Column():
            gr.Image(label="Original Image", interactive=False)
            gr.Image(label="Detected Mask", interactive=False)
            gr.Image(label="Try-On Result", interactive=False)

    btn.click(
        fn=process_image,
        inputs=[img_in, region_sel, prompt_txt],
        outputs=gr.Gallery(columns=1, height="auto")
    )

    gr.Markdown("### 💡 Tips")
    gr.Markdown("- Use full-body front-facing photos for best results.")
    gr.Markdown("- Avoid cluttered backgrounds.")

# Launch
if __name__ == "__main__":
    demo.launch()

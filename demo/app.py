import cv2
import numpy as np
import gradio as gr
from vein_pipeline import run_pipeline

def analyze(image):
    if image is None:
        return None, "Upload an NIR forearm image"
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    res = run_pipeline(gray)
    overlay_rgb = cv2.cvtColor(res["overlay_bgr"], cv2.COLOR_BGR2RGB)
    return overlay_rgb, res["report"]

demo = gr.Interface(
    fn=analyze,
    inputs=gr.Image(type="numpy", label="NIR forearm image"),
    outputs=[gr.Image(type="numpy", label="veins + ranked sites"),
             gr.Textbox(label="Report", lines=12)],
    title="Camera Only NIR Vein Pipeline (Research Demo: NOT FOR CLINICAL USE)",
    description=("upload a near infrared forearm image to see vein segmentation, "
                 "graph extraction, and rule based ranking of blood draw sites."),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
    share=True
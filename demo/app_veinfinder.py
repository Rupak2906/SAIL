import cv2
import numpy as np
import gradio as gr
import vein_finder as vf

def run(image, sensitivity, contrast, work_res):
    if image is None:
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    enh, vmap, mask = vf.extract_veins(gray, work=int(work_res),
                                       clip=float(contrast),
                                       sensitivity=float(sensitivity))
    overlay = vf.overlay_veinfinder(gray, vmap)
    return overlay, (mask * 255).astype(np.uint8)

demo = gr.Interface(
    fn=run,
    inputs=[
        gr.Image(type="numpy", label="NIR forearm image"),
        gr.Slider(0.03, 0.30, value=0.10, step=0.01, label="sensitivity (lower = more veins)"),
        gr.Slider(2.0, 6.0, value=4.0, step=0.5, label="contrast boost (CLAHE)"),
        gr.Slider(500, 1400, value=900, step=100, label="processing resolution"),
    ],
    outputs=[gr.Image(type="numpy", label="Vein overlay"),
             gr.Image(type="numpy", label="Vein mask")],
    title="NIR Vein (Research Demo, Not for Clinical Use)",
    description="Upload a near infrared forearm image."
                "Tune the sliders for your images.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
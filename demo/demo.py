import sys
import cv2
import numpy as np

import config as C
import unet_segment
from vein_pipeline import (extract_graph, compute_segment_geometry,
                           rank_candidates, visualize)

# resize-aware, ROI based quality control
def assess_quality(img_gray, work=768):
    H, W = img_gray.shape
    scale = work / max(H, W)
    small = cv2.resize(img_gray, (max(int(W * scale), 1), max(int(H * scale), 1))) \
        if scale < 1 else img_gray

    _, fg = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        roi = (lab == largest)
    else:
        roi = np.ones_like(small, bool)

    arm_frac = float(roi.mean())
    arm_px = small[roi]
    brightness = float(arm_px.mean()) if arm_px.size else 0.0
    contrast = float(arm_px.std()) if arm_px.size else 0.0
    sharpness = float(cv2.Laplacian(small, cv2.CV_64F).var())

    reasons = []
    if arm_frac < 0.08:   reasons.append("no clear arm region")
    if brightness < 30:   reasons.append("arm too dark / weak NIR signal")
    if contrast < 8:      reasons.append("low contrast inside arm")
    if sharpness < 2.0:   reasons.append("image extremely blurry")

    return dict(passed=len(reasons) == 0, reasons=reasons,
                brightness=round(brightness, 1), contrast=round(contrast, 1),
                sharpness=round(sharpness, 1), arm_frac=round(arm_frac, 2))


def build_report(qc, ranked):
    L = ["Camera Only NIR Vein Pipeline (Research Demo: NOT FOR CLINICAL USE)", ""]
    L.append(f"Image quality: {'PASS' if qc['passed'] else 'FAIL'}")
    if qc["reasons"]:
        L.append("  issues: " + "; ".join(qc["reasons"]))
    L.append(f"  brightness(ROI)={qc['brightness']} contrast(ROI)={qc['contrast']} "
             f"sharpness={qc['sharpness']} arm={qc['arm_frac']}")
    L.append("")
    if not ranked:
        L.append("No suitable vein segments found.")
    else:
        L.append(f"Top {len(ranked)} blood draw candidates:")
        for i, c in enumerate(ranked):
            px, py = c["insertion_point"]
            L.append(f"  #{i+1} score={c['score']:.2f} at (x={px},y={py})  "
                     + ", ".join(c["reasons"]))
    L.append("\nSegmentation: trained U-Net")
    return "\n".join(L)

def run_demo(img_gray):
    qc = assess_quality(img_gray)
    mask, prob = unet_segment.predict(img_gray)
    graph = extract_graph(mask)
    feats = compute_segment_geometry(graph)
    ranked = rank_candidates(feats, img_gray.shape, top_k=3)
    overlay = visualize(img_gray, mask, graph, ranked)
    return dict(qc=qc, mask=mask, prob=prob, ranked=ranked,
                overlay_bgr=overlay, report=build_report(qc, ranked))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python demo.py path/to/nir_image.png"); sys.exit(1)
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("could not read image"); sys.exit(1)
    res = run_demo(img)
    cv2.imwrite("demo_out.png", res["overlay_bgr"])
    print("\n" + res["report"])
    print("\nWrote demo_out.png")
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vein_pipeline import (make_synthetic_nir, assess_quality, detect_roi,
                           segment_veins_classical, extract_graph,
                           compute_segment_geometry, rank_candidates, visualize)

def main():
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
        if img is None:
            print("Could not read image"); return
    else:
        img = make_synthetic_nir()

    qc = assess_quality(img)
    roi = detect_roi(img)
    enhanced, vness, mask = segment_veins_classical(img, roi_mask=roi)
    graph = extract_graph(mask)
    feats = compute_segment_geometry(graph)
    ranked = rank_candidates(feats, img.shape, top_k=3)
    overlay = cv2.cvtColor(visualize(img, mask, graph, ranked), cv2.COLOR_BGR2RGB)

    skel_vis = cv2.cvtColor((mask * 60).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    sy, sx = np.where(graph["skeleton"]); skel_vis[sy, sx] = (255, 255, 0)
    by, bx = np.where(graph["branch_points"])
    for x, y in zip(bx, by):
        cv2.circle(skel_vis, (int(x), int(y)), 3, (255, 0, 0), -1)

    panels = [
        ("1. NIR input + QC", img, "gray"),
        ("2a. CLAHE enhanced", enhanced, "gray"),
        ("2b. Frangi vesselness", vness, "magma"),
        ("2c. Segmentation mask", mask, "gray"),
        ("3. Skeleton + branches", skel_vis, None),
        ("4+6. Ranked sites", overlay, None),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Camera Only NIR Vein Pipeline (Research Demo: NOT FOR CLINICAL USE)",
                 fontsize=15, fontweight="bold")
    for ax, (title, im, cmap) in zip(axes.ravel(), panels):
        ax.imshow(im, cmap=cmap) if cmap else ax.imshow(im)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    qc_txt = f"QC: {'PASS' if qc['passed'] else 'FAIL'}"
    if ranked:
        qc_txt += f"   |   top site score={ranked[0]['score']:.2f}"
    fig.text(0.5, 0.02, qc_txt, ha="center", fontsize=11)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig("pipeline_figure.png", dpi=130, bbox_inches="tight")
    print("wrote pipeline_figure.png")

if __name__ == "__main__":
    main()
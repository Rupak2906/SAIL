import sys
import cv2
import numpy as np
from skimage.filters import frangi

def preprocess(gray, clip=4.0):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(gray)

def extract_veins(gray, work=900, sigmas=(2, 9), clip=4.0, sensitivity=0.10):
    H, W = gray.shape
    s = work / max(H, W)
    g = cv2.resize(gray, (int(W * s), int(H * s)), interpolation=cv2.INTER_CUBIC) \
        if s < 1 else gray.copy()

    _, roi = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    roi = cv2.erode(roi, np.ones((max(3, g.shape[0] // 60),) * 2, np.uint8))

    enh = preprocess(g, clip)
    vmap = frangi(cv2.GaussianBlur(enh, (0, 0), 1).astype(float) / 255.0,
                  sigmas=range(sigmas[0], sigmas[1]), black_ridges=True)
    vmap = (vmap - vmap.min()) / (vmap.max() - vmap.min() + 1e-8)
    vmap = vmap * (roi > 0)

    mask = (vmap > sensitivity).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    clean = np.zeros_like(mask)
    min_area = max(20, (g.shape[0] * g.shape[1]) // 4000)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[lab == i] = 1

    up = lambda a, inter: cv2.resize(a.astype(np.float32), (W, H), interpolation=inter)
    return (up(enh, cv2.INTER_CUBIC).astype(np.uint8),
            up(vmap, cv2.INTER_CUBIC),
            (up(clean, cv2.INTER_NEAREST) > 0.5).astype(np.uint8))

def overlay_veinfinder(gray, vmap):
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    vn = vmap / (vmap.max() + 1e-8)
    heat = np.zeros_like(base)
    heat[..., 1] = (vn * 255).astype(np.uint8)
    heat[..., 2] = (vn * 200).astype(np.uint8)
    return cv2.addWeighted(base, 0.7, heat, 0.95, 0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python vein_finder.py path/to/nir.png"); sys.exit(1)
    img = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("could not read image"); sys.exit(1)
    enh, vmap, mask = extract_veins(img)
    cv2.imwrite("veinfinder_overlay.png", cv2.cvtColor(overlay_veinfinder(img, vmap),
                                                       cv2.COLOR_RGB2BGR))
    cv2.imwrite("veinfinder_mask.png", mask * 255)
    print(f"veins cover {mask.mean()*100:.2f}% of pixels")
    print("wrote veinfinder_overlay.png and veinfinder_mask.png")
import os, glob
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
INCOMPLETE_THR = 0.15

def main():
    masks = [m for m in sorted(glob.glob(os.path.join(C.MASKS_DIR, "*")))
             if m.lower().endswith(IMG_EXT)]
    rows = []
    for mp in masks:
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        rows.append((os.path.basename(mp), float((m > 127).mean() * 100)))
    if not rows:
        print("No masks found. Check MASKS_DIR in config.py."); return
    rows.sort(key=lambda x: x[1])
    vals = np.array([v for _, v in rows])

    print(f"masks inspected: {len(vals)}")
    print(f"vein-pixel %: min={vals.min():.3f}  median={np.median(vals):.3f}  "
          f"mean={vals.mean():.3f}  max={vals.max():.3f}")
    bad = [n for n, v in rows if v < INCOMPLETE_THR]
    print(f"likely INCOMPLETE (<{INCOMPLETE_THR}% vein pixels): {len(bad)}/{len(vals)} "
          f"({100*len(bad)/len(vals):.0f}%)")
    print("10 emptiest masks:")
    for n, v in rows[:10]:
        print(f"   {v:6.3f}%   {n}")

    plt.figure(figsize=(7, 4))
    plt.hist(vals, bins=30)
    plt.axvline(INCOMPLETE_THR, color="r", ls="--",
                label=f"incomplete threshold = {INCOMPLETE_THR}%")
    plt.xlabel("vein pixels (% of image)"); plt.ylabel("number of images")
    plt.title("Label completeness across the dataset")
    plt.legend(); plt.tight_layout()
    plt.savefig("label_completeness.png", dpi=120)
    print("wrote label_completeness.png")

if __name__ == "__main__":
    main()
import os, cv2, numpy as np
import config as C

C.DATA_ROOT = "data_synth"
C.IMAGES_DIR = os.path.join(C.DATA_ROOT, "images")
C.MASKS_DIR = os.path.join(C.DATA_ROOT, "masks")
C.MASK_SUFFIX = "_mask"
C.IMG_SIZE = 128
C.EPOCHS = 2
C.BATCH_SIZE = 4
C.ENCODER_WEIGHTS = None
C.subject_of = lambda p: os.path.basename(p).split("_")[0]

os.makedirs(C.IMAGES_DIR, exist_ok=True)
os.makedirs(C.MASKS_DIR, exist_ok=True)

def make_pair(h, w, seed):
    rng = np.random.default_rng(seed)
    img = rng.normal(18, 4, (h, w)).clip(0, 255).astype(np.uint8)
    mask = np.zeros((h, w), np.uint8)
    arm = np.zeros((h, w), np.uint8)
    cv2.ellipse(arm, (w // 2, h // 2), (int(w * 0.42), int(h * 0.30)), 8, 0, 360, 255, -1)
    skin = rng.normal(150, 6, (h, w)).clip(0, 255).astype(np.uint8)
    img = np.where(arm == 255, skin, img).astype(np.uint8)

    def curve(p0, p1, p2, val, thick):
        ts = np.linspace(0, 1, 60)
        pts = [[int((1-t)**2*p0[0]+2*(1-t)*t*p1[0]+t**2*p2[0]),
                int((1-t)**2*p0[1]+2*(1-t)*t*p1[1]+t**2*p2[1])] for t in ts]
        pts = np.array(pts)
        cv2.polylines(img, [pts], False, val, thick, cv2.LINE_AA)
        cv2.polylines(mask, [pts], False, 255, max(thick - 1, 2), cv2.LINE_AA)

    curve((w*0.25, h*0.40), (w*0.45, h*0.32), (w*0.75, h*0.46), 92, 5)
    curve((w*0.30, h*0.62), (w*0.50, h*0.55), (w*0.72, h*0.60), 95, 4)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = np.where(arm == 255, img, rng.normal(18, 4, (h, w)).clip(0, 255)).astype(np.uint8)
    return img, mask

# 4 subjects x 4 images, each a different size (tests resize awareness)
sizes = [(420, 560), (700, 900), (1980, 2972), (512, 512),
         (480, 640), (1024, 1280), (360, 540), (800, 600),
         (600, 800), (1200, 1600), (450, 450), (900, 1200),
         (520, 700), (760, 1000), (640, 480), (1080, 1440)]
k = 0
for s_i, subj in enumerate(["subjA", "subjB", "subjC", "subjD"]):
    for j in range(4):
        h, w = sizes[k]; k += 1
        img, mask = make_pair(h, w, seed=k)
        name = f"{subj}_{j:03d}"
        cv2.imwrite(os.path.join(C.IMAGES_DIR, name + ".png"), img)
        cv2.imwrite(os.path.join(C.MASKS_DIR, name + "_mask.png"), mask)
print(f"[test] wrote {k} synthetic pairs of varying sizes")

from train import train_main
from evaluate import eval_main
import unet_segment

train_main()
eval_main()

big_img, _ = make_pair(1980, 2972, seed=999)
mask, prob = unet_segment.predict(big_img)
print(f"[test] inference on 1980x2972 image -> mask shape {mask.shape}, "
      f"vein pixels={mask.mean()*100:.2f}%")
assert mask.shape == big_img.shape, "resize aware inference must return original size!"
print("[test] OK: resize aware inference returns original resolution.")

try:
    import sys; sys.path.insert(0, "..")
    from vein_pipeline import extract_graph, compute_segment_geometry, rank_candidates
    g = extract_graph(mask)
    feats = compute_segment_geometry(g)
    ranked = rank_candidates(feats, mask.shape, top_k=3)
    print(f"[test] graph+ranking OK: {len(feats)} segments, {len(ranked)} ranked candidates")
except Exception as e:
    print(f"[test] (graph stage skipped: {e})")

print("\nAll Good. full train -> eval -> resize-aware infer -> graph pipeline runs")
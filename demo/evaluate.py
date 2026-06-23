import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import config as C
from data import find_pairs, split_pairs, VeinDataset, eval_tf
from model import build_model

def metrics_at(model, loader, device, thr):
    tp = fp = fn = inter = union = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = (torch.sigmoid(model(x)) > thr).float()
            tp += (p * y).sum().item()
            fp += (p * (1 - y)).sum().item()
            fn += ((1 - p) * y).sum().item()
            inter += (p * y).sum().item()
            union += p.sum().item() + y.sum().item()
    dice = (2 * inter + 1) / (union + 1)
    iou = (inter + 1) / (union - inter + 1)
    precision = (tp + 1) / (tp + fp + 1)
    recall = (tp + 1) / (tp + fn + 1)
    return dict(dice=dice, iou=iou, precision=precision, recall=recall)

def save_qualitative(model, pairs, device, thr, n=4, out="eval_qualitative.png"):
    n = min(n, len(pairs))
    fig, axes = plt.subplots(n, 3, figsize=(11, 3.2 * n))
    if n == 1:
        axes = axes[None, :]
    tf = eval_tf(C.IMG_SIZE)
    for r in range(n):
        ip, mp = pairs[r]
        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        gt = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        x = tf(image=rgb, mask=gt.astype(np.float32))["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        prob = cv2.resize(prob, (img.shape[1], img.shape[0]))
        pred = (prob > thr).astype(np.uint8)
        for c, (im, t, cm) in enumerate([(img, "input", "gray"),
                                         (gt, "ground truth", "gray"),
                                         (pred, "prediction", "gray")]):
            axes[r, c].imshow(im, cmap=cm); axes[r, c].set_title(t); axes[r, c].axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    print(f"[eval] wrote {out}")

def eval_main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = find_pairs()
    _, va, te = split_pairs(pairs)
    val_dl = DataLoader(VeinDataset(va, eval_tf(C.IMG_SIZE)), batch_size=C.BATCH_SIZE)
    test_dl = DataLoader(VeinDataset(te, eval_tf(C.IMG_SIZE)), batch_size=C.BATCH_SIZE)

    model = build_model().to(device)
    model.load_state_dict(torch.load(C.CKPT, map_location=device))
    model.eval()

    best_thr, best_dice = 0.5, -1
    for thr in np.arange(0.20, 0.80, 0.05):
        d = metrics_at(model, val_dl, device, float(thr))["dice"]
        if d > best_dice:
            best_dice, best_thr = d, float(thr)
    with open(C.THRESH_FILE, "w") as f:
        f.write(str(round(best_thr, 3)))
    print(f"[eval] best threshold (val) = {best_thr:.2f}  val_dice={best_dice:.4f}")

    m = metrics_at(model, test_dl, device, best_thr)
    print(f"[eval] TEST  dice={m['dice']:.4f}  iou={m['iou']:.4f}  "
          f"precision={m['precision']:.4f}  recall={m['recall']:.4f}  @thr={best_thr:.2f}")

    save_qualitative(model, te, device, best_thr)
    return m

if __name__ == "__main__":
    eval_main()
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as C
from data import find_pairs, split_pairs, VeinDataset, train_tf, eval_tf
from model import build_model

def seed_everything(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1.0, bce_weight=0.2):
        super().__init__()
        self.a, self.b, self.g, self.s = alpha, beta, gamma, smooth
        self.bce = nn.BCEWithLogitsLoss()
        self.bw = bce_weight

    def forward(self, logits, targets):
        p = torch.sigmoid(logits).flatten(1)
        t = targets.flatten(1)
        tp = (p * t).sum(1)
        fp = (p * (1 - t)).sum(1)
        fn = ((1 - p) * t).sum(1)
        tversky = (tp + self.s) / (tp + self.a * fp + self.b * fn + self.s)
        focal = ((1 - tversky) ** self.g).mean()
        return focal + self.bw * self.bce(logits, targets)

@torch.no_grad()
def batch_metrics(logits, targets, thr=0.5):
    preds = (torch.sigmoid(logits) > thr).float().flatten(1)
    t = targets.flatten(1)
    inter = (preds * t).sum(1)
    union = preds.sum(1) + t.sum(1)
    dice = ((2 * inter + 1) / (union + 1)).mean().item()
    iou = ((inter + 1) / (union - inter + 1)).mean().item()
    return dice, iou

def evaluate(model, loader, device):
    model.eval(); d = i = n = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            dd, ii = batch_metrics(model(x), y)
            bs = x.size(0); d += dd * bs; i += ii * bs; n += bs
    return d / max(n, 1), i / max(n, 1)

def train_main():
    seed_everything(C.SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device} arch={C.ARCH} img_size={C.IMG_SIZE} loss=FocalTversky")

    pairs = find_pairs()
    if len(pairs) < 6:
        raise SystemExit("Need >= ~6 pairs. Check paths in config.py.")
    tr, va, te = split_pairs(pairs)

    train_dl = DataLoader(VeinDataset(tr, train_tf(C.IMG_SIZE)), batch_size=C.BATCH_SIZE,
                          shuffle=True, num_workers=2, pin_memory=(device == "cuda"))
    val_dl = DataLoader(VeinDataset(va, eval_tf(C.IMG_SIZE)), batch_size=C.BATCH_SIZE,
                        shuffle=False, num_workers=2)

    model = build_model().to(device)
    criterion = FocalTverskyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=C.EPOCHS)
    use_amp = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = 0.0
    for epoch in range(C.EPOCHS):
        model.train(); running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            running += loss.item() * x.size(0)
        scheduler.step()
        vd, vi = evaluate(model, val_dl, device)
        print(f"epoch {epoch+1:>3}/{C.EPOCHS}  loss={running/len(tr):.4f}  "
              f"val_dice={vd:.4f}  val_iou={vi:.4f}")
        if vd > best:
            best = vd
            torch.save(model.state_dict(), C.CKPT)
            print(f"    saved {C.CKPT} (val_dice={best:.4f})")
    print(f"[train] done. best val_dice={best:.4f}")

if __name__ == "__main__":
    train_main()
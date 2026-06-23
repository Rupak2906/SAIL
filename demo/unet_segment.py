import os
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config as C
from model import build_model
from data import preprocess_nir, MEAN, STD

_model = None
_device = None

def load_model(ckpt=None):
    global _model, _device
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = build_model().to(_device)
    _model.load_state_dict(torch.load(ckpt or C.CKPT, map_location=_device))
    _model.eval()
    return _model

def get_threshold(default=0.5):
    if os.path.exists(C.THRESH_FILE):
        try:
            return float(open(C.THRESH_FILE).read().strip())
        except Exception:
            pass
    return default

def predict(img_gray, thr=None, size=None):
    if _model is None:
        load_model()
    thr = get_threshold() if thr is None else thr
    size = size or C.IMG_SIZE
    H, W = img_gray.shape
    enhanced = preprocess_nir(img_gray)
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    tf = A.Compose([A.Resize(size, size),
                    A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0),
                    ToTensorV2()])
    x = tf(image=rgb)["image"].unsqueeze(0).to(_device)
    with torch.no_grad():
        prob = torch.sigmoid(_model(x))[0, 0].cpu().numpy()
    prob = cv2.resize(prob, (W, H), interpolation=cv2.INTER_LINEAR)
    return (prob > thr).astype(np.uint8), prob
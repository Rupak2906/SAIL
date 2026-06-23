import os
import glob
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from sklearn.model_selection import GroupShuffleSplit
import config as C

IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def preprocess_nir(img_gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(img_gray)

def find_pairs():
    imgs = [p for p in sorted(glob.glob(os.path.join(C.IMAGES_DIR, "*")))
            if p.lower().endswith(IMG_EXT)]
    pairs, missing = [], 0
    for ip in imgs:
        stem = os.path.splitext(os.path.basename(ip))[0]
        cands = glob.glob(os.path.join(C.MASKS_DIR, f"{stem}{C.MASK_SUFFIX}.*"))
        if cands:
            pairs.append((ip, cands[0]))
        else:
            missing += 1
    if missing:
        print(f"[data] WARNING: {missing} images had no matching mask and were skipped.")

    min_frac = getattr(C, "MIN_MASK_FRAC", 0.0)
    if min_frac and min_frac > 0:
        kept, dropped = [], 0
        for ip, mp in pairs:
            m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if m is not None and (m > 127).mean() * 100 >= min_frac:
                kept.append((ip, mp))
            else:
                dropped += 1
        print(f"[data] dropped {dropped} pairs with <{min_frac}% vein pixels "
              f"(likely incomplete labels).")
        pairs = kept

    print(f"[data] using {len(pairs)} image/mask pairs.")
    if not pairs:
        raise RuntimeError("No pairs found. Check IMAGES_DIR / MASKS_DIR / MASK_SUFFIX.")
    return pairs

def split_pairs(pairs):
    groups = np.array([C.subject_of(ip) for ip, _ in pairs])
    n_subjects = len(set(groups.tolist()))
    idx = np.arange(len(pairs))
    tv = C.VAL_FRAC + C.TEST_FRAC
    gss = GroupShuffleSplit(n_splits=1, test_size=tv, random_state=C.SEED)
    tr, tmp = next(gss.split(idx, groups=groups))
    tmp_groups = groups[tmp]
    if len(set(tmp_groups.tolist())) < 2:
        print("[data] WARNING: not enough subjects for clean val/test split.")
        half = len(tmp) // 2
        va, te = tmp[:half], tmp[half:]
    else:
        gss2 = GroupShuffleSplit(n_splits=1, test_size=C.TEST_FRAC / tv, random_state=C.SEED)
        va_rel, te_rel = next(gss2.split(tmp, groups=tmp_groups))
        va, te = tmp[va_rel], tmp[te_rel]
    pick = lambda ii: [pairs[i] for i in ii]
    print(f"[data] subjects={n_subjects} train={len(tr)} val={len(va)} test={len(te)}")
    return pick(tr), pick(va), pick(te)

def train_tf(size):
    return A.Compose([
        A.Resize(size, size, interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.RandomGamma(p=0.3),
        A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])

def eval_tf(size):
    return A.Compose([
        A.Resize(size, size, interpolation=cv2.INTER_LINEAR,
                 mask_interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=MEAN, std=STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])

class VeinDataset(Dataset):
    def __init__(self, pairs, transform):
        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, mp = self.pairs[i]
        img = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Could not read image: {ip}")
        img = preprocess_nir(img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read mask: {mp}")
        mask = (mask > 127).astype(np.uint8)

        H, W = mask.shape
        k = max(1, round(max(H, W) / C.IMG_SIZE))
        if k > 1:
            mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
        mask = mask.astype(np.float32)

        out = self.transform(image=img, mask=mask)
        return out["image"], out["mask"].unsqueeze(0).float()
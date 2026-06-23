import os

# DATA
DATA_ROOT = os.path.join("data", "Forearm_Veins_NIR")

IMAGES_DIR = os.path.join(DATA_ROOT, "forearmNIR")

MASKS_DIR = os.path.join(DATA_ROOT, "forearmNIR_masks")

MASK_SUFFIX = "_mask"

IMG_SIZE = 320

# MODEL
# Options: "unet", "unet_attention", "unetplusplus", "manet"
ARCH = "unet"
ENCODER = "resnet34"
ENCODER_WEIGHTS = "imagenet"

# TRAINING
BATCH_SIZE = 4
EPOCHS = 60
LR = 1e-4
WEIGHT_DECAY = 1e-4

VAL_FRAC = 0.15
TEST_FRAC = 0.15
SEED = 42
MIN_MASK_FRAC = 0.15

# OUTPUTS
CKPT = "best_vein_unet.pth"
THRESH_FILE = "threshold.txt"

def subject_of(path):
    """
    Subject IDs are not available.

    So each image is treated as its own subject. This is okay for a demo,
    but if multiple images are from the same person, test performance may
    be slightly optimistic.
    """
    return os.path.splitext(os.path.basename(path))[0]
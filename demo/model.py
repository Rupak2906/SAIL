import segmentation_models_pytorch as smp
import config as C

def build_model(arch=None):
    arch = (arch or C.ARCH).lower()
    common = dict(encoder_name=C.ENCODER, encoder_weights=C.ENCODER_WEIGHTS,
                  in_channels=3, classes=1, activation=None)
    if arch == "unet":
        return smp.Unet(**common)
    if arch == "unet_attention":
        return smp.Unet(decoder_attention_type="scse", **common)
    if arch == "unetplusplus":
        return smp.UnetPlusPlus(**common)
    if arch == "manet":
        return smp.MAnet(**common)
    raise ValueError(f"Unknown ARCH '{arch}'")
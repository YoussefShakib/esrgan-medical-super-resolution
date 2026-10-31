from .srcnn import SRCNN
from .srresnet import SRResNet
from .edsr import EDSR
from .esrgan import RRDBNet
from .discriminator import Discriminator


def build_model(model_name: str, scale: int):
    name = model_name.lower()
    if name == "srcnn":
        return SRCNN()
    if name == "srresnet":
        return SRResNet(scale=scale)
    if name == "edsr":
        return EDSR(scale=scale)
    if name == "esrgan":
        return RRDBNet(scale=scale)
    raise ValueError(f"Unsupported model: {model_name}")

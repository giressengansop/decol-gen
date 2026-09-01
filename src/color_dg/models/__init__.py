"""Neural network architectures"""

from .resnet import create_resnet18
from .vgg import create_vgg11

_FACTORY = {"resnet18": create_resnet18, "vgg11": create_vgg11}


def create_model(name="resnet18", **kwargs):
    if name not in _FACTORY:
        raise ValueError(f"Unknown model {name!r}; verfuegbar: {sorted(_FACTORY)}")
    return _FACTORY[name](**kwargs)


__all__ = ["create_resnet18", "create_vgg11", "create_model"]

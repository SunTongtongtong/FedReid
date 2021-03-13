from .resnet_SNR import *
__model_factory = {
    # image classification models
    'resnet50_SNR': resnet50_SNR,
}


def init_model(name, *args, **kwargs):
    if name not in list(__model_factory.keys()):
        raise KeyError("Unknown model: {}".format(name))
    return __model_factory[name](*args, **kwargs)

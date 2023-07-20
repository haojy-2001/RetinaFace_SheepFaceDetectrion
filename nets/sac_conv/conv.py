from torch import nn as nn
from .conv_ws import ConvWS2d, ConvAWS2d
from .saconv import SAConv2d
conv_cfg = {
    'Conv': nn.Conv2d,
    'ConvWS': ConvWS2d,
    'ConvAWS': ConvAWS2d,
    'SAC': SAConv2d,
    # TODO: octave conv
}

def build_conv_layer(cfg, *args, **kwargs):
    if cfg is None:
        cfg_ = dict(type='Conv')
    else:
        assert isinstance(cfg, dict) and 'type' in cfg
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in conv_cfg:
        raise KeyError('Unrecognized norm type {}'.format(layer_type))
    else:
        conv_layer = conv_cfg[layer_type]

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer

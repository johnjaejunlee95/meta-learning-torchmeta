from collections import OrderedDict
import torch
import torch.nn as nn
from torchmeta.modules import (MetaModule, MetaConv2d, MetaSequential, MetaBatchNorm2d)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def conv_block(in_channels, out_channels, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('norm', MetaBatchNorm2d(out_channels, momentum=1.,
                                track_running_stats=False)),
        ('relu', nn.ReLU(inplace=True)),
        ('pool', nn.MaxPool2d(2))
    ]))


def conv_drop_block(in_channels, out_channels, drop_p, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('norm', MetaBatchNorm2d(out_channels, momentum=1.,
                                track_running_stats=False)),
        ('relu', nn.ReLU(inplace=True)),
        ('dropout', nn.Dropout(drop_p)),
        ('pool', nn.MaxPool2d(2)),
    ]))


class Conv4Proto(MetaModule):
    def __init__(self, args):
        super(Conv4Proto, self).__init__()
        self.in_channels = args.imgc
        self.hidden_size = args.filter_size
        self.drop_p = args.drop_out

        kwargs = {}
        if self.drop_p > 0.:
            conv = conv_drop_block
            kwargs['drop_p'] = self.drop_p
        else:
            conv = conv_block

        self.features = MetaSequential(OrderedDict([
            ('layer1', conv(self.in_channels, self.hidden_size, kernel_size=3,
                            stride=1, padding=1, bias=True, **kwargs)),
            ('layer2', conv(self.hidden_size, self.hidden_size, kernel_size=3,
                            stride=1, padding=1, bias=True, **kwargs)),
            ('layer3', conv(self.hidden_size, self.hidden_size, kernel_size=3,
                            stride=1, padding=1, bias=True, **kwargs)),
            ('layer4', conv(self.hidden_size, self.hidden_size, kernel_size=3,
                            stride=1, padding=1, bias=True, **kwargs))
        ]))

    def forward(self, inputs, params=None):
        params_feature = self.get_subdict(params, 'features')
            
        features = self.features(inputs, params=params_feature)
        features = features.view(features.size(0), -1)

        return features



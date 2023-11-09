from collections import OrderedDict
import torch
import torch.nn as nn
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def meta_conv_block(in_channels, out_channels, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('norm', MetaBatchNorm2d(out_channels, momentum=1.,
                                track_running_stats=False)),
        ('relu', nn.ReLU(inplace=True)),
        ('pool', nn.MaxPool2d(2))
    ]))

def conv3x3(in_channels, out_channels, **kwargs):

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

def conv_drop_block(in_channels, out_channels, drop_p, **kwargs):
    return MetaSequential(OrderedDict([
        ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
        ('norm', MetaBatchNorm2d(out_channels, momentum=1.,
                                track_running_stats=False)),
        ('relu', nn.ReLU(inplace=True)),
        ('dropout', nn.Dropout(drop_p)),
        ('pool', nn.MaxPool2d(2)),
    ]))


class Conv_Model(MetaModule):
    def __init__(self, args):
        super(Conv_Model, self).__init__()
        
        self.in_channels = args.imgc
        self.out_features = args.num_ways
        self.hidden_size = args.filter_size
        if args.datasets == 'CIFAR_FS':
            self.feature_size = args.filter_size*2*2
        else:
            self.feature_size = args.filter_size*5*5
        self.drop_p = args.drop_out
            
        if args.update == 'anil':
            self.anil = True
            self.boil = False
        elif args.update == 'boil':
            self.anil = False
            self.boil = True
        else:
            self.anil = False
            self.boil = False
       

        kwargs = {}
        if self.drop_p > 0.:
            conv = conv_drop_block
            kwargs['drop_p'] = self.drop_p
            self.drop_classifer = nn.Identity()
        else:
            conv = meta_conv_block
            self.drop_classifer = nn.Identity()


        if args.update == 'anil':
            self.features = nn.Sequential(
            conv3x3(self.in_channels, self.hidden_size),
            conv3x3(self.hidden_size, self.hidden_size),
            conv3x3(self.hidden_size, self.hidden_size),
            conv3x3(self.hidden_size, self.hidden_size)
            )
            self.classifier = MetaLinear(self.feature_size, self.out_features, bias=True)
            
        elif args.update == 'boil':
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
            self.classifier = nn.Linear(self.feature_size, self.out_features, bias=True)
            
        else:
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
            self.classifier = MetaLinear(self.feature_size, self.out_features, bias=True)
        

    def forward(self, inputs, params=None):
        if self.anil:
            features = self.features(inputs)
        else:
            params_feature = self.get_subdict(params, 'features')
            features = self.features(inputs, params=params_feature)

        features = features.view((features.size(0), -1))
        
        if self.boil:
            logits = self.classifier(features)
        else:
            params_classifier = self.get_subdict(params, 'classifier')
            logits = self.classifier(features, params=params_classifier)
        
        
        logits = self.drop_classifer(logits)

        return logits
    
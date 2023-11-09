import torch

from .conv4 import Conv_Model
from .conv4proto import Conv4Proto
from .resnet import ResNet12
from .resnetproto import ResNetProto12

def model_selection(args, method):
    
    if method == 'maml': 
        if args.model == 'conv4':
            model = Conv_Model(args).cuda()
        elif args.model == 'resnet':
            model = ResNet12(args).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr,  weight_decay=1e-5)
    
    elif method == 'proto':
        if args.model == 'conv4':
            model = Conv4Proto(args).cuda()
        elif args.model == 'resnet':
            model = ResNetProto12(drop_p=args.drop_out).cuda()
            
        optimizer = torch.optim.Adam(model.parameters(), lr=args.meta_lr, weight_decay=1e-5)
    
    return model, optimizer
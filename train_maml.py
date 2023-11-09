import torch
import numpy as np
import warnings
import random

from method import maml
from model import model_selection
from utils.args import parse_args
from utils.datasets import get_meta_dataset
from torchmeta.utils.data import BatchMetaDataLoader
from utils.validation import Validation
from utils.utils import *

import warnings
import datetime as dt
from copy import deepcopy
warnings.filterwarnings(action="ignore", category=UserWarning)

args = parse_args()

SAVE_PATH = "/data2/jjlee_datasets/model_ckpt/single/" #write your own path
RANDOM_SEED = random.randint(0, 1000) #or specific seed number

x = dt.datetime.now()

name = '{}_{}way_{}shot_{}_model_{}_version_{}'.format(args.datasets, str(args.num_ways), str(args.num_shots), args.update, str(args.model), args.version)

myLogger = log_(name)
myLogger.info(name)
myLogger.info("Start Training!!")


def main(args):
    
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    
    
    args.best_loss = 100
    args.best_acc = 0
    
    kwargs = {'shuffle': True, 'pin_memory': True, 'num_workers': 8}
    
    train_set, val_set = get_meta_dataset(args, dataset=args.datasets)
    model, optimizer = model_selection(args, 'maml')
    
    meta = maml.MAML(model, optimizer)
   
    train_loader = cycle(BatchMetaDataLoader(train_set, batch_size=args.batch_size, **kwargs))
    val_loader = BatchMetaDataLoader(val_set, batch_size=1, **kwargs)
    
    validation_acc = Validation(meta, val_loader, model)
    avg_dict = {'acc': Average(), 'loss': Average()}
    
    for epoch in range(args.epoch):
       
        batch = next(train_loader)
        acc, loss, model = meta.forward(args, batch)

        avg_dict['acc'].add(acc)
        avg_dict['loss'].add(loss.item())

        if (epoch+1) % 500 == 0:
    
            log_acc(args, avg_dict, epoch, 'training', myLogger)
            myLogger.info("====================================================")
            
        if (epoch + 1) % 2500 == 0:
            validation_acc.validation(args, epoch, myLogger)    

        torch.save({
            'model': deepcopy(model).to('cpu'), #add more (e.g., optimizer.state_dict())
            #'optimizer': (optimizer.state_dict()),
            }, SAVE_PATH + "MAML_5-{}_{}_{}_{}_{}.pt".format(str(args.num_shots), args.datasets, args.update, args.model, args.version))
        
if __name__ == "__main__":
    
    arg = parse_args()
    torch.cuda.set_device(arg.gpu_id)
    main(arg)
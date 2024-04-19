import torch
import torch.nn.functional as F
import os
import logging


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).float().mean().item()

def log_acc(args, avg_dict, epoch, is_train, logger):
    
    check = is_train

    logger.info('{} -> updated steps:{}'.format(check, (epoch+1)))    

    acc_ = 100 * avg_dict["acc"].item()
    loss_ = avg_dict["loss"].item()
    logger.info("Datasets: {}".format(args.datasets))
    logger.info("Average acc = {:.4f}".format(acc_))
    logger.info("Average Loss = {:.4f}".format(loss_))
    
    if check != 'training':
        return acc_ , loss_
    else:
        pass
    
def cycle(loader):
    while True:
        for x in loader:
            yield x
            
class Average():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v

def log_(name):
    
    if os.path.exists("./log/{}.log".format(name)) == True:
        os.remove("./log/{}.log".format(name))
    mylogger = logging.getLogger("my")
    mylogger.setLevel(logging.INFO)

     
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler("./log/{}.log".format(name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    
    mylogger.addHandler(stream_handler)
    mylogger.addHandler(file_handler)
    
    return mylogger


def euclidean_distance(a, b):
    n = a.shape[0]
    m = b.shape[0]
    
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)

    logits = -((a - b)**2).sum(dim=2)
    return logits

    
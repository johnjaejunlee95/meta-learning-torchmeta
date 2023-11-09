import torch
import numpy as np
import warnings
import random

from utils.args import parse_args
from utils.datasets import get_meta_dataset
from torchmeta.utils.data import BatchMetaDataLoader
from method.test import Meta_Test
warnings.filterwarnings(action="ignore", category=UserWarning)

def eval_proto(args, dataloader, model, num_test=1):
    
    acc_list = []
    loss_list = []
    
    test = Meta_Test(model)

    for _ in range(num_test):
        RANDOM_SEED = random.randint(0, 1000)
    
        torch.manual_seed(RANDOM_SEED)
        torch.cuda.manual_seed_all(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        
        
        test_acc, test_loss = test.proto_test(args, dataloader)
        
        acc_list.append(test_acc)
        loss_list.append(test_loss)
        
    
    avg_acc = np.array(acc_list).mean(axis=0)
    avg_loss = np.array(loss_list).mean(axis=0)

    acc_stds = np.std(np.array(acc_list), 0)
    acc_ci95 = 1.96*acc_stds/np.sqrt(len(acc_list))
   
    loss_stds = np.std(np.array(loss_list), 0)
    loss_ci95 = 1.96*loss_stds/np.sqrt(len(loss_list))
    
    print("{} with ci95: Average Accuracy -> {:.2f} +- {:.2f} Average Loss -> {:.2f} +- {:.2f}".format(args.datasets, avg_acc, acc_ci95, avg_loss, loss_ci95))
    
    print("{}: {}-ways {}-shots".format(args.datasets, args.num_ways, args.num_shots))
    print("Average Accuracy with ci95 -> {:.2f} +- {:.2f}".format(avg_acc, acc_ci95))
    print("Average Loss with ci95 -> {:.2f} +- {:.2f}".format(avg_loss, loss_ci95))
    
    return avg_acc, avg_loss


if __name__ == "__main__":
    
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    SAVE_PATH = "/data2/jjlee_datasets/model_ckpt/single/" #write your own path
    
    checkpoint = torch.load(SAVE_PATH +  "PROTO_5-{}_{}_{}_{}_{}_best.pt".format(str(args.num_shots), args.datasets,args.update, args.model, args.version))
    
    kwargs = {'shuffle': True, 'pin_memory': True, 'num_workers': 8}
    
    test_sets = get_meta_dataset(args, dataset=args.datasets, only_test=True)
    test_loader = BatchMetaDataLoader(test_sets, batch_size=1, **kwargs)
    eval_proto(args, test_loader, checkpoint['model'], num_test=3)

        
    

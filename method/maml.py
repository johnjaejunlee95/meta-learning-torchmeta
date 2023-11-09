import torch
import warnings

from copy import deepcopy
from torch import nn
from utils.utils import count_acc, Average, log_acc
from .gradient_utils import Finetuning
warnings.filterwarnings("ignore", category=UserWarning)

class MAML(nn.Module):
    def __init__(self, network, optimizer):

        super(MAML, self).__init__()

        self.network = network
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.finetuning = Finetuning()
       
    def forward(self, args, batch):

        acc = 0.
        loss = torch.tensor(0.,).cuda()
        
        for task_idx, (train_inputs, train_targets, test_inputs, test_targets) in enumerate(zip(*batch['train'], *batch['test'])):
            
            train_inputs = train_inputs.cuda()
            train_targets = train_targets.cuda()
            test_inputs = test_inputs.cuda()
            test_targets = test_targets.cuda()
        
            self.network = self.network.cuda()
        
            params, _ = self.finetuning.maml_inner_adapt(self.network, train_inputs, train_targets, args.update_lr, args.update_step)
            logits = self.network(test_inputs, params=params)
            
            loss_test = self.loss(logits, test_targets)
            loss += (loss_test.clone())/args.batch_size
            acc += count_acc(logits, test_targets)/args.batch_size

    
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
        return acc, loss_test, self.network

    
    def validation(self, args, epoch, loader, logger):
        acc = 0.        
        avg_dict = {'acc': Average(), 'loss': Average()}

        network = deepcopy(self.network).cuda()
        
        for n, batch in enumerate(loader):
            if n >= args.max_test_task:
                break
            
            train_inputs, train_targets = batch['train']
            test_inputs, test_targets = batch['test']
            
            train_inputs = train_inputs.cuda().squeeze()
            train_targets = train_targets.cuda().squeeze()
            test_inputs = test_inputs.cuda().squeeze()
            test_targets = test_targets.cuda().squeeze()
                
            params, _ = self.finetuning.maml_inner_adapt(network, train_inputs, train_targets, args.update_lr, args.update_step_test, first_order=True)

            with torch.no_grad():
                outputs_test = network(test_inputs, params)   
                loss_ = self.loss(outputs_test, test_targets)
                acc = count_acc(outputs_test, test_targets)                        

            avg_dict["loss"].add(loss_.item())
            avg_dict["acc"].add(acc)
      
        avg_acc, avg_loss = log_acc(args, avg_dict, epoch, 'Validation', logger)
        logger.info("====================================================")
                    
        return avg_acc, avg_loss
    
    
def main():
    pass

if __name__ == "__main__":
    main()

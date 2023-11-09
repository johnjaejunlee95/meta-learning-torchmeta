import torch
import warnings

from copy import deepcopy
from torch import nn
from utils.utils import * #Average, log_acc, count_acc, euclidean_distance

warnings.filterwarnings("ignore", category=UserWarning)

class ProtoNet(nn.Module):
    def __init__(self, network, optimizer):

        super(ProtoNet, self).__init__()

        self.network = network
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optimizer
       
    def forward(self, args, loader):
        
        acc = 0.
        loss = 0.
       
        for n, batch in enumerate(loader):
            if n >= args.batch_size:
                break
            
            train_inputs, train_targets = batch['train']
            test_inputs, test_targets = batch['test']
            
            train_inputs = train_inputs.cuda().squeeze()
            train_targets = train_targets.cuda().squeeze()
            test_inputs = test_inputs.cuda().squeeze()
            test_targets = test_targets.cuda().squeeze()
            
            train_targets = train_targets.sort()[0]
            test_targets = test_targets.sort()[0]
        
            self.network = self.network.cuda()
            self.network.train()
            
            train_embeddings = self.network(train_inputs)
            test_embeddings = self.network(test_inputs)
            
            proto = train_embeddings.reshape(args.num_ways_proto, args.num_shots, -1).mean(dim=1)
            logits = euclidean_distance(test_embeddings, proto)
            
            loss_test = self.loss(logits, test_targets)
            
            self.optimizer.zero_grad()
            loss_test.backward()
            self.optimizer.step()
            
            acc += count_acc(logits, test_targets)/args.batch_size
            loss += loss_test.clone().cpu()/args.batch_size
            
        return acc, loss, self.network
        
    
    def validation(self, args, epoch, loader, logger):
        avg_dict = {'acc': Average(), 'loss': Average()}

        network = deepcopy(self.network).cuda()
        network.eval()
        
        for n, batch in enumerate(loader):
            if n >= args.max_test_task:
                break
            
            train_inputs, train_targets = batch['train']
            test_inputs, test_targets = batch['test']
            
            train_inputs = train_inputs.cuda().squeeze()
            train_targets = train_targets.cuda().squeeze()
            test_inputs = test_inputs.cuda().squeeze()
            test_targets = test_targets.cuda().squeeze()
            
            train_targets = train_targets.sort()[0]
            test_targets = test_targets.sort()[0]
            
            with torch.no_grad():

                train_embeddings = network(train_inputs)
                test_embeddings = network(test_inputs)
                
                proto = train_embeddings.reshape(args.num_ways, args.num_shots, -1).mean(dim=1)
                logits = euclidean_distance(test_embeddings, proto)
                            
                loss_= self.loss(logits, test_targets)
                acc = count_acc(logits, test_targets)

            avg_dict["loss"].add(loss_.item())
            avg_dict["acc"].add(acc)
        
        del network
        torch.cuda.empty_cache()
        
        avg_acc, avg_loss = log_acc(args, avg_dict, epoch, 'Validation', logger)
        logger.info("====================================================")
    
        return avg_acc, avg_loss

def main():
    pass

if __name__ == "__main__":
    main()
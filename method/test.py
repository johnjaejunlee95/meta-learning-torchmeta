import torch
import warnings

from copy import deepcopy
from torch import nn
from torch.nn import functional as F 
from utils.utils import count_acc, Average, euclidean_distance
from .gradient_utils import Finetuning
warnings.filterwarnings("ignore", category=UserWarning)



class Meta_Test(nn.Module):
    def __init__(self, model_parameters):

        super(Meta_Test, self).__init__()

        self.loss = nn.CrossEntropyLoss()
        self.finetuning = Finetuning()
        self.network = model_parameters
        
    def maml_test(self, args, loader):
        
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
                
        avg_acc = 100 * avg_dict["acc"].item()
        avg_loss = avg_dict["loss"].item()
        print("Average acc = {:.4f}".format(avg_acc))
        print("Average Loss = {:.4f}".format(avg_loss))

        return avg_acc, avg_loss
    
    def proto_test(self, args, loader):
        
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
            
            train_targets = train_targets.sort()[0]
            test_targets = test_targets.sort()[0]
            
            with torch.no_grad():

                train_embeddings = network(train_inputs)
                test_embeddings = network(test_inputs)
                
                proto = train_embeddings.reshape(args.num_ways, args.num_shots, -1).mean(dim=1)
                if args.classifier == 'euclidean':
                    logits = euclidean_distance(test_embeddings, proto)
                    loss_= self.loss(logits, test_targets)
                elif args.classifier == 'cosine':
                    logits = F.cosine_similarity(test_embeddings.unsqueeze(2), proto.t().unsqueeze(0))
                    loss_ = self.loss(logits*args.scale_factor, test_targets)

                acc = count_acc(logits, test_targets)

            avg_dict["loss"].add(loss_.item())
            avg_dict["acc"].add(acc)
        
        avg_acc = 100 * avg_dict["acc"].item()
        avg_loss = avg_dict["loss"].item()
        print("Average acc = {:.4f}".format(avg_acc))
        print("Average Loss = {:.4f}".format(avg_loss))

        return avg_acc, avg_loss
    
    
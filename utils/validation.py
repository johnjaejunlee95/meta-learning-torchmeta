import torch
from copy import deepcopy


class Validation():
    def __init__(self, maml, dataloader, model):
        super(Validation, self).__init__()
        
        self.maml = maml
        self.dataloader = dataloader
        self.model = model

    def validation(self, args, epoch, logger, method='maml'):
        
     
        SAVE_PATH = "/data2/jjlee_datasets/model_ckpt/single/"
        
        acc, loss = self.maml.validation(args, epoch, self.dataloader, logger)
        if args.best_acc <= acc:
            args.best_acc = acc
            args.best_loss = loss
            
            if method == 'maml':
                torch.save({'model': deepcopy(self.model).to('cpu')}, # add more parameters
                            SAVE_PATH + "MAML_5-{}_{}_{}_{}_{}_best.pt".format(str(args.num_shots), args.datasets,args.update, args.model, args.version))
            else:
                torch.save({'model': deepcopy(self.model).to('cpu')}, # add more parameters
                            SAVE_PATH + "PROTO_5-{}_{}_{}_{}_{}_best.pt".format(str(args.num_shots), args.datasets,args.update, args.model, args.version))
        logger.info("Current Acc: {:.2f}, Best Acc: {:.2f}".format(acc, args.best_acc))
        logger.info("====================================================")
            

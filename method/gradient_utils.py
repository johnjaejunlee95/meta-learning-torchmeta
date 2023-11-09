import torch 
import torch.nn as nn
from torchmeta.modules import MetaModule
from collections import OrderedDict

class Finetuning:
    def __init__(self):
        self.loss = nn.CrossEntropyLoss()

    def maml_inner_adapt(self, model, inputs, targets, step_size, num_steps, params=None, first_order=False):
        for _ in range(num_steps):
            outputs_train = model(inputs, params=params)

            loss = self.loss(outputs_train, targets)
            model.zero_grad()
            params = self.gradient_update_parameters(model, loss, params=params, step_size=step_size, first_order=first_order)
        return params, loss


    def gradient_update_parameters(self, model, loss, params=None, step_size=0.5, first_order=False):
        if not isinstance(model, MetaModule):
            raise ValueError('The model must be an instance of `torchmeta.modules.'
                             'MetaModule`, got `{0}`'.format(type(model)))

        if params is None:
            params = OrderedDict(model.meta_named_parameters())
        grads = torch.autograd.grad(loss, 
                                    params.values(), 
                                    create_graph=not first_order, 
                                    retain_graph=True, 
                                    allow_unused=True)

        updated_params = OrderedDict()

        for (name, param), grad in zip(params.items(), grads):
            if grad is None:
                grad = 0.0
            updated_params[name] = param - step_size * grad

        return updated_params

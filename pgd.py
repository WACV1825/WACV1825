import torch
import torch.nn as nn
import torch.nn.functional as F

class PGD_Linf():

    def __init__(self, model, epsilon=8*4/255, step_size=4/255, num_steps=10, random_start=True, target_mode= False, bn_mode='eval', train=True, vat=False):

        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.random_start = random_start
        self.target_mode = target_mode
        self.bn_mode = bn_mode
        self.train = train
        self.criterion_ce = nn.CrossEntropyLoss()
        self.vat = vat

    def perturb(self, x_nat, targets=None, num_classes=None):
        if self.bn_mode == 'eval':
            self.model.eval()
            
        if self.random_start:
            x_adv = x_nat.detach() + torch.empty_like(x_nat).uniform_(-self.epsilon, self.epsilon).cuda().detach()
            x_adv = torch.clamp(x_adv, min=0, max=1)
        else:
            x_adv = x_nat.clone().detach()

        for _ in range(self.num_steps):
            x_adv.requires_grad_()
            outputs = self.model(x_adv)
            loss = self.criterion_ce(outputs, targets)
            loss.backward()
            grad = x_adv.grad

            if self.target_mode:
                x_adv = x_adv - self.step_size * grad.sign()
            else:
                x_adv = x_adv + self.step_size * grad.sign()
            
            x_adv = torch.min(torch.max(x_adv, x_nat - self.epsilon), x_nat + self.epsilon)
            x_adv = torch.clamp(x_adv, min=0, max=1).detach()
            d_adv = torch.clamp(x_adv - x_nat, min=-self.epsilon, max=self.epsilon).detach()
            
        if self.train:
            self.model.train()
        
        return x_adv, d_adv

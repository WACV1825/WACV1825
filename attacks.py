import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


def PGD_for_testing(model,
                    X,
                    y,
                    epsilon=0.031,
                    perturb_steps=20):
    assert isinstance(epsilon, float)
    step_size = epsilon / 10

    model.eval()
    X_pgd = Variable(X.data, requires_grad=True)
    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).cuda().detach()
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(perturb_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    model.zero_grad()
    X_pgd = Variable(X_pgd, requires_grad=False)
    X_pgd = X_pgd.detach()
    return X_pgd


def PGD_for_training(model,
                x_natural,
                y,
                epsilon=0.031,
                perturb_steps=10,
                inner='kl',
                distance='l_inf'):
    criterion_kl = nn.KLDivLoss(size_average=False)
    criterion_ce = nn.CrossEntropyLoss()
    model.eval()

    if torch.is_tensor(epsilon):
        epsilon = epsilon.view(-1, 1, 1, 1).cuda()
        step_size = epsilon/4
        step_size = step_size.cuda()
    else:
        step_size = epsilon/4

    x_adv = x_natural.detach() + torch.empty_like(x_natural).uniform_(-epsilon, epsilon).cuda().detach()

    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()

    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                if inner in ['kl']:
                    loss_inner = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                        torch.clamp(F.softmax(model(x_natural), dim=1), min=1e-8))
                    grad = torch.autograd.grad(loss_inner, [x_adv])[0]
                elif inner in ['ce']:
                    loss_inner = criterion_ce(model(x_adv), y)
                    loss_inner.backward()
                    grad = x_adv.grad
                else:
                    raise ValueError('not supported loss for inner maximization')
                
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    x_adv = x_adv.detach()
    return x_adv



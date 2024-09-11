from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

from models.wideresnet import *
from models.resnet import *
from attacks import PGD_for_testing, PGD_for_training
from dataset import SemiSupervisedDataset
from torch.utils.data import DataLoader, Subset
from utils import logger, AverageMeter

import wandb

parser = argparse.ArgumentParser(description='PyTorch Semi-Supervised Adversarial Training (Interpolated)')

parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10 or svhn')
parser.add_argument('--num-classes', type=int, default=10, metavar='N', help='number of classes')

parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--weight-decay', '--wd', default=2e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', type=float, default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=10,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.007,
                    help='perturb step size')
parser.add_argument('--lambd', type=float, default=6.0,
                    help='weight for consistence loss')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model-dir', default='model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--tau', type=float, default=2, help='temperature')

parser.add_argument('--rho-init', type=float, default=0.05, help='initial rho')
parser.add_argument('--model', default='wideresnet', 
                    help='model architecture resnet or wideresnet')
parser.add_argument('--depth', default=28, type=int, help='hyperparameter depth for WideResNet')
parser.add_argument('--widen-factor', default=5, type=int, help='hyperparameter widen_factor for WideResNet')

parser.add_argument('--rho-setup', type=int, default=0, help='0: rho = [0.05, 0.1] increased at epoch 75; \
                                                              1: rho = 0.05')

parser.add_argument('--ges', help='Global epsilon scheduling, e.g., `const`, `linear-70`, `curious-1.25-70`')
parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')
parser.add_argument('--num-labels-per-class', type=int, default=400, help='number of labeled data points per class')
parser.add_argument('--no-wandb', action='store_false', dest='wandb', help='disable wandb')
parser.add_argument('--wandb-entity', type=str, help='wandb entity')
parser.add_argument('--wandb-project', type=str, help='wandb project')
parser.add_argument('--loss', default='trades', help='loss function for outer minimization')
parser.add_argument('--loss-inner', default='kl', help='loss function for inner maximization')
parser.add_argument('--init', type=str, default='random', help='init model with `random` weights or `pretrained` model')
parser.add_argument('--smooth-label', type=str, default='none', help='label smoothing: ls or temp or none')
parser.add_argument('--mixed', action='store_true', default=False, help='fuse intp and adv')
parser.add_argument('--mixed-beta', type=float, default=0.5, help='')
parser.add_argument('--intp-steps', type=int, default=4, help='steps for binary search')
parser.add_argument('--outer', default='rst', help='loss function for outer minimization: rst, uat')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

dict_teacher_dir = {
    'cifar10': '/path/to/fixmatch/teacher/model/cifar10/best.pth.tar',
    'svhn': '/path/to/fixmatch/teacher/model/svhn/best.pth.tar',
    'cifar100': '/path/to/fixmatch/teacher/model/cifar100/best.pth.tar',
}

dict_sampled_label_idx = {
    'cifar10': '/path/to/fixmatch/teacher/model/cifar10/sampled_label_idx_4000.npy',
    'svhn': '/path/to/fixmatch/teacher/model/svhn/sampled_label_idx_1000.npy',
    'cifar100': '/path/to/fixmatch/teacher/model/cifar100/sampled_label_idx_4000.npy',
}

teacher_dir = dict_teacher_dir[args.dataset]
sampled_label_idx = dict_sampled_label_idx[args.dataset]

# settings
model_dir = os.path.join('checkpoints', args.model_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

log = logger(path=model_dir)
log.info(str(args))

if args.wandb:
    wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.model_dir)
    log.info('wandb name: {}'.format(args.model_dir))

# setup data loader
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])


####### Prepare Labeled and Unlabeled Training Set #######

if args.dataset in ['cifar10']:
    full_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
elif args.dataset in ['svhn']:
    full_dataset = datasets.SVHN(root='./data', split='train', download=True)
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
elif args.dataset in ['cifar100']:
    full_dataset = datasets.CIFAR100(root='./data', train=True, download=True)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
else:
    raise ValueError('dataset must be cifar10, cifar100, or svhn')

if args.dataset in ['cifar10', 'svhn']:
    num_classes = 10
elif args.dataset in ['cifar100']:
    num_classes = 100
else:
    raise ValueError('dataset must be cifar10, cifar100, or svhn')

labeled_indices = np.load(sampled_label_idx)
unlabeled_indices = np.setdiff1d(np.arange(len(full_dataset)), labeled_indices)
if num_classes * args.num_labels_per_class != len(labeled_indices):
    log.info('Warning: argument --num-labels-per-class is overridden by --sampled-labels-idx')

if args.dataset in ['cifar10', 'cifar100']:
    targets = torch.tensor(full_dataset.targets)
elif args.dataset in ['svhn']:
    targets = torch.tensor(full_dataset.labels)

# Create the semi-supervised dataset
trainset = SemiSupervisedDataset(full_dataset, labeled_indices, unlabeled_indices, name=args.dataset, transform=transform_train, transform_test=transform_test)
trainset.targets = torch.eye(num_classes)[targets]
trainset.targets[unlabeled_indices] = -1.0

labeled_subset = Subset(trainset, labeled_indices)
unlabeled_subset = Subset(trainset, unlabeled_indices)

log.info("Size of labeled data: {}".format(len(labeled_subset)))
log.info("Size of unlabeled data: {}".format(len(unlabeled_subset)))

labeled_dataloader = DataLoader(labeled_subset, batch_size=64, shuffle=True, num_workers=2)
unlabeled_dataloader = DataLoader(unlabeled_subset, batch_size=64, shuffle=False, num_workers=2)
train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

criterion = nn.CrossEntropyLoss()

rho = args.rho_init #0.05

parts = args.ges.split('-')
ges = {'mode': parts[0]}
if ges['mode'] == 'const':
    pass
elif ges['mode'] == 'linear':
    ges['t'] = int(parts[-1])
elif ges['mode'] == 'curious':
    ges['gamma'], ges['t'] = float(parts[1]), int(parts[-1])
elif ges['mode'] == 'curious_smooth':
    ges['gamma'], ges['t'] = float(parts[1]), int(parts[-1])
else:
    raise ValueError('not supported ges')

max_eps = args.epsilon
wandb_dict = {}

def train(args, model, model_c, device, train_loader, optimizer, epoch):
    model.train()
    log.info('current rho: {}'.format(rho))
    log.info('current max_epsilon: {}'.format(max_eps))

    if args.wandb:
        global wandb_dict
        wandb_dict['epoch'] = epoch
        wandb_dict['rho'] = rho
        wandb_dict['max_eps'] = max_eps

    losses_model = AverageMeter()
    losses_sup = AverageMeter()
    losses_rob = AverageMeter()
    losses_model.reset()
    losses_sup.reset()
    losses_rob.reset()

    # Assign pseudo-labels
    if epoch == 1:
        trainset.train = False
        if not args.smooth_label in ['none']:
            import re
            match = re.match(r"(ls|temp)(\d+\.?\d*)", args.smooth_label)
            if not match:
                raise argparse.ArgumentTypeError(f"Invalid argument --smooth-label")
            alpha = float(match.group(2))
            print(match.group(1), alpha)

        for batch_idx, (data, _, indices) in enumerate(unlabeled_dataloader):
            data = data.to(device)
            model_c.eval()
            out = model_c(data).detach()

            if 'ls' in args.smooth_label:
                pseudo = torch.softmax(out, dim=1).cpu()
                pseudo = pseudo * (1-alpha) + alpha / args.num_classes
            elif 'temp' in args.smooth_label:
                pseudo = torch.softmax(out/alpha, dim=1).cpu()
            elif 'none' in args.smooth_label:
                pseudo = torch.softmax(out, dim=1).cpu()
            else:
                raise argparse.ArgumentTypeError(f"Invalid argument --smooth-label")

            trainset.update_pseudo_labels(indices, pseudo)

 
    trainset.train = True
    for batch_idx, (data, target, indices) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        model.eval()
        optimizer.zero_grad()
        # calculate robust loss

        x_adv = PGD_for_training(model=model,
                                x_natural=data,
                                y=target,
                                epsilon=max_eps,
                                perturb_steps=args.num_steps,
                                inner=args.loss_inner)

        # Do interpolation
        bs = data.size(0)
        alpha_l = torch.zeros(bs, device=device)
        alpha_r = torch.ones(bs, device=device)
        for interp_step in range(args.intp_steps):
            alpha = (alpha_l + alpha_r) / 2
            x = alpha.view(bs, 1, 1, 1) * x_adv + (1-alpha.view(bs, 1, 1, 1)) * data
            x = x.detach()

            model.eval()
            out = model(x).detach()
            model.train()

            softmax_with_temp = F.softmax(out / args.tau, dim=1)
            max_class_scores, _ = torch.max(softmax_with_temp, dim=1)  # [batch]
            margin = -(softmax_with_temp * target).sum(dim=1) + max_class_scores # [batch]

            mask = margin < rho
            alpha_l[mask] = alpha[mask]
            alpha_r[~mask] = alpha[~mask]

        model.train()
        x_intp = alpha_r.view(bs, 1, 1, 1) * x_adv + (1-alpha_r.view(bs, 1, 1, 1)) * data
        x_intp = x_intp.detach()
        optimizer.zero_grad()

        if args.outer in ['rst']:
            logits = model(data)
            loss_sup = F.cross_entropy(logits, target)
            if args.mixed:
                loss_robust_intp = (1.0 / bs) * nn.KLDivLoss(size_average=False)(F.log_softmax(model(x_intp), dim=1),
                                                            torch.clamp(F.softmax(logits, dim=1), min=1e-8))
                loss_robust_adv = (1.0 / bs) * nn.KLDivLoss(size_average=False)(F.log_softmax(model(x_adv), dim=1),
                                                            torch.clamp(F.softmax(logits, dim=1), min=1e-8))        
                loss_robust = args.mixed_beta * loss_robust_intp + (1 - args.mixed_beta) * loss_robust_adv
            else:
                loss_robust = (1.0 / bs) * nn.KLDivLoss(size_average=False)(F.log_softmax(model(x_intp), dim=1),
                                                            torch.clamp(F.softmax(logits, dim=1), min=1e-8))
            loss = loss_sup + args.lambd * loss_robust

        else:
            raise ValueError('args.outer must be rst')

        loss.backward()
        optimizer.step()
        losses_model.update(float(loss.detach().cpu()), data.shape[0])
        losses_sup.update(float(loss_sup.detach().cpu()), data.shape[0])
        losses_rob.update(float(loss_robust.detach().cpu()), data.shape[0])

        model.eval()

        # print progress
        if True:
            log.info('Train Epoch: {} [{}/{} ({:.0f}%)]\t'
                     'Loss: {loss_model.val:.4f} ({loss_model.avg:.4f})\t'
                     'Loss_sup: {loss_sup.val:.4f} ({loss_sup.avg:.4f})\t'
                     'Loss_rob: {loss_rob.val:.4f} ({loss_rob.avg:.4f})\t'
                     .format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss_model=losses_model,
                        loss_sup=losses_sup, loss_rob=losses_rob))
            
    if args.wandb:
        wandb_dict['loss_model'] = losses_model.avg
        wandb_dict['loss_sup'] = losses_sup.avg
        wandb_dict['loss_robust'] = losses_rob.avg
        

def eval_test(model, device, test_loader, repeat=1):
    model.eval()

    meter_test_loss = AverageMeter()
    meter_test_adv_loss = AverageMeter()
    meter_correct = AverageMeter()
    meter_robust_correct = AverageMeter()

    for _ in range(repeat):
        test_loss = 0
        test_adv_loss = 0
        correct = 0
        robust_correct = 0

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
                test_loss += F.cross_entropy(output, target, size_average=False).item()
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()
            data_adv = PGD_for_testing(model=model,
                                        X=data,
                                        y=target,
                                        epsilon=8/255)
            with torch.no_grad():
                output = model(data_adv)
                test_adv_loss += F.cross_entropy(output, target, size_average=False).item()
                pred = output.max(1, keepdim=True)[1]
                robust_correct += pred.eq(target.view_as(pred)).sum().item()
                
        test_loss /= len(test_loader.dataset)
        test_adv_loss /= len(test_loader.dataset)

        meter_test_loss.update(test_loss)
        meter_test_adv_loss.update(test_adv_loss)
        meter_correct.update(correct)
        meter_robust_correct.update(robust_correct)

    test_loss = meter_test_loss.avg
    test_adv_loss = meter_test_adv_loss.avg
    correct = meter_correct.avg
    robust_correct = meter_robust_correct.avg

    log.info('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Average adv loss: {:.4f}, Robust Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, 
            correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset),
            test_adv_loss,
            robust_correct, len(test_loader.dataset), 100. * robust_correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    robust_accuracy = robust_correct / len(test_loader.dataset)

    if args.wandb:
        global wandb_dict
        wandb_dict['eval_test_loss'] = test_loss
        wandb_dict['eval_test_adv_loss'] = test_adv_loss
        wandb_dict['eval_test_accuracy'] = test_accuracy
        wandb_dict['eval_test_robust_accuracy'] = robust_accuracy

    return test_loss, test_adv_loss, test_accuracy, robust_accuracy



def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    if epoch >= 60:
        lr = args.lr * 0.1 
    if epoch >= 70:
        lr = args.lr * 0.01 
    if epoch >= 90:
        lr = args.lr * 0.005 
    if args.wandb:
        global wandb_dict
        wandb_dict['lr'] = lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_rho(epoch):
    global rho
    if args.rho_setup == 0:
        rho = 0.05
        if epoch >= 75:
            rho = 0.1
    elif args.rho_setup == 1:
        rho = 0.05
    else:
        raise ValueError('rho-setup must be 0 or 1')

def adjust_max_eps(epoch):
    global max_eps
    if ges['mode'] == 'const':
        max_eps = args.epsilon
    elif ges['mode'] == 'linear':
        if epoch < ges['t']:
            max_eps = args.epsilon * (epoch / ges['t'])
        else:
            max_eps = args.epsilon
    elif ges['mode'] == 'curious':
        if epoch < ges['t']:
            max_eps = args.epsilon * (ges['gamma'] * epoch / ges['t'])
        else:
            max_eps = args.epsilon
    elif ges['mode'] == 'curious_smooth':
        if epoch < ges['t']:
            max_eps = args.epsilon * (ges['gamma'] * epoch / ges['t'])
        elif epoch < ges['t'] + 10:
            max_eps = args.epsilon * (ges['gamma'] - 0.1 * (ges['gamma'] - 1) * (epoch - ges['t']))
        else:
            max_eps = args.epsilon 
    else:
        raise ValueError('not supported ges mode')


def eval_init(model, model_c, device, test_loader):
    model.eval()
    model_c.eval()
    test_loss_model = 0
    correct_model = 0
    test_loss_c = 0
    correct_c = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            output_model = model(data)
            test_loss_model += F.cross_entropy(output_model, target, size_average=False).item()
            pred = output_model.max(1, keepdim=True)[1]
            correct_model += pred.eq(target.view_as(pred)).sum().item()
            
            output_c = model_c(data)
            test_loss_c += F.cross_entropy(output_c, target, size_average=False).item()
            pred = output_c.max(1, keepdim=True)[1]
            correct_c += pred.eq(target.view_as(pred)).sum().item()

    test_loss_model /= len(test_loader.dataset)
    test_loss_c /= len(test_loader.dataset)
    log.info('Test [model]: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss_model, 
            correct_model, len(test_loader.dataset), 100. * correct_model / len(test_loader.dataset)))
    log.info('Test [c]: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss_c, 
            correct_c, len(test_loader.dataset), 100. * correct_c / len(test_loader.dataset)))

def main():
    # init model, ResNet18() can be also used here for training
    if args.model == 'resnet':
        print('resnet18')
        model = ResNet18().to(device)
        model_c = ResNet18().to(device)
    elif args.model == 'wideresnet':
        print('wideresnet')
        model = WideResNet(num_classes=num_classes, depth=args.depth, widen_factor=args.widen_factor).to(device)
        model_c = WideResNet(num_classes=num_classes, depth=args.depth, widen_factor=args.widen_factor).to(device)
    else:
        raise ValueError('model must be resnet or wideresnet')
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    start_epoch = 1
    ckpt = torch.load(teacher_dir, map_location='cuda')
    if args.init in ['pretrained']:
        model.load_state_dict(ckpt['state_dict'])
    else:
        log.info('random init')
    model_c.load_state_dict(ckpt['state_dict'])
    eval_init(model, model_c, device, test_loader)

    for epoch in range(start_epoch, args.epochs + 1):
        # adjust learning rate for SGD
        adjust_learning_rate(optimizer, epoch)
        adjust_rho(epoch)
        adjust_max_eps(epoch)

        # adversarial training
        train(args, model, model_c, device, train_loader, optimizer, epoch)

        # evaluation on natural examples
        if epoch >= 60:
            eval_test(model, device, test_loader, repeat=1)
        else:
            eval_test(model, device, test_loader, repeat=1)

        if args.wandb:
            global wandb_dict
            wandb.log(wandb_dict)
            wandb_dict = {}

        # save checkpoint
        if epoch % args.save_freq == 0:
            torch.save({'epoch': epoch, 
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
            }, os.path.join(model_dir, 'ckpt-epoch{}.pt'.format(epoch)))
    if args.wandb:
        wandb.finish()

if __name__ == '__main__':
    main()

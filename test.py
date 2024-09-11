from __future__ import print_function
import os
import argparse
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms

from models.wideresnet import *
from models.resnet import *

from autoattack import AutoAttack
from pgd import PGD_Linf
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Test Adversarial Training')
parser.add_argument('--dataset', type=str, default='cifar10', help='cifar10, svhn, or cifar100')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epsilon', type=float, default=0.031,
                    help='perturbation')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--model-dir', default='model-cifar-wideResNet',
                    help='directory of model for saving checkpoint')
parser.add_argument('--attack-method', type=str, default='pgd', help='pgd or autoattack')
parser.add_argument('--pgd-steps', type=int, default=20,
                    help='number of steps for pgd')
parser.add_argument('--model', default='wideresnet', 
                    help='model architecture resnet or wideresnet')
parser.add_argument('--depth', default=28, type=int, help='hyperparameter depth for WideResNet')
parser.add_argument('--widen-factor', default=5, type=int, help='hyperparameter widen_factor for WideResNet')
parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

if args.dataset in ['cifar10']:
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
elif args.dataset in ['svhn']:
    testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=transform_test)
elif args.dataset in ['cifar100']:
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
else:
    raise ValueError('dataset must be cifar10, cifar100, or svhn')

if args.dataset in ['cifar10', 'svhn']:
    num_classes = 10
elif args.dataset in ['cifar100']:
    num_classes = 100
else:
    raise ValueError('dataset must be cifar10, cifar100, or svhn')

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    test_adv_loss = 0

    correct = []
    robust_correct = []

    if args.attack_method == 'autoattack':
        auto_attack = AutoAttack(model, norm='Linf', eps=args.epsilon, version='standard', verbose=True)
        auto_attack.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab', 'square']

    if args.attack_method == 'apgd-ce':
        auto_attack = AutoAttack(model, norm='Linf', eps=args.epsilon, version='standard', verbose=False)
        auto_attack.attacks_to_run = ['apgd-ce']

    if args.attack_method == 'apgd-dlr':
        auto_attack = AutoAttack(model, norm='Linf', eps=args.epsilon, version='standard', verbose=False)
        auto_attack.attacks_to_run = ['apgd-dlr']

    if args.attack_method == 'apgd-t':
        auto_attack = AutoAttack(model, norm='Linf', eps=args.epsilon, version='standard', verbose=False)
        auto_attack.attacks_to_run = ['apgd-t']

    if args.attack_method in ['pgd']:
        pgd_attack = PGD_Linf(model = model,
                        epsilon=args.epsilon,
                        step_size=args.epsilon/10,
                        num_steps=args.pgd_steps,
                        random_start=True,
                        bn_mode = 'eval',
                        train = False)

    for data, target in tqdm(test_loader):
        data, target = data.to(device), target.to(device)
        # evaluate standard
        with torch.no_grad():
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct.extend(pred.eq(target.view_as(pred)).squeeze().cpu().numpy())

        # evaluate robust
        if args.attack_method == 'pgd':
            data_adv, _ = pgd_attack.perturb(data, target)
        elif args.attack_method in ['autoattack', 'apgd-ce', 'apgd-dlr', 'apgd-t']:
            data_adv = auto_attack.run_standard_evaluation(data, target, bs=args.test_batch_size)
        else:
            raise ValueError('--attack-method must be pgd or autoattack')

        with torch.no_grad():
            output = model(data_adv)
            test_adv_loss += F.cross_entropy(output, target, size_average=False).item()
            pred_adv = output.max(1, keepdim=True)[1]
            robust_correct.extend(pred_adv.eq(target.view_as(pred_adv)).squeeze().cpu().numpy())
        
    correct = sum(correct)
    robust_correct = sum(robust_correct)

    test_loss /= len(test_loader.dataset)
    test_adv_loss /= len(test_loader.dataset)

    if args.attack_method in ['autoattack']:
        attack_name = 'AUTOATTACK'    
    elif args.attack_method in ['pgd']:
        attack_name = f'PGD{args.pgd_steps}'
    elif args.attack_method in ['apgd-ce']:
        attack_name = 'APGD-CE'
    elif args.attack_method in ['apgd-dlr']:
        attack_name = 'APGD-DLR'

    print('{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Average adv loss: {:.4f}, Robust Accuracy: {}/{} ({:.2f}%)'.format(
            attack_name,
            test_loss, 
            correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset),
            test_adv_loss,
            robust_correct, len(test_loader.dataset), 100. * robust_correct / len(test_loader.dataset)))
    test_accuracy = correct / len(test_loader.dataset)
    robust_accuracy = robust_correct / len(test_loader.dataset)

    return test_loss, test_adv_loss, test_accuracy, robust_accuracy

def main():
    if args.model == 'resnet':
        model = ResNet18().to(device)
    elif args.model == 'wideresnet':
        model = WideResNet(num_classes=num_classes, depth=args.depth, widen_factor=args.widen_factor).to(device)
    else:
        raise ValueError('model must be resnet or wideresnet')

    if 'srstawr' in args.model_dir:
        ckpt = torch.load(args.model_dir, map_location='cuda')
        model.load_state_dict(ckpt['swa_state_dict'])
        ckpt = torch.load(args.model_dir, map_location='cuda')
        state_dict = {}
        for key, value in ckpt['state_dict'].items():
            new_key = key.replace('module.', '')
            state_dict[new_key] = value
        model.load_state_dict(state_dict)
    else:
        ckpt = torch.load(os.path.join('checkpoints', args.model_dir))
        model.load_state_dict(ckpt['model'])

    eval_test(model, device, test_loader)

if __name__ == '__main__':
    main()

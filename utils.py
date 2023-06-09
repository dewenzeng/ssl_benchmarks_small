from typing import Dict, Optional, Callable, Tuple, Any

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data.dataset import Dataset

class ContrastiveLearningViewGenerator(object):
    """Generate a multiple views of the same image."""

    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, x):
        return [transform(x) for transform in self.transform_list]

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def cosine_simililarity(x, y):
    # x shape: (N, 1, C)
    # y shape: (1, N, C)
    # v shape: (N, N)
    v = torch.nn.CosineSimilarity(dim=-1)(x.unsqueeze(1), y.unsqueeze(0))
    return v

def get_resnet(backbone='resnet18', size='small', head='mlp', feat_dim=256, hidden_size=2048):
    if backbone == 'resnet18':
        model = torchvision.models.resnet18()
    elif backbone == 'resnet50':
        model = torchvision.models.resnet50()
    input_features = model.fc.in_features
    if head == 'mlp':
        model.fc = nn.Sequential(
                nn.Linear(input_features, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, feat_dim)
            )
    elif head == 'linear':
        model.fc = nn.Linear(input_features, feat_dim)
    elif head == 'simsiam':
        model.fc = nn.Sequential(
                nn.Linear(input_features, hidden_size, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, hidden_size, bias=False),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, feat_dim, bias=False),
            )
    # For small resolution dataset like cifar10 and cifar100.
    if size == 'small':
       model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
       model.maxpool = nn.Identity()
    return model

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_ssl_dataset(modelConfig):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    if modelConfig['dataset'] == 'cifar10':
        transform_list = [
            transforms.Compose([
                transforms.RandomResizedCrop(size=modelConfig["img_size"], scale=(0.08, 1.0)),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            ]) for _ in range(2)
        ]
        dataset = datasets.CIFAR10(
            root='/home/ubuntu/data/cifar10/', 
            train=True, download=True,
            transform=ContrastiveLearningViewGenerator(transform_list),
        )
    elif modelConfig['dataset'] == 'cifar100':
        transform_list = [
            transforms.Compose([
                transforms.RandomResizedCrop(size=modelConfig["img_size"], scale=(0.08, 1.0)),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomHorizontalFlip(),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
            ]) for _ in range(2)
        ]
        dataset = datasets.CIFAR100(
            root='/home/ubuntu/data/cifar100/', 
            train=True, download=True,
            transform=ContrastiveLearningViewGenerator(transform_list),
        )
    return dataset

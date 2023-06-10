import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from utils import *

def get_linear_eval_dataset(dataset='cifar10', batch_size=256):
    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(
                                    root='/home/ubuntu/data/cifar10/',
                                    train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
                                    ]
                                )
                            )
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=8, drop_last=False, shuffle=True)
        
        test_dataset = datasets.CIFAR10(
                                    root='/home/ubuntu/data/cifar10/',
                                    train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
                                    ]
                                )
                            )
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                num_workers=8, drop_last=False, shuffle=False)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(
                                    root='/home/ubuntu/data/cifar100/',
                                    train=True, download=True,
                                    transform=transforms.Compose([
                                        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
                                    ]
                                )
                            )
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                num_workers=8, drop_last=False, shuffle=True)
        
        test_dataset = datasets.CIFAR100(
                                        root='/home/ubuntu/data/cifar100/',
                                        train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.CenterCrop(size=32),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)),
                                        ]))
        test_loader = DataLoader(test_dataset, batch_size=batch_size,
                                num_workers=8, drop_last=False, shuffle=False)
    return train_loader, test_loader


def eval(backbone, state_dict, dataset_name, device='cuda:0'):
    # define model.
    if dataset_name in ['cifar10']:
        num_classes = 10
        model = get_resnet(backbone=backbone, size='small', head='linear', feat_dim=num_classes)
    elif dataset_name == 'cifar100':
        num_classes = 100
        model = get_resnet(backbone=backbone, size='small', head='linear', feat_dim=num_classes)

    # ignore the fc in the saved state_dict.
    for k in list(state_dict.keys()):
        if k.startswith('fc'):
            del state_dict[k]

    log = model.load_state_dict(state_dict, strict=False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']

    in_features = model.fc.in_features
    model.fc = torch.nn.Identity()
    classifier = torch.nn.Linear(in_features, num_classes)

    model.to(device)
    classifier.to(device)

    train_loader, test_loader = get_linear_eval_dataset(dataset=dataset_name, batch_size=128)

    optimizer = torch.optim.SGD(classifier.parameters(), lr=1.0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60, 80], gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    epochs = 100
    model.eval()
    for epoch in range(epochs):
        train_top1 = AverageMeter()
        classifier.train()
        for _, (x_batch, y_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            features = model(x_batch)
            logits = classifier(features.detach())
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            train_top1.update(top1[0].item(), x_batch.shape[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        test_top1 = AverageMeter()
        test_top5 = AverageMeter()
        classifier.eval()
        with torch.no_grad():
            for _, (x_batch, y_batch) in enumerate(test_loader):
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                features = model(x_batch)
                logits = classifier(features.detach())
                top1, top5 = accuracy(logits, y_batch, topk=(1,5))
                test_top1.update(top1[0].item(), x_batch.shape[0])
                test_top5.update(top5[0].item(), x_batch.shape[0])
        
        print(f"Linear eval\tEpoch {epoch+1}/{epochs}\tTop1 Train accuracy {train_top1.avg:.2f}\tTop1 Test accuracy: {test_top1.avg:.2f}\tTop5 test acc: {test_top5.avg:2f}")

    return test_top1.avg, test_top5.avg

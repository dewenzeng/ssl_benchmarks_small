import os
import copy
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import manifold
from torch.utils.data import DataLoader
from Scheduler import GradualWarmupScheduler
from datetime import datetime
from linear_eval import eval
from utils import *
from torch.utils.tensorboard import SummaryWriter

def info_nce_loss_with_label(features, temperature=0.1, labels=None, mask=None):
    """Supervsed info_nce loss.
    
    https://github.com/HobbitLong/SupContrast/blob/master/losses.py#L11
    """
    
    device = features.device

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    if labels is not None and mask is not None:
        raise ValueError('Cannot define both `labels` and `mask`')
    elif labels is None and mask is None:
        mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    elif labels is not None:
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)
    else:
        mask = mask.float().to(device)

    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
    anchor_feature = contrast_feature
    anchor_count = contrast_count

    anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            temperature)
    # for numerical stability
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()

    # tile mask
    mask = mask.repeat(anchor_count, contrast_count)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    # compute mean of log-likelihood over positive
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # loss
    loss = -1.0 * mean_log_prob_pos
    loss = loss.view(anchor_count, batch_size).mean()

    return loss

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    dataset = get_ssl_dataset(modelConfig)
    dataloader = DataLoader(dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=8, drop_last=True, pin_memory=False)
    # tensorboard summary
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    summary_string = f"simclr_{modelConfig['dataset']}_{modelConfig['backbone']}_bs-{modelConfig['batch_size']}_temp-{modelConfig['temperature']}_{current_time}"
    summary_dir = f"runs/{summary_string}"
    writer = SummaryWriter(summary_dir)
    # recording the modelConfig in the summary writer.
    writer.add_text('config', str(modelConfig), 0)
    save_weight_dir = os.path.join(f"./output/{summary_string}", modelConfig["save_weight_dir"])
    plot_dir = os.path.join(f"./output/{summary_string}", modelConfig["plot_dir"])
    os.makedirs(save_weight_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    # model setup
    if modelConfig['dataset'] in ['cifar10', 'cifar100']:
        model = get_resnet(backbone=modelConfig["backbone"], size='small', head='mlp', feat_dim=modelConfig["feat_dim"])
    else:
        model = get_resnet(backbone=modelConfig["backbone"], size='big', head='mlp', feat_dim=modelConfig["feat_dim"])
    if modelConfig["use_dp"]:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=modelConfig["lr"], momentum=modelConfig["momentum"], weight_decay=modelConfig["weight_decay"])
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=10, after_scheduler=cosineScheduler)

    # start training
    n_iter = 0
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            model.train()
            for images, labels in tqdmDataLoader:
                # train
                bsz = images[0].shape[0]
                images = torch.cat([images[0], images[1]], dim=0).cuda()
                features = model(images)
                features = torch.nn.functional.normalize(features, dim=1)
                f1, f2 = torch.split(features, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                loss = info_nce_loss_with_label(features, temperature=modelConfig['temperature'])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "Train epoch": e,
                    "loss": loss.item(),
                    "lr": optimizer.state_dict()['param_groups'][0]["lr"]
                })
                writer.add_scalar('loss', loss.item(), global_step=n_iter)
                n_iter += 1

        warmUpScheduler.step()
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=e)
        if (e+1) % modelConfig["save_ckpt_interval"] == 0:
            # Do one evaluation.
            top1, top5 = eval(
                backbone=modelConfig["backbone"],
                state_dict=model.module.state_dict() if modelConfig["use_dp"] else model.state_dict(), 
                dataset_name=modelConfig['dataset'], 
                device=device
            )
            writer.add_scalar('test/top1', top1, global_step=e+1)
            writer.add_scalar('test/top5', top5, global_step=e+1)

            torch.save(model.module.state_dict() if modelConfig["use_dp"] else model.state_dict(), os.path.join(
                save_weight_dir, 'ckpt_' + str(e) + "_.pt"))
            
            if modelConfig["draw_tsne"]:
                features_list = []
                labels_list = []
                with torch.no_grad():
                    model_eval = copy.deepcopy(model.module if modelConfig["use_dp"] else model)
                    model_eval.fc = nn.Identity()
                    model_eval.to(device)
                    model_eval.eval()
                    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
                        num = 0
                        for images, labels in tqdmDataLoader:
                            x_0 = images[0].to(device)
                            features = model_eval(x_0)
                            features = torch.flatten(features, start_dim=1)
                            features_list.append(features.detach().cpu())
                            labels_list.append(labels)
                            num += 1
                            if num > 3:
                                break
                del model_eval
                features_list = torch.cat(features_list).numpy()
                labels_list = torch.cat(labels_list).numpy()

                X = features_list
                y = labels_list

                tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
                X_tsne = tsne.fit_transform(X)

                x_min, x_max = X_tsne.min(0), X_tsne.max(0)
                X_norm = (X_tsne - x_min) / (x_max - x_min)
                plt.figure(figsize=(6, 6))
                colors = np.linspace(0,1,100) if modelConfig["dataset"] == "cifar100" else np.linspace(0,1,10)
                color_map = matplotlib.colormaps['viridis'] if modelConfig["dataset"] == "cifar100" else matplotlib.colormaps['tab10']
                for i in range(X_norm.shape[0]):
                    circle = plt.Circle([X_norm[i, 0], X_norm[i, 1]], radius=0.005, color=color_map(colors[y[i]]), fill=True)
                    plt.gca().add_patch(circle)
                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.savefig(os.path.join(plot_dir, f'tsne_{e:03d}.png'))

if __name__ == '__main__':
    modelConfig = {
        "epoch": 1000,
        "batch_size": 1024,
        "lr": 1e-2,
        "multiplier": 50,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "img_size": 32,
        "device": "cuda:0",
        "save_weight_dir": "checkpoints",
        "plot_dir": "tsne",
        "save_ckpt_interval": 200,
        "dataset": "cifar100",
        "feat_dim": 256,
        "temperature": 0.5,
        "backbone": "resnet18",
        "use_dp": True,    # use DataParallel
        "draw_tsne": False,
    }
    train(modelConfig)

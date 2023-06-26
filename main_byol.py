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
from byol_pytorch.byol_pytorch import BYOL
from linear_eval import eval
from utils import *
from torch.utils.tensorboard import SummaryWriter

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    dataset = get_ssl_dataset(modelConfig)
    dataloader = DataLoader(dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=8, drop_last=True, pin_memory=False)
    # tensorboard summary
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    summary_string = f"byol_{modelConfig['dataset']}_{modelConfig['backbone']}_bs-{modelConfig['batch_size']}_{current_time}"
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
        # doesn't matter what feat_dim is used here cause it's never used.
        model = get_resnet(backbone=modelConfig["backbone"], size='small', head='linear', feat_dim=10)
    else:
        model = get_resnet(backbone=modelConfig["backbone"], size='big', head='linear', feat_dim=10)
    
    learner = BYOL(
        model,
        image_size = modelConfig["img_size"],
        hidden_layer = 'avgpool',
        projection_size = modelConfig["projection_size"],
        projection_hidden_size = modelConfig["projection_hidden_size"],
    )
    
    if modelConfig["use_dp"]:
        learner = nn.DataParallel(learner)
    learner.to(device)

    optimizer = torch.optim.SGD(learner.parameters(), lr=modelConfig["lr"], momentum=modelConfig["momentum"], weight_decay=modelConfig["weight_decay"])
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
                loss = learner(images[0].to(device), images[1].to(device))
                loss = loss.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # update EMA and momentum tau.
                if modelConfig["use_dp"]:
                    learner.module.update_moving_average()
                    learner.module.update_tau(
                        cur_step=n_iter,
                        max_steps=modelConfig["epoch"] * len(dataloader),
                    )
                else:
                    learner.update_moving_average()
                    learner.update_tau(
                        cur_step=n_iter,
                        max_steps=modelConfig["epoch"] * len(dataloader),
                    )
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
                state_dict=learner.module.online_encoder.net.state_dict() if modelConfig["use_dp"] else learner.online_encoder.net.state_dict(),
                dataset_name=modelConfig['dataset'], 
                device=device
            )
            writer.add_scalar('test/top1', top1, global_step=e+1)
            writer.add_scalar('test/top5', top5, global_step=e+1)

            torch.save(learner.module.online_encoder.net.state_dict() if modelConfig["use_dp"] else learner.online_encoder.net.state_dict(), os.path.join(save_weight_dir, 'ckpt_' + str(e) + "_.pt"))
            
            if modelConfig["draw_tsne"]:
                features_list = []
                labels_list = []
                with torch.no_grad():
                    model_eval = copy.deepcopy(learner.module.online_encoder.net if modelConfig["use_dp"] else learner.online_encoder.net)
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
        "batch_size": 512,
        "lr": 1e-2,
        "multiplier": 100,
        "momentum": 0.9,
        "weight_decay": 1e-5,
        "img_size": 32,
        "device": "cuda:0",
        "save_weight_dir": "checkpoints",
        "plot_dir": "tsne",
        "save_ckpt_interval": 200,
        "dataset": "cifar100",
        "backbone": "resnet18",
        "projection_size": 256,
        "projection_hidden_size": 4096,
        "use_dp": True,    # use DataParallel
        "draw_tsne": False,
    }
    train(modelConfig)

import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms

from torchvision.utils import save_image

from ddpm_cfg.model import UNet
from ddpm_cfg.scheduler import GradualWarmupScheduler
from ddpm_cfg.diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler

from tqdm import tqdm


def train(model_config):
    device = torch.device(model_config["device"])
    
    dataset = CIFAR10(
        root = "./CIFAR10",
        train = True,
        download = True,
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )
    )
    
    dataloader = DataLoader(dataset = dataset, 
                            batch_size = model_config["batch_size"],
                            shuffle = True,
                            num_workers = 4,
                            drop_last = True,
                            pin_memory = True)
    
    net_model = UNet(T=model_config["T"], 
                     num_labels=10, 
                     dim=model_config["dim"], 
                     dim_scale=model_config["dim_scale"], 
                     num_res_blocks=model_config["num_res_blocks"],
                     dropout=model_config["dropout"]).to(device)
    
    if not os.path.exists(model_config["save_dir"]):
        os.makedirs(model_config["save_dir"])
    
    if model_config["load_weight"] is not None:
        model_dict = torch.load(os.path.join(model_config["save_dir"], model_config["load_weight"]), map_location=device)
        net_model.load_state_dict(model_dict)
        print("model load weight done.")

    optimizer = torch.optim.AdamW(net_model.parameters(), lr=model_config["lr"], weight_decay=1e-4)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                              T_max=model_config["epoch"],
                                                              eta_min=0,
                                                              last_epoch=-1)
    warmup_scheduler = GradualWarmupScheduler(optimizer=optimizer, 
                                              multiplier=model_config["multiplier"],
                                              warm_epoch=model_config["epoch"] // 10,
                                              after_scheduler=cosine_scheduler)
    
    trainer = GaussianDiffusionTrainer(net_model, model_config["beta_1"], model_config["beta_T"], model_config["T"]).to(device)
    
    for epoch in range(model_config["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                optimizer.zero_grad()
                b = images.shape[0]
                x_0 = images.to(device)
                labels = labels.to(device) + 1
                if np.random.rand() < 0.1:
                    labels = torch.zeros_like(labels).to(device)
                loss = trainer(x_0, labels).sum() / b ** 2
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net_model.parameters(), model_config["clip_grad"])
                optimizer.step()
                
                tqdmDataLoader.set_postfix(ordered_dict = {
                    "epoch": epoch,
                    "loss": loss.item(),
                    "image shape": x_0.shape,
                    "lr": optimizer.state_dict()["param_groups"][0]["lr"],
                    })
        warmup_scheduler.step()
        
        if epoch % 20 == 0 or epoch == model_config["epoch"]-1:
            torch.save(net_model.state_dict(), os.path.join(model_config["save_dir"], f"ckpt_{epoch}.pt"))
        

def eval(model_config):
    device = torch.device(model_config["device"])
    
    with torch.no_grad():
        step = int(model_config["batch_size"] // 10)
        labelList = []
        k = 0
        
        for i in range(1, model_config["batch_size"] + 1):
            labelList.append(torch.ones(size=[1]).long() * k)
            if i % step == 0:
                if k < 10 - 1:
                    k += 1
        labels = torch.cat(labelList, dim=0).long().to(device) + 1
        print(labels)
        model = UNet(T=model_config["T"], 
                     num_labels=10,
                     dim=model_config["dim"], 
                     dim_scale=model_config["dim_scale"], 
                     num_res_blocks=model_config["num_res_blocks"],
                     dropout=0.).to(device)
        
        ckpt = torch.load(os.path.join(model_config["save_dir"], model_config["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        
        model.eval()
        
        sampler = GaussianDiffusionSampler(model, model_config["beta_1"], model_config["beta_T"], model_config["T"]).to(device)
        
        noise_image = torch.randn(size=[model_config["batch_size"], 3, model_config["img_size"], model_config["img_size"]], device=device)
        save_noise = torch.clamp(noise_image * 0.5 + 0.5, 0, 1)
        
        if not os.path.exists(model_config["sampled_dir"]):
            os.makedirs(model_config["sampled_dir"])
        
        save_image(save_noise, os.path.join(model_config["sampled_dir"], model_config["sampled_noise_name"]), nrow=model_config["nrow"])
        sampled_image = sampler(noise_image, labels)
        sampled_image = sampled_image * 0.5 + 0.5
        
        save_image(sampled_image, os.path.join(model_config["sampled_dir"], model_config["sampled_image_name"]), nrow=model_config["nrow"])
        
        
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import distance





class FeatureAttention(nn.Module):
    def __init__(self, input_dim, reduction_ratio=2):
        super().__init__()
        # Reduction ratio keeps the parameter count low
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // reduction_ratio),
            nn.ReLU(),
            nn.Linear(input_dim // reduction_ratio, input_dim),
            nn.Sigmoid() # Forces weights to be between 0 and 1
        )

    def forward(self, x):
        # Generate the attention weights (the "mask")
        attention_weights = self.fc(x)
        # Multiply original features by the weights
        return x * attention_weights




class EmbeddingNet(nn.Module):
    def __init__(self, mode = 'normal', input_ftrs = 141):
        super().__init__()
        self.attention = FeatureAttention(input_ftrs)

        if  mode == 'leaky5':
            self.emb_net = nn.Sequential(
            nn.Linear(in_features= input_ftrs, out_features= 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 16)
        )
        
        elif  mode == 'leaky6':
            self.emb_net = nn.Sequential(
            nn.Linear(in_features= input_ftrs, out_features= 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 16),
            nn.Sigmoid()
        )


        else:
            self.emb_net = nn.Sequential(
                nn.Linear(in_features= input_ftrs, out_features= 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 10)
            )


    def forward(self, x):

        x = self.attention(x)
        x = self.emb_net(x)

        return x


class SiameseNet(nn.Module):
    def __init__(self, emb_net):
        super().__init__()
        self.emb_net = emb_net
    
    def forward(self, ft1, ft2):
        z1 = self.emb_net(ft1)
        z2 = self.emb_net(ft2)

        return z1, z2



def stress_loss(z1, z2, high_dim_distance):

    low_dim_distance = torch.linalg.vector_norm(z1 - z2, ord=2, dim=1)

    eps = 1e-7
    high_dim_safe = high_dim_distance + eps

    error = ((high_dim_distance - low_dim_distance) ** 2) / high_dim_safe

    loss = torch.sum(error) / (torch.sum(high_dim_safe) + eps)

    return loss




def train(train_dataloader, model, device, loss_fn, optim):

    dataset_size = len(train_dataloader.dataset)
    batch_size = train_dataloader.batch_size

    model.train()


    for batch, (x1, x2, dist) in enumerate(train_dataloader):
        x1, x2, dist = x1.to(device), x2.to(device), dist.to(device)

        z1, z2 = model(x1, x2)

        loss = loss_fn(z1, z2, dist)


        optim.zero_grad()
        loss.backward()
        optim.step()

        if (batch%2) == 0:
            train_loss, current = loss.item(), batch * batch_size + len(z1)
            print(f"loss: {train_loss:>7f}  [{current:>5d}/{dataset_size:>5d}]")



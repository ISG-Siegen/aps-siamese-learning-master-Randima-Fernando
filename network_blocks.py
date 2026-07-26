import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import distance





class FeatureAttention(nn.Module):
    """
    Input-level Squeeze-and-Excitation (SE) attention block.

    This module dynamically recalibrates the importance of each input feature 
    by generating an attention mask. It uses a bottleneck multi-layer perceptron 
    (MLP) to keep the parameter count low, outputting scaling weights between 0 and 1 
    that are multiplied element-wise with the original input.

    Args:
        input_dim (int): The number of input meta-features.
        reduction_ratio (int, optional): The factor by which the hidden layer 
            dimension is reduced to restrict model complexity. Defaults to 2.

    Methods:
        forward(x):
            Generates the attention weights and applies them to the input features.
            
            Shape:
                - Input (x): (batch_size, input_dim)
                - Output: (batch_size, input_dim) - the scaled input features.
    """


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
    """
    A Siamese Neural Network wrapper for dual-branch feature extraction.

    This module takes an existing embedding network and applies it identically 
    (with shared weights) to a pair of input vectors. By processing both the 
    anchor and positive dataset features through the exact same network, it 
    allows the model to learn a meaningful latent space where distances can 
    be accurately measured.

    Args:
        emb_net (nn.Module): The underlying neural network (e.g., EmbeddingNet) 
            used to process both branches of the Siamese network.

    Shape:
        - Inputs (ft1, ft2): Two tensors, each of shape (batch_size, input_dim), 
          representing paired meta-feature vectors.
        - Outputs (z1, z2): A tuple of two tensors, each of shape 
          (batch_size, embedding_dim), representing the latent embeddings of 
          the respective inputs.
    """

    def __init__(self, emb_net):
        super().__init__()
        self.emb_net = emb_net
    
    def forward(self, ft1, ft2):
        z1 = self.emb_net(ft1)
        z2 = self.emb_net(ft2)

        return z1, z2



def stress_loss(z1, z2, high_dim_distance):
    """
    Calculates a Sammon Stress-based loss for the latent embeddings.

    This function computes the pairwise Euclidean (L2) distance between the 
    generated latent embeddings (z1 and z2) and compares it against their 
    ground-truth distance in the high-dimensional Algorithm Performance Space (APS). 
    The loss penalizes discrepancies proportionally to the original distance, 
    using a small epsilon value to prevent division by zero errors.

    Args:
        z1 (torch.Tensor): The latent embedding of the first input (Anchor). 
            Expected shape: (batch_size, embedding_dim).
        z2 (torch.Tensor): The latent embedding of the second input (Positive). 
            Expected shape: (batch_size, embedding_dim).
        high_dim_distance (torch.Tensor): The actual APS distance between the 
            two original datasets. Expected shape: (batch_size,).

    Returns:
        torch.Tensor: A scalar tensor representing the normalized stress loss 
        for the batch.
    """

    low_dim_distance = torch.linalg.vector_norm(z1 - z2, ord=2, dim=1)

    eps = 1e-7
    high_dim_safe = high_dim_distance + eps

    error = ((high_dim_distance - low_dim_distance) ** 2) / high_dim_safe

    loss = torch.sum(error) / (torch.sum(high_dim_safe) + eps)

    return loss




def train(train_dataloader, model, device, loss_fn, optim):
    """
    Executes a single training epoch for the model.

    This function sets the model to training mode and iterates through the 
    provided dataloader. For each batch, it moves the data to the specified 
    device, performs a forward pass to generate latent embeddings, calculates 
    the loss, and updates the network weights using backpropagation. It also 
    prints the current training loss and progress every 2 batches.

    Args:
        train_dataloader (torch.utils.data.DataLoader): The dataloader providing 
            batches of training data. It is expected to yield tuples of.
        model (torch.nn.Module): The neural network being trained (e.g., SiameseNet).
        device (torch.device): The hardware device to run computations on.
        loss_fn: The function used to calculate the training loss.
        optim (torch.optim.Optimizer): The PyTorch optimizer used to update 
            the model's parameters.
    """

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



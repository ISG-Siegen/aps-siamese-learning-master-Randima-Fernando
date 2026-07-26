import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from  utils import sort_by_distance, matches_closes_nots, distance_rmse_mae
from sklearn.preprocessing import StandardScaler
from aps_data_handling import FetchAPSFeatureData, leave_one_out
from network_blocks import EmbeddingNet, SiameseNet
import getpass
import sys



def test_left_out(aps_vectors, feature_dataset, aps_left_out, feature_left_out, needed_range, weights_path):

    dataset_size = len(aps_vectors)
    input_features_num = len(feature_dataset.columns[1:])

    device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

    print(f"Using {device} device")


    model = SiameseNet(EmbeddingNet(mode= 'leaky6', input_ftrs= input_features_num)).to(device)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model = model.to(device)

    embedding_list = []


    model.eval()
    with torch.no_grad():
        for i in range(dataset_size):

            # Passes meta features from dataframe to model to obtain low-dimensional embeddings. 
            # Data preprocessed into numpy float arrays with an extra dimension (batch dim) before being input
            x = torch.tensor(feature_dataset.iloc[i,1:].astype(float).to_numpy(), dtype= torch.float32).unsqueeze(0).to(device)

            z, _ = model(x, x)

            embedding_list.append(z.squeeze(0).to('cpu').tolist())
        
        # Left-out dataset processed into numpy float array with a batch dimension and input to network
        x = torch.tensor(feature_left_out.iloc[1:].astype(float).to_numpy(), dtype= torch.float32).unsqueeze(0).to(device)
        z, _ = model(x, x)

        # Embedding for left-out dataset and concatenates the dataset name
        lo_dataset_embedded = z.squeeze(0).to('cpu').tolist()
        lo_dataset_embedded = pd.Series(lo_dataset_embedded, index= [i+1 for i in range(16)])
        temp_sr = pd.Series([aps_left_out.loc['dataset']], index = ['dataset'])     # Gets left-out dataset name for concatenation
        lo_dataset_embedded = pd.concat([temp_sr, lo_dataset_embedded])

    # Defines embedded vectors dataframe containing the low-dim embedding of each APS vector
    embedded_vectors = pd.DataFrame(embedding_list, columns= [i+1 for i in range(16)])
    embedded_vectors.insert(0, column='dataset' ,value= aps_vectors['dataset'])

    aps_sorted = sort_by_distance(aps_vectors, aps_left_out).reset_index(drop= True)
    embedded_sorted =  sort_by_distance(embedded_vectors, lo_dataset_embedded).reset_index(drop= True)

    _, rmse, mae = distance_rmse_mae(aps_sorted, embedded_sorted, needed_range)

    
    nrmse = (rmse/aps_sorted['Distance'].mean())*100
    nmae_mean = (mae /aps_sorted['Distance'].mean())*100
    nmae_range = (mae /26)*100


    return [rmse, nrmse, mae, nmae_mean, nmae_range]


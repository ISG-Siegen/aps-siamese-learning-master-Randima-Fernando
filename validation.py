import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import distance
from network_blocks import EmbeddingNet, SiameseNet
import getpass
import sys



def test_left_out(aps_vectors, feature_dataset, aps_left_out, feature_left_out, needed_range, weights_path):
    """
    Evaluates the trained Siamese network on a left-out dataset for LOOCV.

    This function loads saved model weights and generates latent space embeddings 
    for all datasets in the training set as well as the single left-out dataset. 
    It then calculates the distances from the left-out dataset to all other 
    datasets in the latent space, compares these against the ground-truth 
    Algorithm Performance Space (APS) distances, and computes various error metrics.

    Args:
        aps_vectors (pd.DataFrame): The ground-truth APS coordinate data for the 
            training datasets. Expected to include a 'dataset' column.
        feature_dataset (pd.DataFrame): The meta-feature data for the training 
            datasets. Expected to have the dataset name in the first column and 
            features in the subsequent columns.
        aps_left_out (pd.Series): The ground-truth APS vector for the left-out 
            validation dataset.
        feature_left_out (pd.Series): The meta-feature vector for the left-out 
            validation dataset.
        needed_range (int): The range or number of nearest neighbors to consider 
            when calculating the distance error metrics (passed to `distance_rmse_mae`).
        weights_path (str): The file path to the saved model weights (`.pt` or `.pth`) 
            for the current LOOCV fold.

    Returns:
        list of float: A list containing five calculated error metrics:
            - rmse (float): Root Mean Squared Error.
            - nrmse (float): Normalized RMSE (scaled by the mean true distance).
            - mae (float): Mean Absolute Error.
            - nmae_mean (float): Normalized MAE (scaled by the mean true distance).
            - nmae_range (float): Normalized MAE (scaled by the constant range value 26).
    """


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


    model = SiameseNet(EmbeddingNet(network_num= 5 , input_ftrs= input_features_num)).to(device)
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



def sort_by_distance(data, selected_ds):
    """
    Accepts a dataframe of vectors to be sorted according to the distance relative to "selected_ds".
    selected_ds should be a pandas series. "data" should be a pandas dataframe with a column 'dataset'.
    Returns the 'sorted_data' dataframe with the columns:
    0 : 'dataset' 1 : 'Distance'  2-end : Respective vectors
    """
    df_data_dist = pd.DataFrame({'dataset': data['dataset']})
    data_dist = []
    dataset_size = len(data)
    aps_vec1 = selected_ds.iloc[1:].astype(float).to_numpy()

    for i in range(dataset_size):
        
        aps_vec2 = data.iloc[i, 1:].astype(float).to_numpy()

        dist_data = distance.euclidean(aps_vec1, aps_vec2)

        data_dist.append(dist_data)

    df_data_dist['Distance'] = data_dist 
    aps_dist_sorted = df_data_dist.sort_values('Distance', ascending= True).reset_index(drop= True)
    data = data.set_index("dataset", drop= False)
    sorted_data = data.loc[aps_dist_sorted['dataset'].to_list()].reset_index(drop= True)
    sorted_data = pd.concat([aps_dist_sorted, sorted_data.iloc[:,1:]], axis=1)
    data = data.reset_index(drop= True)

    return sorted_data



def distance_rmse_mae(aps, embedded, relevant_datasets):
    """
    Accepts the pandas dataframes for the sorted aps values and sorted embedded values according to their distances to the 
    left-out dataset. Compares the distances given by the APS space vs the Embedded space, to iteratively obtain the errors
    and eventually return the RMSE. 
    """

    merged = aps.merge(embedded, on='dataset', suffixes=('_1', '_2'))

    error = (merged['Distance_1'] - merged['Distance_2']).iloc[:relevant_datasets].tolist()

    rmse = np.sqrt(np.mean(np.array(error)**2))
    mae = np.mean(np.absolute(np.array(error)))

    return error, rmse, mae
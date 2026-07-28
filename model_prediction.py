import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
from aps_data_handling import leave_one_out, FetchAPSFeatureData
from network_blocks import EmbeddingNet, SiameseNet
from scipy.optimize import least_squares
import time
import getpass
import sys



def make_prediction(aps_vectors, feature_dataset, aps_left_out, feature_left_out, weights_path):

    """
    Generates APS vector predictions using Nearest Neighbor and Trilateration.

    Loads the saved model weights, projects both the training dataset and the 
    left-out validation dataset into the latent embedding space, and sorts the 
    training datasets by their distance to the left-out dataset. 

    It then calculates predictions in two ways:
    1. Nearest Neighbor: Directly takes the APS vector of the closest embedding.
    2. Trilateration: Uses the nearest anchors and their latent distances to 
       solve a least-squares optimization problem, predicting a new APS vector.

    Args:
        aps_vectors (pd.DataFrame): Ground-truth APS coordinates for the training set.
        feature_dataset (pd.DataFrame): Meta-features for the training set.
        aps_left_out (pd.Series): Ground-truth APS vector for the left-out dataset.
        feature_left_out (pd.Series): Meta-features for the left-out dataset.
        weights_path (str): File path to the PyTorch model weights for this fold.

    Returns:
        tuple of float: A tuple containing four error metrics:
            - mae_nearest (float): Mean Absolute Error of the Nearest Neighbor prediction.
            - dist_nearest (float): RMSE/Euclidean distance of the Nearest Neighbor prediction.
            - mae_tri (float): Mean Absolute Error of the Trilateration prediction.
            - dist_tri (float): RMSE/Euclidean distance of the Trilateration prediction.
    """

    dataset_size = len(aps_vectors)
    input_features = len(feature_dataset.columns[1:])

    device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )



    model = SiameseNet(EmbeddingNet(network_num= 5, input_ftrs= input_features)).to(device)
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
        
        x = torch.tensor(feature_left_out.iloc[1:].astype(float).to_numpy(), dtype= torch.float32).unsqueeze(0).to(device)
        z, _ = model(x, x)

        # Embedding for left-out dataset
        lo_dataset_embedded = z.squeeze(0).to('cpu').tolist()
        lo_dataset_embedded = pd.Series(lo_dataset_embedded, index= [i+1 for i in range(16)])
        temp_sr = pd.Series([aps_left_out.loc['dataset']], index = ['dataset'])     # Gets left-out dataset name for concatenation
        lo_dataset_embedded = pd.concat([temp_sr, lo_dataset_embedded])

    # Defines embedded vectors dataframe containing the low-dim embedding of each APS vector
    embedded_vectors = pd.DataFrame(embedding_list, columns= [i+1 for i in range(16)])
    embedded_vectors.insert(0, column='dataset' ,value= aps_vectors['dataset'])

    aps_sorted = sort_by_distance(aps_vectors, aps_left_out).reset_index(drop= True)
    embedded_sorted =  sort_by_distance(embedded_vectors, lo_dataset_embedded).reset_index(drop= True)

    # Nearest dataset in embedding space and get APS vector
    nearest_dataset_name = embedded_sorted.iloc[0]['dataset']
    nearest_aps_row = aps_vectors[aps_vectors['dataset'] == nearest_dataset_name].iloc[0]
    predicted_aps = nearest_aps_row.iloc[1:].astype(float).to_numpy()

    # The actual aps of left-out dataset
    actual_aps = aps_left_out.iloc[1:].astype(float).to_numpy()

    # Calculate error metrics
    # Absolute Errors
    absolute_errors = np.abs(predicted_aps - actual_aps)
    mae_nearest = np.mean(absolute_errors)

    # Squared Errors
    squared_errors = (predicted_aps - actual_aps) ** 2
    dist_nearest = np.sqrt(np.mean(squared_errors))


    # Trilateration

    # Define nearest neighbors for trilateration
    k_anchors = min(3, len(embedded_sorted)) 
    
    anchor_aps_list = []
    target_distances = []

    for idx in range(k_anchors):
        anchor_name = embedded_sorted.iloc[idx]['dataset']
        
        # Pull the pre-calculated distance directly from the dataframe
        emb_distance = embedded_sorted.iloc[idx]['Distance']
        target_distances.append(emb_distance)
        
        # Get Anchor's 26D APS Coordinates from the original aps_vectors dataframe
        anchor_aps = aps_vectors[aps_vectors['dataset'] == anchor_name].iloc[0, 1:].astype(float).to_numpy()
        anchor_aps_list.append(anchor_aps)

    anchors = np.array(anchor_aps_list)          # Shape: (k_anchors, 26)
    target_distances = np.array(target_distances) # Shape: (k_anchors,)
    

    # The objective function for least squares
    def distance_residuals(guess_aps, anchors, target_distances):
        # Calculate Euclidean distance from the guess to all anchors in 26D space
        current_distances = np.linalg.norm(anchors - guess_aps, axis=1)
        # We want the difference between the 26D distances and the 16D embedding distances to be zero
        return current_distances - target_distances
    
    # Initial Guess
    # Start the optimizer at the exact coordinates of the closest neighbor
    initial_guess = anchors[0]

    # Run the Optimizer
    result = least_squares(distance_residuals, initial_guess, args=(anchors, target_distances))
    predicted_aps = result.x


    # 5. Calculate error metrics against the actual left-out dataset
    actual_aps = aps_left_out.iloc[1:].astype(float).to_numpy()
    
    absolute_errors = np.abs(predicted_aps - actual_aps)
    mae_tri = np.mean(absolute_errors)
    
    squared_errors = (predicted_aps - actual_aps) ** 2
    dist_tri = np.sqrt(np.mean(squared_errors))


    return mae_nearest, dist_nearest, mae_tri, dist_tri
 

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




if __name__ == '__main__':

    # Specify the meta-feature configuration to test [45, 75, or 137]
    num_meta_ftrs = 137


    # Fetches meta-features and APS data. Returns two dataframes containing the APS vectors and the Meta Features
    FetchData = FetchAPSFeatureData(metric = 'ndcg@5', num_features= num_meta_ftrs)
    aps_dataset, feature_dataset = FetchData.get_data()


    aps_vectors_M = aps_dataset
    feature_dataset_M = feature_dataset

    baseline_errors = pd.read_csv(f'baseline_results/baseline_errors_{num_meta_ftrs}.csv')

    for i in range(10):
        baseline_errors[f'model{i}_mae_nearest'] = np.nan
        baseline_errors[f'model{i}_dist_nearest'] = np.nan
        baseline_errors[f'model{i}_mae_tri'] = np.nan
        baseline_errors[f'model{i}_dist_tri'] = np.nan




    mae = []
    for run in range(10):
        for idx in range(66):


            weights_path = f'weights/{num_meta_ftrs}_meta_features/training_run_{run}/APS_SiamNet_weights_{idx}.pth'
            aps_vec, feature_data_raw, aps_left_out, feature_left_out_raw = leave_one_out(aps_vectors_M, feature_dataset_M, index= idx)


            feature_data = feature_data_raw.copy()
            feature_left_out = feature_left_out_raw.copy()

            # Scale the training and test meta features 
            scaler = StandardScaler(with_mean=True, with_std=True)
            feature_data.iloc[:, 1:] = scaler.fit_transform(feature_data.iloc[:, 1:].to_numpy())

            test_features_scaled = scaler.transform(feature_left_out.iloc[1:].to_numpy().reshape(1, -1))
            feature_left_out.iloc[1:] = test_features_scaled.flatten()

            # Pass scaled data into prediction function
            errors = make_prediction(aps_vec, feature_data, aps_left_out, feature_left_out, weights_path)
            baseline_errors.loc[idx, f'model{run}_mae_nearest'] = errors[0]
            baseline_errors.loc[idx, f'model{run}_dist_nearest'] = errors[1]
            baseline_errors.loc[idx, f'model{run}_mae_tri'] = errors[2]
            baseline_errors.loc[idx, f'model{run}_dist_tri'] = errors[3]
            mae.append(errors[0])

        
        print(baseline_errors)
        print("\n",baseline_errors.mean(numeric_only= True)[f'model{run}_mae_nearest'], '\t', baseline_errors.mean(numeric_only= True)[f'model{run}_mae_tri'])


    baseline_errors.to_csv(f'baseline_results/snn_baseline_errors_{num_meta_ftrs}.csv')
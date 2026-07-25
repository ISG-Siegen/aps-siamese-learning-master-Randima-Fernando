import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from  utils import sort_by_distance, matches_closes_nots, distance_rmse_mae
from sklearn.preprocessing import StandardScaler
from aps_data_handling import EmbeddingNet, SiameseNet, leave_one_out, FetchAPSFeatureData
from scipy.optimize import least_squares
import getpass
import sys



def get_embedded_set(aps_vectors, feature_dataset, aps_left_out, feature_left_out, weights_path):

    dataset_size = len(aps_vectors)
    input_features = len(feature_dataset.columns[1:])

    device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )



    model = SiameseNet(EmbeddingNet(mode= 'leaky5', input_ftrs= input_features)).to(device)
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
    k_anchors = min(2, len(embedded_sorted)) 
    
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
    # nrmse = rmse / np.mean(actual_aps) if np.mean(actual_aps) != 0 else rmse



    return mae_nearest, dist_nearest, mae_tri, dist_tri
 




if __name__ == '__main__':


    # Fetches data from postgres Database. Returns two dataframes containing the APS vectors and the Meta Features
    FetchData = FetchAPSFeatureData('ndcg@5')
    aps_dataset, feature_dataset = FetchData.get_data()


    aps_vectors_M = aps_dataset

    baseline_errors = pd.read_csv('baseline_errors.csv')

    for i in range(10):
        baseline_errors[f'model{i}_mae_nearest'] = np.nan
        baseline_errors[f'model{i}_dist_nearest'] = np.nan
        baseline_errors[f'model{i}_mae_tri'] = np.nan
        baseline_errors[f'model{i}_dist_tri'] = np.nan



    # Scaling the features
    feature_data = feature_dataset.iloc[:,1:].to_numpy()
    scaler = StandardScaler(with_mean= True, with_std= True)
    scaled_data = scaler.fit_transform(feature_data)

    feature_vectors = pd.DataFrame(scaled_data, columns= feature_dataset.columns.tolist()[1:])
    feature_vectors.insert(0, column='dataset' ,value= feature_dataset['dataset'].to_list())
    feature_dataset_M = feature_vectors


    mae = []
    for run in range(10):
        for idx in range(66):
            weights_path = f'weights/137_meta_features/training_run_{run}/APS_SiamNet_weights_{idx}.pth'
            aps_vec, feature_data, aps_left_out, feature_left_out = leave_one_out(aps_vectors_M, feature_dataset_M, index= idx)

            # res = test_left_out(aps_vec, feature_data, aps_left_out, feature_left_out, 66, weights_path)
            # print(res)
            errors = get_embedded_set(aps_vec, feature_data, aps_left_out, feature_left_out, weights_path)
            baseline_errors.loc[idx, f'model{run}_mae_nearest'] = errors[0]
            baseline_errors.loc[idx, f'model{run}_dist_nearest'] = errors[1]
            baseline_errors.loc[idx, f'model{run}_mae_tri'] = errors[2]
            baseline_errors.loc[idx, f'model{run}_dist_tri'] = errors[3]
            mae.append(errors[0])
        
        
        print(baseline_errors)
        print("\n",baseline_errors.mean(numeric_only= True)[f'model{run}_mae_nearest'])
        print(baseline_errors.mean(numeric_only= True)[f'model{run}_mae_tri'])

    baseline_errors.to_csv('final_baseline_errors.csv')
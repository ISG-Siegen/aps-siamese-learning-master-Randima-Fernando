import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from  utils import sort_by_distance, matches_closes_nots, distance_rmse_mae
from sklearn.preprocessing import StandardScaler
from aps_data_handling import EmbeddingNet, SiameseNet, FetchAPSFeatureData, leave_one_out
import getpass
import sys


def check_model(aps_vectors, feature_dataset, weights_path):
    dataset_size = len(aps_vectors)

    device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )

    print(f"Using {device} device")


    model = SiameseNet(EmbeddingNet(mode= 'leaky')).to(device)

    # model = EmbeddingNet().to(device)

    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model = model.to(device)

    embedding_list = []


    model.eval()
    with torch.no_grad():
        for i in range(dataset_size):
            x = torch.tensor(feature_dataset.iloc[i,1:].astype(float).to_numpy(), dtype= torch.float32).unsqueeze(0).to(device)

            z, _ = model(x, x)
            # z = model(x)

            embedding_list.append(z.squeeze(0).to('cpu').tolist())


    # Defines embedded vectors dataframe containing the low-dim embedding of each APS vector
    embedded_vectors = pd.DataFrame(embedding_list, columns= [i+1 for i in range(10)])
    embedded_vectors.insert(0, column='dataset' ,value= aps_vectors['dataset'])



    # Following block sorts the embedded vectors and aps vectors according to distance, and checks whether the embedding matches the rank of the sorted aps or not
    matches, closes, nots = 0, 0, 0
    needed_range = 5

    for i in range(dataset_size):
        m, c, n = matches_closes_nots((sort_by_distance(aps_vectors, aps_vectors.iloc[i])), sort_by_distance(embedded_vectors, embedded_vectors.iloc[i]), needed_range)

        matches += m
        closes += c
        nots += n


    print(f"Total matches: {matches}/{needed_range*dataset_size}  \nTotal closes: {closes}/{needed_range*dataset_size}  \nTotal nots: {nots}/{needed_range*dataset_size}")



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


    model = SiameseNet(EmbeddingNet(mode= 'leaky5', input_ftrs= input_features_num)).to(device)
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
    _, rmse_10, mae_10 = distance_rmse_mae(aps_sorted, embedded_sorted, 10)
    
    nrmse = (rmse/aps_sorted['Distance'].mean())*100
    nmae_mean = (mae /aps_sorted['Distance'].mean())*100
    nmae_range = (mae /26)*100


    # print(pd.concat([aps_sorted.iloc[:13,:2], embedded_sorted.iloc[:13,:2]], axis= 1))
    # print('Mean: ',aps_sorted['Distance'].mean())
    # print("The RMSE: ", rmse)
    # print('NRMSE: ', (rmse/aps_sorted['Distance'].mean())*100 )

    return [rmse, nrmse, mae, nmae_mean, nmae_range]


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


    # Closest APS vec and closes Embedded vec compared
    closest_dataset_aps = aps_sorted.iloc[0].copy()
    closest_dataset_embedded = embedded_sorted.set_index('dataset', drop=False).loc[closest_dataset_aps['dataset']].reset_index(drop= True)
    closest_dataset_predicted = embedded_sorted.iloc[0].reset_index(drop=True).copy()
    closest_5datasets_predicted = embedded_sorted.iloc[0, 2:]

    # print(closest_5datasets_predicted.reset_index(drop= True))
    # print(closest_dataset_embedded.iloc[2:].reset_index(drop= True))
    # print(closest_dataset_predicted.iloc[2:].reset_index(drop= True))
    

    # print(closest_dataset_embedded)
    # print(closest_dataset_predicted)
    # error = closest_dataset_embedded.iloc[2:] - closest_dataset_predicted.iloc[2:]
    error = closest_dataset_embedded.iloc[2:].reset_index(drop= True) - closest_5datasets_predicted.reset_index(drop= True)
    mae =  np.mean(np.absolute(np.array(error)))

    dist = np.linalg.norm(error)

    lo_dataset = aps_left_out['dataset']
    # print(f'{lo_dataset} : {mae}')


    # _, rmse, mae = distance_rmse_mae(aps_sorted, embedded_sorted, 66)

    return mae, dist
 




if __name__ == '__main__':
    weights_path = 'weights/traininrun_8_weights/APS_SiamNet_weights_1.pth'
    # weights_path = 'weights/auto_train/APS_SiamNet_weights_1.pth'


    # Fetches data from postgres Database. Returns two dataframes containing the APS vectors and the Meta Features
    pwd = 'SlsFux11'
    FetchData = FetchAPSFeatureData('ndcg@5')
    aps_dataset, feature_dataset = FetchData.get_data()


    aps_vectors_M = aps_dataset

    baseline_errors = pd.read_csv('baseline_errors.csv')
    baseline_errors['model_mae'] = np.nan
    baseline_errors['model_dist'] = np.nan



    # Scaling the features
    feature_data = feature_dataset.iloc[:,1:].to_numpy()
    scaler = StandardScaler(with_mean= True, with_std= True)
    scaled_data = scaler.fit_transform(feature_data)

    feature_vectors = pd.DataFrame(scaled_data, columns= feature_dataset.columns.tolist()[1:])
    feature_vectors.insert(0, column='dataset' ,value= feature_dataset['dataset'].to_list())
    feature_dataset_M = feature_vectors


    mae = []
    for idx in range(66):
        weights_path = f'weights/traininrun_8_weights/APS_SiamNet_weights_{idx}.pth'
        aps_vec, feature_data, aps_left_out, feature_left_out = leave_one_out(aps_vectors_M, feature_dataset_M, index= idx)

        # res = test_left_out(aps_vec, feature_data, aps_left_out, feature_left_out, 66, weights_path)
        # print(res)
        current_mae, dist = get_embedded_set(aps_vec, feature_data, aps_left_out, feature_left_out, weights_path)
        baseline_errors.loc[idx, 'model_mae'] = current_mae
        baseline_errors.loc[idx, 'model_dist'] = dist
        mae.append(current_mae)
    
    
    print(baseline_errors)
    baseline_errors.to_csv('final_baseline_errors.csv')
    print("\n\n", baseline_errors.mean(numeric_only= True))


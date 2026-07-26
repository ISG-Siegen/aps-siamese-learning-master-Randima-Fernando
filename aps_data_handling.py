import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import distance
import sys





class FetchAPSFeatureData():
    """ 
    The following fetches the relevant data from the Postgres database and returns the datasets as pandas dataframes. 
    Primarily returns the APS dataset and the relevant Metafeatures of said dataset. 
    Constructor arguments require the metric from the APS to be extracted.
    """


    def __init__(self, metric = 'ndcg@5', num_features = 137):

        # Check metric and number of features
        available_metrics = ['ndcg@5', 'ndcg@10']
        available_ftr_configs = [137, 75, 45]

        if metric not in available_metrics:
            raise ValueError(f"Invalid input. Choose from: {available_metrics}")

        if num_features not in available_ftr_configs:
            raise ValueError(f"Invalid input. Choose from: {available_ftr_configs}")


        # Load features 

        if num_features == 137:
            features_path = "Dataset_Meta_Features/Dataset_Meta_features.csv"
        elif num_features == 75:
            features_path = "Dataset_Meta_Features/75_Dataset_Meta_features.csv"
        else:
            features_path = "Dataset_Meta_Features/45_Dataset_Meta_features.csv"
        

        features_df = pd.read_csv(features_path).iloc[:]                                    # Dataframe of features

        
        if metric == 'ndcg@5':
            aps_df = pd.read_csv('APS_Data/APS_Complete.csv'                                # Returns dataframe of ndcg5 with algorithms as columns and datasets as rows
                                 ).set_index('dataset', drop = 'False')                     # Changes row indices to column 'dataset'
        

        
        mask = features_df.dropna(subset= ['num_users'])['dataset'].to_list()               # Drops datasets with missing features
        

        self.aps_dataset_df = aps_df.loc[mask].reset_index(drop= False)
        self.meta_features_dataset_df = features_df.dropna(subset= ['num_users']).reset_index(drop= True)

    
    def get_data(self):
        return self.aps_dataset_df, self.meta_features_dataset_df




class APSFeatureDataset(Dataset):
    """ 
    The following Dataset should return upon indexing the meta features for two datasets and the corresponding 
    Euclidean distance between the two in the Algorithm Performance Space
    """


    def __init__(self, aps_vectors, feature_vectors):
        super().__init__()
        distance_aps_vec = []                                                           # Temp list to store distances between APS vectors and the vectors themselves
        dataset_size = len(aps_vectors)
        self.num_aps_dimensions = len(aps_vectors.iloc[0, 1:])                          # Number of dimensions of an APS vector
        self.num_ftrs_dimensions = len(feature_vectors.iloc[0, 1:])                     # Number of dimensions of a meta-feature vector

        for j in range(dataset_size):
            df_aps_dist = pd.DataFrame({'dataset': aps_vectors.loc[:(dataset_size-1-j),'dataset']})             # Initializes dataframe to store distances between APS vectors
            temp_features = feature_vectors.iloc[:(dataset_size-j)].copy()
            selected_dataset = j
            aps_distances = []

            for i in range(dataset_size-j):
                aps_vec1 = aps_vectors.iloc[selected_dataset, 1:].astype(float).to_numpy()
                aps_vec2 = aps_vectors.iloc[i, 1:].astype(float).to_numpy()

                aps_dist = distance.euclidean(aps_vec1, aps_vec2)

                aps_distances.append(aps_dist)


            df_aps_dist['Distance'] = aps_distances                                        # Assigns distance data to dataframe


            # Makes a sorted list of APS vectors corresponding to a sorted distance dataframe and concatenates two
            aps_dist_sorted = df_aps_dist.sort_values('Distance', ascending= True).reset_index(drop= True)
            temp_features = temp_features.set_index("dataset", drop= False)
            sorted_features = temp_features.loc[aps_dist_sorted['dataset'].to_list()].reset_index(drop= True)
            temp_features = temp_features.reset_index(drop= True)


            # Includes the APS vector of the base/anchor dataset to the main dataframe. Hence column 0-1: [Dataset, Distance] column 2-27:[Anchor] column 28:53: [Other]
            repeat_data = feature_vectors.iloc[[selected_dataset], 1:]
            temp_anchor_df = pd.DataFrame(np.repeat(repeat_data.values, (dataset_size - j), axis=0), columns=repeat_data.columns)
            dist_aps_df = pd.concat([aps_dist_sorted, temp_anchor_df, sorted_features.iloc[:,1:]], axis=1)


            distance_aps_vec.append(dist_aps_df.iloc[:])                # Includes each df to a list to be later concatenated



        self.dist_and_features = pd.concat(distance_aps_vec, axis = 0, ignore_index= True)

        

    
    def __len__(self):
        return self.dist_and_features.__len__()

    def __getitem__(self, index):
        
        # Row slices are accordingly converted to numpy to be then passed as torch values
        Anc = self.dist_and_features.iloc[index, 2:(self.num_ftrs_dimensions + 2)].astype(float).to_numpy()
        Pos = self.dist_and_features.iloc[index, (self.num_ftrs_dimensions + 2):].astype(float).to_numpy()
        dist = self.dist_and_features.iloc[index, 1].item()

        return torch.tensor(Anc, dtype= torch.float32), torch.tensor(Pos, dtype= torch.float32), torch.tensor(dist, dtype= torch.float32)




def leave_one_out(aps_df, features_df, index):
    left_out_aps = aps_df.iloc[index]
    left_out_features = features_df.iloc[index]
    aps_df = aps_df.drop(index).reset_index(drop = True)
    features_df = features_df.drop(index).reset_index(drop = True)

    return aps_df, features_df, left_out_aps, left_out_features





# class APSFeatureDataset(Dataset):
#     """ 
#     The following Dataset should return upon indexing the meta features for two datasets and the corresponding 
#     Euclidean distance between the two in the Algorithm Performance Space
#     """


#     def __init__(self, aps_vectors, feature_vectors, needed_range = 20):
#         super().__init__()
#         distance_aps_vec = []                                                           # Temp list to store distances between APS vectors and the vectors themselves
#         dataset_size = len(aps_vectors)
#         self.num_aps_dimensions = len(aps_vectors.iloc[0, 1:])
#         self.num_ftrs_dimensions = len(feature_vectors.iloc[0, 1:])

#         for j in range(dataset_size):
#             df_aps_dist = pd.DataFrame({'dataset': aps_vectors['dataset']})             # Initializes dataframe to store distances between APS vectors
#             selected_dataset = j
#             aps_distances = []

#             for i in range(dataset_size):
#                 aps_vec1 = aps_vectors.iloc[selected_dataset, 1:].astype(float).to_numpy()
#                 aps_vec2 = aps_vectors.iloc[i, 1:].astype(float).to_numpy()

#                 aps_dist = distance.euclidean(aps_vec1, aps_vec2)

#                 aps_distances.append(aps_dist)


#             df_aps_dist['Distance'] = aps_distances                                        # Assigns distance data to dataframe

#             # Makes a sorted list of APS vectors corresponding to a sorted distance dataframe and concatenates two
#             aps_dist_sorted = df_aps_dist.sort_values('Distance', ascending= True).reset_index(drop= True)
#             feature_vectors = feature_vectors.set_index("dataset", drop= False)
#             sorted_features = feature_vectors.loc[aps_dist_sorted['dataset'].to_list()].reset_index(drop= True)
#             feature_vectors = feature_vectors.reset_index(drop= True)


#             # Includes the APS vector of the base/anchor dataset to the main dataframe. Hence column 0-1: [Dataset, Distance] column 2-27:[Anchor] column 28:53: [Other]
#             repeat_data = feature_vectors.iloc[[selected_dataset], 1:]
#             temp_anchor_df = pd.DataFrame(np.repeat(repeat_data.values, dataset_size, axis=0), columns=repeat_data.columns)
#             dist_aps_df = pd.concat([aps_dist_sorted, temp_anchor_df, sorted_features.iloc[:,1:]], axis=1)


#             distance_aps_vec.append(dist_aps_df.iloc[:needed_range])                # Includes each df to a list to be later concatenated



#         self.dist_and_features = pd.concat(distance_aps_vec, axis = 0, ignore_index= True)

        

    
#     def __len__(self):
#         return self.dist_and_features.__len__()

#     def __getitem__(self, index):
        
#         # Row slices are accordingly converted to numpy to be then passed as torch values
#         Anc = self.dist_and_features.iloc[index, 2:(self.num_ftrs_dimensions + 2)].astype(float).to_numpy()
#         Pos = self.dist_and_features.iloc[index, (self.num_ftrs_dimensions + 2):].astype(float).to_numpy()
#         dist = self.dist_and_features.iloc[index, 1].item()

#         return torch.tensor(Anc, dtype= torch.float32), torch.tensor(Pos, dtype= torch.float32), torch.tensor(dist, dtype= torch.float32)


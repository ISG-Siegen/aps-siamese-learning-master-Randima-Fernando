import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader
from utils import heatmap, multi_plot
from scipy.spatial import distance
from sqlalchemy import create_engine




class FetchAPS_FeatureData():
    """ 
    The following fetches the relevant data from the Postgres database and returns the datasets as pandas dataframes. 
    Primarily returns the APS dataset and the relevant Metafeatures of said dataset. 
    Constructor arguments require the password for the postgres database, and the required metric from the APS to be extracted.
    """


    def __init__(self, password, metric, add_features = True):

        # Check metric
        available_metrics = ['ndcg@5', 'ndcg@10', 'ndcg@5_26']

        # Include additional features path and make dataframe
        add_features_path = 'Datasets/filled_df.csv'
        add_features_df = pd.read_csv(add_features_path).iloc[:,5:]

        if metric not in available_metrics:
            raise ValueError(f"Invalid input. Choose from: {available_metrics}")

        # Initializes password and engine for fetching data from postgres database
        self.password = password
        self.engine = create_engine(f"postgresql://randima:{self.password}@localhost:5432/movielens")

        # Queries for accessing database

        query_complete_datasets = """
        SELECT
            performance_results.data_set_name AS dataset,
            COUNT(performance_results.data_set_name) AS dataset_freq
        FROM
            performance_results
            
        GROUP BY
            performance_results.data_set_name
        HAVING
            COUNT(performance_results.data_set_name) = 29;
        """

        query_aps_ndcg5 = """
        SELECT
            performance_results.data_set_name AS dataset,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name = 'BPR') AS BPR,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'CDAE') AS CDAE,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'ConvNCF') AS ConvNCF,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'DGCF') AS DGCF,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'DiffRec') AS DiffRec,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'DMF') AS DMF,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'EASE') AS EASE,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'ENMF') AS ENMF,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'FISM') AS FISM,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'GCMC') AS GCMC,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'ItemKNN') AS ItemKNN,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'LDiffRec') AS LDiffRec,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'LightGCN') AS LightGCN,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'LINE') AS LINE_alg,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'MacridVAE') AS MacridVAE,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'MultiDAE') AS MultiDAE,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'MultiVAE') AS MultiVAE,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NAIS') AS NAIS,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NCEPLRec') AS NCEPLRec,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NCL') AS NCL,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NeuMF') AS NeuMF,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NGCF') AS NGCF,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NNCF') AS NNCF,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'Pop') AS Pop,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'Random') AS Random,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'RecVAE') AS RecVAE,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'SGL') AS SGL,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'SimpleX') AS SimpleX,
            SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'SpectralCF') AS SpectralCF

        FROM
            performance_results
        GROUP BY
            performance_results.data_set_name;
        """
        query_aps_ndcg5_26 = """
        WITH pivoted_data AS(
            SELECT
                performance_results.data_set_name AS dataset,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name = 'BPR') AS BPR,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'CDAE') AS CDAE,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'ConvNCF') AS ConvNCF,
                --SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'DGCF') AS DGCF,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'DiffRec') AS DiffRec,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'DMF') AS DMF,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'ENMF') AS ENMF,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'FISM') AS FISM,
                -- SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'GCMC') AS GCMC,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'ItemKNN') AS ItemKNN,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'LDiffRec') AS LDiffRec,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'LightGCN') AS LightGCN,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'LINE') AS LINE_alg,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'MacridVAE') AS MacridVAE,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'MultiDAE') AS MultiDAE,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'MultiVAE') AS MultiVAE,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NAIS') AS NAIS,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NCEPLRec') AS NCEPLRec,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NCL') AS NCL,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NeuMF') AS NeuMF,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NGCF') AS NGCF,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NNCF') AS NNCF,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'Pop') AS Pop,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'Random') AS Random,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'RecVAE') AS RecVAE,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'SGL') AS SGL,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'SimpleX') AS SimpleX,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'SpectralCF') AS SpectralCF

            FROM
                performance_results
            GROUP BY
                performance_results.data_set_name
        )
        SELECT
            ROW_NUMBER() OVER() AS idx,
            *FROM pivoted_data
        WHERE bpr IS NOT NULL
        AND cdae IS NOT NULL
        AND convncf IS NOT NULL
        -- AND dgcf IS NOT NULL
        AND diffrec IS NOT NULL
        AND dmf IS NOT NULL
        AND enmf IS NOT NULL
        AND fism IS NOT NULL
        -- AND gcmc IS NOT NULL
        AND itemknn IS NOT NULL
        AND ldiffrec IS NOT NULL
        AND lightgcn IS NOT NULL
        AND line_alg IS NOT NULL
        AND macridvae IS NOT NULL
        AND multidae IS NOT NULL
        AND multivae IS NOT NULL
        AND nais IS NOT NULL
        AND nceplrec IS NOT NULL
        AND ncl IS NOT NULL
        AND neumf IS NOT NULL
        AND ngcf IS NOT NULL
        AND nncf IS NOT NULL
        AND pop IS NOT NULL
        AND random IS NOT NULL
        AND recvae IS NOT NULL
        AND sgl IS NOT NULL
        AND simplex IS NOT NULL
        AND spectralcf IS NOT NULL;

        """


        query_aps_ndcg10 ="""
        SELECT
            performance_results.data_set_name AS dataset,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name = 'BPR') AS BPR,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'CDAE') AS CDAE,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'ConvNCF') AS ConvNCF,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'DGCF') AS DGCF,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'DiffRec') AS DiffRec,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'DMF') AS DMF,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'EASE') AS EASE,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'ENMF') AS ENMF,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'FISM') AS FISM,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'GCMC') AS GCMC,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'ItemKNN') AS ItemKNN,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'LDiffRec') AS LDiffRec,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'LightGCN') AS LightGCN,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'LINE') AS LINE_alg,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'MacridVAE') AS MacridVAE,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'MultiDAE') AS MultiDAE,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'MultiVAE') AS MultiVAE,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'NAIS') AS NAIS,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'NCEPLRec') AS NCEPLRec,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'NCL') AS NCL,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'NeuMF') AS NeuMF,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'NGCF') AS NGCF,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'NNCF') AS NNCF,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'Pop') AS Pop,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'Random') AS Random,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'RecVAE') AS RecVAE,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'SGL') AS SGL,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'SimpleX') AS SimpleX,
            SUM(performance_results.ndcg10_avg) FILTER (WHERE performance_results.algorithm_name  = 'SpectralCF') AS SpectralCF

        FROM
            performance_results
        GROUP BY
            performance_results.data_set_name;
        """

        query_features = """
        SELECT
            aps_merged.data_set_name AS dataset,
            AVG(aps_merged.num_users)::INT AS num_users,
            AVG(aps_merged.num_items)::INT AS num_items,
            AVG(aps_merged.num_interactions)::INT AS num_inter,
            ROUND(AVG(aps_merged.density),6) AS density,
            ROUND(AVG(aps_merged.user_item_ratio),6) AS u_i_ratio,
            ROUND(AVG(aps_merged.item_user_ratio),6) AS i_u_ratio,
            AVG(aps_merged.highest_num_rating_by_single_user)::INT AS highest_num_rating_u,
            AVG(aps_merged.lowest_num_rating_by_single_user)::INT AS lowest_num_rating_u,
            AVG(aps_merged.highest_num_rating_on_single_item)::INT AS highest_num_rating_i,
            AVG(aps_merged.lowest_num_rating_on_single_item)::INT AS lowest_num_rating_i,
            ROUND(AVG(aps_merged.mean_num_ratings_by_user),6) AS mean_num_rating_u,
            ROUND(AVG(aps_merged.mean_num_ratings_on_item),6) AS mean_num_rating_i
        FROM
            aps_merged
        GROUP BY
            aps_merged.data_set_name;
        """

        complete_datasets_df = pd.read_sql(query_complete_datasets, self.engine)            # Gets a dataframe containing all datasets with complete APS vectors
        features_df = pd.read_sql(query_features,                                           # Returns dataframe of dataset meta features
                                self.engine).set_index("dataset", drop= False)
        
        if metric == 'ndcg@5':
            aps_df = pd.read_sql(query_aps_ndcg5,                                           # Returns dataframe of ndcg5 with algorithms as columns and datasets as rows
                        self.engine).set_index("dataset", drop= False)                      # Changes row indices to column 'dataset'
        
        if metric == 'ndcg@10':
            aps_df = pd.read_sql(query_aps_ndcg10,                                          # Returns dataframe of ndcg10 with algorithms as columns and datasets as rows
                        self.engine).set_index("dataset", drop= False)                      # Changes row indices to column 'dataset'
        
        mask = complete_datasets_df['dataset'].to_list()                                    # This will be used to index all the complete vector datasets

        self.aps_dataset_df = aps_df.loc[mask].reset_index(drop= True)                      # Extracts requisite values and resets index back to numbers
        self.meta_feature_dataset_df = features_df.loc[mask].reset_index(drop= True)

        if add_features:
            self.meta_feature_dataset_df = pd.concat([self.meta_feature_dataset_df, add_features_df], axis= 1)
    
    def get_data(self):
        return self.aps_dataset_df, self.meta_feature_dataset_df



class FetchAPSFeatureDataNew():
    """ 
    The following fetches the relevant data from the Postgres database and returns the datasets as pandas dataframes. 
    Primarily returns the APS dataset and the relevant Metafeatures of said dataset. 
    Constructor arguments require the password for the postgres database, and the required metric from the APS to be extracted.
    """


    def __init__(self, password, metric):

        # Check metric
        available_metrics = ['ndcg@5', 'ndcg@10']

        # Include additional features path and make dataframe
        features_path = "Datasets/Updated_Features/feature_data.csv"
        

        if metric not in available_metrics:
            raise ValueError(f"Invalid input. Choose from: {available_metrics}")

        # Initializes password and engine for fetching data from postgres database
        self.password = password
        self.engine = create_engine(f"postgresql://randima:{self.password}@localhost:5432/coldstart_aps")

        # Queries for accessing database

        query_aps_ndcg5 = """
        WITH pivoted_data AS(
            SELECT
                performance_results.data_set_name AS dataset,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name = 'BPR') AS BPR,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'CDAE') AS CDAE,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'ConvNCF') AS ConvNCF,
                --SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'DGCF') AS DGCF,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'DiffRec') AS DiffRec,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'DMF') AS DMF,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'ENMF') AS ENMF,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'FISM') AS FISM,
                -- SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'GCMC') AS GCMC,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'ItemKNN') AS ItemKNN,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'LDiffRec') AS LDiffRec,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'LightGCN') AS LightGCN,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'LINE') AS LINE_alg,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'MacridVAE') AS MacridVAE,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'MultiDAE') AS MultiDAE,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'MultiVAE') AS MultiVAE,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NAIS') AS NAIS,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NCEPLRec') AS NCEPLRec,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NCL') AS NCL,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NeuMF') AS NeuMF,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NGCF') AS NGCF,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'NNCF') AS NNCF,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'Pop') AS Pop,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'Random') AS Random,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'RecVAE') AS RecVAE,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'SGL') AS SGL,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'SimpleX') AS SimpleX,
                SUM(performance_results.ndcg5_avg) FILTER (WHERE performance_results.algorithm_name  = 'SpectralCF') AS SpectralCF

            FROM
                performance_results
            GROUP BY
                performance_results.data_set_name
        )
        SELECT
            *FROM pivoted_data
        WHERE bpr IS NOT NULL
        AND cdae IS NOT NULL
        AND convncf IS NOT NULL
        -- AND dgcf IS NOT NULL
        AND diffrec IS NOT NULL
        AND dmf IS NOT NULL
        AND enmf IS NOT NULL
        AND fism IS NOT NULL
        -- AND gcmc IS NOT NULL
        AND itemknn IS NOT NULL
        AND ldiffrec IS NOT NULL
        AND lightgcn IS NOT NULL
        AND line_alg IS NOT NULL
        AND macridvae IS NOT NULL
        AND multidae IS NOT NULL
        AND multivae IS NOT NULL
        AND nais IS NOT NULL
        AND nceplrec IS NOT NULL
        AND ncl IS NOT NULL
        AND neumf IS NOT NULL
        AND ngcf IS NOT NULL
        AND nncf IS NOT NULL
        AND pop IS NOT NULL
        AND random IS NOT NULL
        AND recvae IS NOT NULL
        AND sgl IS NOT NULL
        AND simplex IS NOT NULL
        AND spectralcf IS NOT NULL;

        """

        features_df = pd.read_csv(features_path).iloc[:]                                    # Dataframe of features

        
        if metric == 'ndcg@5':
            aps_df = pd.read_sql(query_aps_ndcg5,                                           # Returns dataframe of ndcg5 with algorithms as columns and datasets as rows
                        self.engine).set_index('dataset', drop = 'False')                   # Changes row indices to column 'dataset'
        

        
        mask = features_df.dropna(subset= ['num_users'])['dataset'].to_list()               # Drops datasets with missing features
        

        self.aps_dataset_df = aps_df.loc[mask].reset_index(drop= False)
        self.meta_features_dataset_df = features_df.dropna(subset= ['num_users'])

    
    def get_data(self):
        return self.aps_dataset_df, self.meta_features_dataset_df


class APSSiameseDataset(Dataset):
    def __init__(self, features, aps_vectors):
        super().__init__()
        self.features = features
        self.aps_vectors = aps_vectors

        self.data = []                                  # Will contain a list of tuples as data

        # The following allows every combination of the dataset vectors to be created for the dataset
        for idx1, f_vec1 in enumerate(self.features):
            for idx2, f_vec2 in enumerate(self.features):
                similarity = F.pairwise_distance(self.aps_vectors[idx1], self.aps_vectors[idx2], p= 2)
                # similarity = F.cosine_similarity(self.aps_vectors[idx1], self.aps_vectors[idx2], dim= 0)
                y = torch.clamp((similarity - 0.9) / 0.1, 0.0, 1.0)
                self.data.append((f_vec1, f_vec2, self.aps_vectors[idx1], self.aps_vectors[idx2], similarity))
                # self.data.append((f_vec1, f_vec2, self.aps_vectors[idx1], y, similarity))                         # Returns the y mapped between 1 and 0

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        return self.data[index]                      # (ft1, ft2, aps1, aps2, sim)
    


class MultiDimScalingDataset(Dataset):
    """ 
    The following Dataset should return upon indexing the APS vectors of two datasets, and the corresponding 
    Eculidean distance between the two 
    """


    def __init__(self, aps_vectors, needed_range = 20):
        super().__init__()
        self.aps_vectors = aps_vectors                                                  # Includes the 29-dim APS vectors
        self.needed_range = needed_range                                                # Determines how many nearest neighbors are valid
        distance_aps_vec = []                                                           # Temp list to store distances between APS vectors and the vectors themselves


        for j in range(68):
            df_aps_dist = pd.DataFrame({'dataset': self.aps_vectors['dataset']})       # Initializes dataframe to store distances between APS vectors
            selected_dataset = j
            aps_dist = []

            for i in range(68):
                aps_vec1 = aps_vectors.iloc[selected_dataset, 1:].astype(float).to_numpy()
                aps_vec2 = aps_vectors.iloc[i, 1:].astype(float).to_numpy()

                dist_aps = distance.euclidean(aps_vec1, aps_vec2)

                aps_dist.append(dist_aps)


            df_aps_dist['Distance'] = aps_dist                                          # Assigns distance data to dataframe


            # Makes a sorted list of APS vectors corresponding to a sorted distance dataframe and concatenates two
            aps_dist_sorted = df_aps_dist.sort_values('Distance', ascending= True).reset_index(drop= True)
            self.aps_vectors = self.aps_vectors.set_index("dataset", drop= False)
            sorted_aps = self.aps_vectors.loc[aps_dist_sorted['dataset'].to_list()].reset_index(drop= True)
            self.aps_vectors = self.aps_vectors.reset_index(drop= True)

            # Includes the APS vector of the base/anchor dataset to the main dataframe. Hence column 0-1: [Dataset, Distance] column 2-30:[Anchor] column 31:60: [Other]
            repeat_data = self.aps_vectors.iloc[[selected_dataset], 1:]
            temp_anchor_df = pd.DataFrame(np.repeat(repeat_data.values, 68, axis=0), columns=repeat_data.columns)
            dist_aps_df = pd.concat([aps_dist_sorted, temp_anchor_df, sorted_aps.iloc[:,1:]], axis=1)

            distance_aps_vec.append(dist_aps_df.iloc[:self.needed_range])                # Includes each df to a list to be later concatenated


        self.dist_and_aps = pd.concat(distance_aps_vec, axis = 0, ignore_index= True)

    
    def __len__(self):
        return self.dist_and_aps.__len__()

    def __getitem__(self, index):
        
        # Row slices are accordingly converted to numpy to be then passed as torch values
        Anc = self.dist_and_aps.iloc[index, 2:31].astype(float).to_numpy()
        Pos = self.dist_and_aps.iloc[index, 31:].astype(float).to_numpy()
        dist = self.dist_and_aps.iloc[index, 1].item()

        return torch.tensor(Anc, dtype= torch.float32), torch.tensor(Pos, dtype= torch.float32), torch.tensor(dist, dtype= torch.float32)


class APSFeatureDataset(Dataset):
    """ 
    The following Dataset should return upon indexing the meta features for two datasets and the corresponding 
    Euclidean distance between the two 
    """


    def __init__(self, aps_vectors, feature_vectors, needed_range = 20):
        super().__init__()
        distance_aps_vec = []                                                           # Temp list to store distances between APS vectors and the vectors themselves
        dataset_size = len(aps_vectors)
        self.num_aps_dimensions = len(aps_vectors.iloc[0, 1:])
        self.num_ftrs_dimensions = len(feature_vectors.iloc[0, 1:])

        for j in range(dataset_size):
            df_aps_dist = pd.DataFrame({'dataset': aps_vectors['dataset']})             # Initializes dataframe to store distances between APS vectors
            selected_dataset = j
            aps_distances = []

            for i in range(dataset_size):
                aps_vec1 = aps_vectors.iloc[selected_dataset, 1:].astype(float).to_numpy()
                aps_vec2 = aps_vectors.iloc[i, 1:].astype(float).to_numpy()

                aps_dist = distance.euclidean(aps_vec1, aps_vec2)

                aps_distances.append(aps_dist)


            df_aps_dist['Distance'] = aps_distances                                        # Assigns distance data to dataframe


            # Makes a sorted list of APS vectors corresponding to a sorted distance dataframe and concatenates two
            aps_dist_sorted = df_aps_dist.sort_values('Distance', ascending= True).reset_index(drop= True)
            feature_vectors = feature_vectors.set_index("dataset", drop= False)
            sorted_features = feature_vectors.loc[aps_dist_sorted['dataset'].to_list()].reset_index(drop= True)
            feature_vectors = feature_vectors.reset_index(drop= True)

            # Includes the APS vector of the base/anchor dataset to the main dataframe. Hence column 0-1: [Dataset, Distance] column 2-30:[Anchor] column 31:60: [Other]
            repeat_data = feature_vectors.iloc[[selected_dataset], 1:]
            temp_anchor_df = pd.DataFrame(np.repeat(repeat_data.values, dataset_size, axis=0), columns=repeat_data.columns)
            dist_aps_df = pd.concat([aps_dist_sorted, temp_anchor_df, sorted_features.iloc[:,1:]], axis=1)

            distance_aps_vec.append(dist_aps_df.iloc[:needed_range])                # Includes each df to a list to be later concatenated


        self.dist_and_features = pd.concat(distance_aps_vec, axis = 0, ignore_index= True)

    
    def __len__(self):
        return self.dist_and_features.__len__()

    def __getitem__(self, index):
        
        # Row slices are accordingly converted to numpy to be then passed as torch values
        Anc = self.dist_and_features.iloc[index, 2:(self.num_ftrs_dimensions + 2)].astype(float).to_numpy()
        Pos = self.dist_and_features.iloc[index, (self.num_ftrs_dimensions + 2):].astype(float).to_numpy()
        dist = self.dist_and_features.iloc[index, 1].item()

        return torch.tensor(Anc, dtype= torch.float32), torch.tensor(Pos, dtype= torch.float32), torch.tensor(dist, dtype= torch.float32)


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

        if mode == 'mds':
            self.emb_net = nn.Sequential(
                nn.Linear(in_features= input_ftrs, out_features= 128),
                nn.ReLU(),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.Linear(32, 10)
            )
        
        elif mode == 'leaky':
            self.emb_net = nn.Sequential(
                nn.Linear(in_features= input_ftrs, out_features= 128),
                nn.LeakyReLU(),
                nn.Linear(128, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 12)
            )
        
        elif mode == 'leaky2':
            self.emb_net = nn.Sequential(
                nn.Linear(in_features= input_ftrs, out_features= 128),
                nn.ReLU(),
                nn.Linear(128, 12)
            )

        elif mode == 'leaky3':
            self.emb_net = nn.Sequential(
            nn.Linear(in_features= input_ftrs, out_features= 512),
            nn.LeakyReLU(),
            nn.Linear(512, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 16)
        )

        elif mode == 'leaky4':
            self.emb_net = nn.Sequential(
            nn.Linear(in_features= input_ftrs, out_features= 512),
            nn.LeakyReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 26)
        )
            
        elif mode == 'leaky5':
            self.emb_net = nn.Sequential(
            nn.Linear(in_features= input_ftrs, out_features= 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 16)
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
        # return nn.functional.normalize(x, p=2, dim=1)
        return x


class GatedExpertEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=3):
        super().__init__()
        self.num_experts = num_experts
        
        # 1. The Gating Network (The Router)
        # It looks at the features and decides which expert to use.
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_experts),
            nn.Softmax(dim=-1) # Output adds up to 1.0 (e.g., [0.8, 0.1, 0.1])
        )
        
        # 2. The Experts (Parallel Sub-networks)
        # Each expert specializes in a different type of dataset.
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 32),
                nn.LeakyReLU()
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        # Determine routing weights
        gate_weights = self.gate(x) # Shape: [batch, num_experts]
        # print(gate_weights)
        # Collect outputs from all experts
        # We use a list comprehension and stack them
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1) 
        # Shape: [batch, num_experts, hidden_dim]
        
        # Weighted sum: Multiply expert results by their gate probability
        # This "mixes" the expertise based on the dataset type
        weighted_output = torch.sum(expert_outputs * gate_weights.unsqueeze(-1), dim=1)
        
        return weighted_output


class GatedSiameseArm(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        # STEP 1: Feature Attention (Your "Volume Knobs")
        self.attention = FeatureAttention(input_dim)
        
        # STEP 2: Task-Aware Gating (Your "Expert Panel")
        self.gated_encoder = GatedExpertEncoder(input_dim, 128, num_experts=4)
        
        # STEP 3: Final Projection
        self.projection = nn.Linear(32, 16)

    def forward(self, x):
        # 1. Dynamic Feature Filtering
        weighted_features = self.attention(x)
        
        # 2. Routing through specialized experts
        mixed_expert_signal = self.gated_encoder(weighted_features)
        
        # 3. Final embedding for the Siamese distance calculation
        return self.projection(mixed_expert_signal)



class ResidualGatedArm(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Your previous Expert Gating logic
        self.expert_gate = GatedExpertEncoder(input_dim, hidden_dim)
        
        # The "Shortcut" - resizes raw input to match expert output
        self.shortcut = nn.Linear(input_dim, 16)

    def forward(self, x):
        # Path A: The complex logic (The Highway)
        identity = self.shortcut(x) 
        
        # Path B: The Experts (The specialized "corrections")
        experts_out = self.expert_gate(x)
        
        # The Residual Connection: F(x) + x
        return torch.relu(experts_out + identity)


class SiameseNet(nn.Module):
    def __init__(self, emb_net):
        super().__init__()
        self.emb_net = emb_net
    
    def forward(self, ft1, ft2):
        z1 = self.emb_net(ft1)
        z2 = self.emb_net(ft2)

        return z1, z2
    
def leave_one_out(aps_df, features_df, index):
    left_out_aps = aps_df.iloc[index]
    left_out_features = features_df.iloc[index]
    aps_df = aps_df.drop(index).reset_index(drop = True)
    features_df = features_df.drop(index).reset_index(drop = True)

    return aps_df, features_df, left_out_aps, left_out_features


def contrastive_loss(x1, x2, label, margin=2.0):
    # Calculate L2 distance
    dist = F.pairwise_distance(x1, x2, p=2)
    
    # If label=1 (same), minimize distance. 
    # If label=0 (different), maximize distance up to margin.
    loss = (label * torch.pow(dist, 2) + (1 - label) * torch.pow(torch.clamp(margin - dist, min=0.0), 2))
    
    return torch.mean(loss)



def loss_function(z1, z2, high_dim_distance):

    low_dim_distance = torch.linalg.vector_norm(z1 - z2, ord=2, dim=1)

    loss = torch.mean((low_dim_distance - high_dim_distance) ** 2)

    return loss


def stress_loss(z1, z2, high_dim_distance):

    low_dim_distance = torch.linalg.vector_norm(z1 - z2, ord=2, dim=1)

    eps = 1e-7
    high_dim_safe = high_dim_distance + eps

    error = ((high_dim_distance - low_dim_distance) ** 2) / high_dim_safe

    loss = torch.sum(error) / (torch.sum(high_dim_safe) + eps)

    return loss



class DistancePreservationLoss(nn.Module):
    def __init__(self):
        super(DistancePreservationLoss, self).__init__()

    def forward(self, low_dim_output, high_dim_input):
        # 1. Calculate pairwise distances for the original 30-D vectors
        # pdist returns a flat vector of distances between all pairs
        dist_high = torch.pdist(high_dim_input, p=2)
        
        # 2. Calculate pairwise distances for the 10-D predicted vectors
        dist_low = torch.pdist(low_dim_output, p=2)
        
        # 3. Use Mean Squared Error to minimize the difference
        # This is essentially the "Stress" concept from MDS
        loss = torch.mean((dist_high - dist_low) ** 2)
        return loss


def train(train_dataloader, model, device, loss_fn, optim, loss_data):

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

        if (batch%10) == 0:
            train_loss, current = loss.item(), batch * batch_size + len(z1)
            print(f"loss: {train_loss:>7f}  [{current:>5d}/{dataset_size:>5d}]")
            loss_data.append(train_loss)



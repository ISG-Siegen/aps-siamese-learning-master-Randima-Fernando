import pandas as pd
import numpy as np
import torch
from aps_data_handling import FetchAPSFeatureData, APSFeatureDataset, leave_one_out
from network_blocks import EmbeddingNet, SiameseNet, stress_loss, train
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from validation import test_left_out



# Fetches data from postgres Database. Returns two dataframes containing the APS vectors and the Meta Features

FetchData = FetchAPSFeatureData(metric='ndcg@5', num_features= 137)
aps_dataset, feature_dataset = FetchData.get_data()


aps_dataset_complete = aps_dataset
feature_dataset_complete = feature_dataset

dataset_size = len(aps_dataset_complete)

for training_run in range(10):
    results = []
    df_results = pd.DataFrame(columns=['dataset','Epoch', 'rmse', 'nrmse', 'mae', 'nmae_mean', 'nmae_range'])


    for idx in range(dataset_size):
    # Leave one dataset out

        aps_vectors, feature_vectors, aps_left_out, feature_left_out = leave_one_out(aps_dataset_complete, feature_dataset_complete, index= idx)


        # Scaling the features
        scaler = StandardScaler(with_mean= True, with_std= True)

        # Convert to numpy arrays
        train_features_raw = feature_vectors.iloc[:, 1:].to_numpy()
        left_out_features_raw = feature_left_out.iloc[1:].to_numpy()

        # Fit on train features
        scaled_train_features = scaler.fit_transform(train_features_raw)

        # Transform left-out features
        left_out_features_reshaped = left_out_features_raw.reshape(1, -1)
        scaled_left_out_array = scaler.transform(left_out_features_reshaped)
        scaled_left_out_flat = scaled_left_out_array.flatten()

        # Reconstruct the Dataframes and Series
        scaled_feature_vectors = pd.DataFrame(scaled_train_features, columns=feature_vectors.columns[1:])
        scaled_feature_vectors.insert(0, column='dataset', value=feature_vectors['dataset'].to_list())
        feature_vectors = scaled_feature_vectors
        
        scaled_feature_left_out = pd.Series(data=[feature_left_out.iloc[0]] + list(scaled_left_out_flat),index=feature_left_out.index)
        feature_left_out = scaled_feature_left_out



        dataset = APSFeatureDataset(aps_vectors= aps_vectors, feature_vectors= feature_vectors)

        device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps"
                    if torch.backends.mps.is_available()
                    else "cpu"
                )
        print(f"Using {device} device")


        train_dataloader = DataLoader(
            dataset,
            batch_size= 1024,
            shuffle= True,
            num_workers= 5,
            pin_memory= True
        )


        # Model declaration

        epochs = 20

        model = SiameseNet(EmbeddingNet(network_num= 5, input_ftrs= 137)).to(device)

        loss_func = stress_loss

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Lists for storing ongoing loss data and validation results
        loss_data = []
        inter_val_results_list = []


        for t in range(epochs):
            print(f"Running Epoch{t}")
            train(train_dataloader, model, device, loss_func, optimizer)

            if (t%2) == 0:
                torch.save(model.state_dict(), 'weights/auto_train/temp_weights.pth')
                weights_path = ('weights/auto_train/temp_weights.pth')

            # Calculates the validation results of current model
            inter_val_results = test_left_out(aps_vectors, feature_vectors, aps_left_out, feature_left_out, 66, weights_path)
            inter_val_results.insert(0, t)
            inter_val_results.insert(0, aps_dataset_complete.iloc[idx]['dataset'])
            inter_val_results_list.append(inter_val_results)

        

        print(pd.DataFrame(inter_val_results_list, columns= ['dataset','Epoch', 'rmse', 'nrmse', 'mae', 'nmae_mean', 'nmae_range']))
        
        df_results = pd.concat([df_results, pd.DataFrame(inter_val_results_list, columns= ['dataset','Epoch', 'rmse', 'nrmse', 'mae', 'nmae_mean', 'nmae_range'])])
        df_results.to_csv(f'training_results/validation_{training_run}.csv', index= False)
        

        torch.save(model.state_dict(), f'weights/auto_train/training_run_{training_run}/APS_SiamNet_weights_{idx}.pth')
        print("Model saved!")

        weights_path = (f'weights/auto_train/training_run_{training_run}/APS_SiamNet_weights_{idx}.pth')

        r = test_left_out(aps_vectors, feature_vectors, aps_left_out, feature_left_out, 66, weights_path)
        r.insert(0, aps_dataset_complete.iloc[idx]['dataset'])
        results.append(r)
        pd.DataFrame(results, columns= ['dataset', 'rmse', 'nrmse', 'mae', 'nmae_mean', 'nmae_range']).to_csv(f'training_results/test_results_{training_run}.csv', index = False)



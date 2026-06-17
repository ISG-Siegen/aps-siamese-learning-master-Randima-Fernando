import pandas as pd
import numpy as np
import torch
from aps_data_handling import FetchAPSFeatureData, EmbeddingNet, SiameseNet, APSFeatureDataset, stress_loss, train, leave_one_out
from torch.utils.data import Dataset, DataLoader
from  utils import multi_plot,plot_loss
from sklearn.preprocessing import StandardScaler
from inference import test_left_out
import getpass
import sys




# Fetches data from postgres Database. Returns two dataframes containing the APS vectors and the Meta Features
pwd = getpass.getpass()

FetchData = FetchAPSFeatureData(pwd, 'ndcg@5')
aps_dataset, feature_dataset = FetchData.get_data()

# Baseline features
# feature_dataset = feature_dataset.iloc[:, :41]
# feature_dataset = pd.read_csv('Datasets/Updated_Features/feature_data_small.csv')

aps_vectors_M = aps_dataset



# Scaling the features
feature_data = feature_dataset.iloc[:,1:].to_numpy()
scaler = StandardScaler(with_mean= True, with_std= True)
scaled_data = scaler.fit_transform(feature_data)

feature_vectors = pd.DataFrame(scaled_data, columns= feature_dataset.columns.tolist()[1:])
feature_vectors.insert(0, column='dataset' ,value= feature_dataset['dataset'].to_list())
feature_dataset_M = feature_vectors


dataset_size = len(aps_vectors_M)

for training_run in range(10):
    results = []
    df_results = pd.DataFrame(columns=['dataset','Epoch', 'rmse', 'nrmse', 'mae', 'nmae_mean', 'nmae_range'])


    for idx in range(dataset_size):
    # Leave one dataset out

        aps_vectors, feature_vectors, aps_left_out, feature_left_out = leave_one_out(aps_vectors_M, feature_dataset_M, index= idx)

        dataset = APSFeatureDataset(aps_vectors= aps_vectors, feature_vectors= feature_vectors, needed_range= 65)


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

        # model = SiameseNet(EmbeddingNet(mode= 'leaky2', input_ftrs= 32)).to(device)
        model = SiameseNet(EmbeddingNet(mode= 'leaky5', input_ftrs= 137)).to(device)
        # model = SiameseNet(GatedSiameseArm(input_dim= 137)).to(device)
        # model = SiameseNet(ResidualGatedArm(input_dim= 137, hidden_dim= 256)).to(device)

        loss_func = stress_loss

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        loss_data = []
        res_w = []
        # loss_rmse_data = [[], [], [], [], []]

        for t in range(epochs):
            print(f"Running Epoch{t}")
            train(train_dataloader, model, device, loss_func, optimizer, loss_data)

            if (t%2) == 0:
                torch.save(model.state_dict(), 'weights/auto_train/temp_weights.pth')
                weights_path = ('weights/auto_train/temp_weights.pth')
                w = test_left_out(aps_vectors, feature_vectors, aps_left_out, feature_left_out, 66, weights_path)
                w.insert(0, t)
                w.insert(0, aps_vectors_M.iloc[idx]['dataset'])
                res_w.append(w)

        
        print(pd.DataFrame(res_w, columns= ['dataset','Epoch', 'rmse', 'nrmse', 'mae', 'nmae_mean', 'nmae_range']))
        
        df_results = pd.concat([df_results, pd.DataFrame(res_w, columns= ['dataset','Epoch', 'rmse', 'nrmse', 'mae', 'nmae_mean', 'nmae_range'])])
        df_results.to_csv(f'training_results/debug/validation_{training_run}.csv', index= False)
        

        torch.save(model.state_dict(), f'weights/auto_train/APS_SiamNet_weights_{idx}.pth')
        print("Model saved!")

        weights_path = (f'weights/auto_train/APS_SiamNet_weights_{idx}.pth')

        r = test_left_out(aps_vectors, feature_vectors, aps_left_out, feature_left_out, 66, weights_path)
        r.insert(0, aps_vectors_M.iloc[idx]['dataset'])
        results.append(r)
        pd.DataFrame(results, columns= ['dataset', 'rmse', 'nrmse', 'mae', 'nmae_mean', 'nmae_range']).to_csv(f'training_results/debug/test_results_{training_run}.csv', index = False)



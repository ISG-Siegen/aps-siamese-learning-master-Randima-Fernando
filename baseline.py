import pandas as pd
import numpy as np
import torch
from aps_data_handling import FetchAPSFeatureData, leave_one_out
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
import sys



FetchData = FetchAPSFeatureData('ndcg@5')
aps_dataset, feature_dataset = FetchData.get_data()

aps_dataset_complete = aps_dataset


# Scaling the features
feature_data = feature_dataset.iloc[:,1:].to_numpy()
scaler = StandardScaler(with_mean= True, with_std= True)
scaled_data = scaler.fit_transform(feature_data)

feature_vectors = pd.DataFrame(scaled_data, columns= feature_dataset.columns.tolist()[1:])
feature_vectors.insert(0, column='dataset' ,value= feature_dataset['dataset'].to_list())
feature_dataset_complete = feature_vectors

baseline_errors= aps_dataset[['dataset']].copy()
baseline_errors['floor_error'] = np.nan
baseline_errors['floor_dist'] = np.nan
baseline_errors['kmeans_error'] = np.nan
baseline_errors['knn_error'] = np.nan
baseline_errors['knn_dist'] = np.nan
baseline_errors['rf_error'] = np.nan
baseline_errors['rf_dist'] = np.nan
baseline_errors['xgb_error'] = np.nan
baseline_errors['xgb_dist'] = np.nan




#
# Global Mean Basline --------------------------------
#

dataset_size = len(aps_dataset_complete)

for idx in range(dataset_size):
    aps_vectors, _, aps_left_out, _ = leave_one_out(aps_dataset_complete, feature_dataset_complete, index= idx)

    left_out_performance = aps_left_out[1:]
    mean_performance = aps_vectors.mean(numeric_only= True)
    
    error = np.linalg.norm(left_out_performance - mean_performance)
    baseline_errors.loc[idx, 'floor_dist'] = error

    error = np.mean(np.abs(left_out_performance - mean_performance))
    baseline_errors.loc[idx, 'floor_error'] = error



#
# K Means ----------------------------------
#


NUM_CLUSTERS = 5
kmeans_prediction_errors = []

for idx in range(dataset_size):
    # 1. Split using your existing leave_one_out function
    aps_vectors, feature_vectors, aps_left_out, feature_left_out = leave_one_out(
        aps_dataset_complete, feature_dataset_complete, index=idx
    )
    
    # 2. Extract arrays and scale features strictly within the fold
    X_train = feature_vectors.iloc[:, 1:].to_numpy()
    y_train = aps_vectors.iloc[:, 1:].to_numpy() # Shape: (N-1, 26)
    
    X_test = feature_left_out.to_frame().T.iloc[:, 1:].to_numpy() # Shape: (1, 137)
    y_test = aps_left_out[1:].to_numpy().reshape(26) # Shape: (26,)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Fit K-Means on the training meta-features
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42, n_init='auto')
    train_cluster_labels = kmeans.fit_predict(X_train_scaled)
    
    # 4. Compute the average APS target profile for each cluster
    cluster_profiles = {}
    for cluster_id in range(NUM_CLUSTERS):
        # Extract indices of training datasets belonging to this cluster
        indices_in_cluster = np.where(train_cluster_labels == cluster_id)[0]
        
        if len(indices_in_cluster) > 0:
            # Average 26-dim vector for this cluster
            cluster_profiles[cluster_id] = np.mean(y_train[indices_in_cluster], axis=0)
        else:
            # Fallback: If a cluster is empty (rare), use global training mean
            cluster_profiles[cluster_id] = np.mean(y_train, axis=0)
            
    # 5. Predict for the left-out dataset
    # Determine which cluster the test dataset drops into
    predicted_cluster = kmeans.predict(X_test_scaled)[0]
    kmeans_vector_prediction = cluster_profiles[predicted_cluster]
    
    # 6. Calculate direct Vector MAE
    fold_mae = np.mean(np.abs(kmeans_vector_prediction - y_test))
    kmeans_prediction_errors.append(fold_mae)
    baseline_errors.loc[idx, 'kmeans_error'] = fold_mae




#
# K Nearest Neighbor ----------------------------------
#


for idx in range(dataset_size):
    aps_vectors, feature_vectors, aps_left_out, feature_left_out = leave_one_out(aps_dataset_complete, feature_dataset_complete, index= idx)


    knn = KNeighborsRegressor(n_neighbors=5, metric= 'euclidean', weights= 'distance')
    knn.fit(feature_vectors.iloc[:,1:], aps_vectors.iloc[:,1:])
    knn_prediction = knn.predict(feature_left_out.to_frame().T.iloc[:,1:])
    knn_prediction = knn_prediction.reshape(26)
    aps_left_out = aps_left_out[1:].to_numpy()

    error = np.linalg.norm(knn_prediction - aps_left_out)
    baseline_errors.loc[idx, 'knn_dist'] = error

    error = np.mean(np.abs(knn_prediction - aps_left_out))
    baseline_errors.loc[idx, 'knn_error'] = error
    


baseline_errors.to_csv('baseline_errors.csv', index=False)
sys.exit()

#
# XGBoost Baseline and Random Forest Baseline ----------------------------------
#


for idx in range(dataset_size):
    aps_vectors, feature_vectors, aps_left_out, feature_left_out = leave_one_out(aps_dataset_complete, feature_dataset_complete, index= idx)

    X_train = feature_vectors.iloc[:, 1:].to_numpy()
    y_train = aps_vectors.iloc[:, 1:].to_numpy() # Shape: (N-1, 26)
    
    X_test = feature_left_out.to_frame().T.iloc[:, 1:].to_numpy() # Shape: (1, 137)
    y_test = aps_left_out[1:].to_numpy().reshape(26) # Shape: (26,)

    # Model A: Multioutput Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    rf_prediction = rf.predict(X_test).reshape(26)

    # Model B: Multi-output XGBoost 
    xgb = MultiOutputRegressor(XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1))
    xgb.fit(X_train, y_train)
    
    xgb_prediction = xgb.predict(X_test).reshape(26)

    # print("rf: \n",rf_prediction, "\nxgb: \n", xgb_prediction )

    error_rf = rf_prediction - y_test
    error_xgb = xgb_prediction - y_test

    rf_mae = np.mean(np.abs(error_rf))
    rf_dist = np.linalg.norm(error_rf)
    xgb_mae = np.mean(np.abs(error_xgb))
    xgb_dist = np.linalg.norm(error_xgb)

    baseline_errors.loc[idx, 'rf_error'] = rf_mae
    baseline_errors.loc[idx, 'rf_dist'] = rf_dist
    baseline_errors.loc[idx, 'xgb_error'] = xgb_mae
    baseline_errors.loc[idx, 'xgb_dist'] = xgb_dist
    
# print("KNN Error: ", baseline_errors['knn_error'].mean())
print(baseline_errors)
print(baseline_errors.mean(numeric_only= True))
baseline_errors.to_csv('baseline_errors.csv', index=False)
sys.exit()


results_mae = pd.DataFrame({})
test_num = 8
metric = 'mae'
results_mae['dataset'] = pd.read_csv(f'training_results/{test_num}th_test/test_results_0.csv')['dataset']

for i in range(10):
    results_df = pd.read_csv(f'training_results/{test_num}th_test/test_results_{i}.csv')
    results_mae[f'{metric}_{i}'] = results_df[metric]


# print(results_df)
df_sorted = results_mae.assign(row_mean= results_mae.iloc[:,:].mean(numeric_only=True,axis=1)).sort_values("row_mean")
df_row_means = df_sorted['row_mean'].copy()
# print(df_sorted[(df_sorted['row_mean'] < 0.05)].reset_index())

print(baseline_errors.join(df_row_means))


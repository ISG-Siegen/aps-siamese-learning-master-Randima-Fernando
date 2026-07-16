import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial import distance
from sklearn.metrics import ndcg_score
from scipy.stats import spearmanr
import sys


def plot_example(dataset, num):

    plt.figure(figsize=(4,2))

    image, label = dataset[num]

    img_display = image.squeeze().numpy()
    plt.imshow(img_display, cmap= 'gray')
    plt.title(f"Label: Example {num}")
    plt.axis('off')
    plt.show()


def plot_example2(img):

    plt.figure(figsize=(4,2))

    img_display = img.numpy()
    plt.imshow(img_display, cmap= 'gray')
    plt.title("Example")
    plt.axis('off')
    plt.show()


def plot_example3(img1, img2):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2))

    im1 = ax1.imshow(img1, cmap='viridis')
    ax1.set_title("Image 1")
    im2 = ax2.imshow(img1, cmap='magma')
    ax2.set_title("Image 2")

    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


def heatmap(data, title):

    plt.imshow(data, cmap="hot_r", interpolation="nearest")
    plt.colorbar(label= 'Intensity')
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.show()    


def heatmap_new(data, title, labels = []):


    fig, ax = plt.subplots(figsize=(18, 18))  # Wider figure helps with 50 names
    im = ax.imshow(data, aspect='auto')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=5)

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, rotation=0, ha='right', fontsize=5)

    plt.colorbar(im)
    fig.gca().invert_yaxis()
    fig.tight_layout()
    plt.show()

def plot_loss(loss_data, plot_title, y_name, y_lim = 3):
    plt.plot(loss_data)
    plt.xlabel("Iteration")
    plt.ylabel(y_name)
    plt.title(plot_title)
    plt.ylim(0, y_lim)
    plt.show()

def multi_plot(data1, data2, label1, label2,y_label, title):

    iter = range(1, len(data1)+1)
    plt.plot(iter, data1, label= label1)
    plt.plot(iter, data2, label= label2)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel= y_label)
    plt.title(title)
    plt.legend()
    plt.show()

def multi_plot_adv(data, labels, title, y_label ,y_lim, data_extra = None):

    # iter = range(1, len(data[0])+1)
    # iter = [int(i*2) for i in range(10)]
    iter = [int(i*2) for i in range(20)]
    

    for i in range(len(data)):
        plt.plot(iter, data[i], label= f"curr_{labels[i]}", color = 'green' )

    if data_extra is not None:
        for i in range(len(data_extra)):
            plt.plot(iter, data_extra[i], label= f"prev_{labels[i]}", color = 'red' )

    plt.xticks(iter)
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.ylim(0, y_lim)
    plt.title(title)
    plt.legend()
    plt.show()


def dataset_class(X, labels, show = True, threshold= 1.3):

    Z = linkage(X, method= 'ward')

    if show:
        plt.figure(figsize=(14, 17))
        plt.title("Hierarchical Clustering Dendrogram (Ward's Method)")
        plt.xlabel("Vector Index (or Cluster Size)")
        plt.ylabel("L2 Distance (Dissimilarity)")

        dendrogram(
            Z,
            labels= labels,
            leaf_rotation= 90,
            leaf_font_size= 8,
            color_threshold= threshold
        )

        plt.axhline(y= threshold, color='r', linestyle='--') # Visual 'cut' line
        plt.tight_layout()
        plt.show()

    max_d = threshold 
    clusters = fcluster(Z, max_d, criterion='distance')

    clusters_df = pd.DataFrame(clusters, columns= ['clusters'], index= [i for i in range(len(labels))])
    new_df = pd.DataFrame(labels, columns= ['dataset'], index= [i for i in range(len(labels))])
    

    return pd.concat([new_df, clusters_df], axis=1)



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


def matches_closes_nots(sorted_data1, sorted_data2, needed_range):
    """
    Accepts two dataframes sorted relative to a single corresponding vector, and checks whether they match,
    or the elements are within an acceptable range of each other. 
    """
    
    matches, close, nots = 0, 0, 0

    for i in range(needed_range):
        if sorted_data1.loc[i, 'dataset'] == sorted_data2.loc[i, 'dataset']:
            matches += 1
        
        elif check_if_close(sorted_data1, sorted_data2, i, 1):
            close += 1
        
        elif check_if_close(sorted_data1, sorted_data2, i, 2):
            close += 1
        
        elif check_if_close(sorted_data1, sorted_data2, i, 3):
            close +=1
        
        elif check_if_close(sorted_data1, sorted_data2, i, -1):
            close += 1
        
        elif check_if_close(sorted_data1, sorted_data2, i, -2):
            close += 1
        
        elif check_if_close(sorted_data1, sorted_data2, i, -3):
            close += 1
            
        else:
            nots += 1
    print(f"Matches: {matches}/{needed_range}   Close: {close}/{needed_range}   Nots: {nots}/{needed_range}")

    return matches, close, nots



def check_if_close(sorted_data1, sorted_data2, base, increment):

    if ((base + increment) < 0) or ((base + increment) > len(sorted_data1)):
        return False

    if sorted_data1.loc[base, 'dataset'] == sorted_data2.loc[base + increment, 'dataset']:
        return True
    
    else:
        return False
    


def calc_list_ndcg(sorted_data1, sorted_data2, k=5):
    # 1. Properly set the index so we can look up by dataset name
    sorted_data2 = sorted_data2.set_index("dataset")
    
    # 2. Initialize relevance columns
    sorted_data1['relevance'] = 0
    sorted_data2['relevance'] = 0

    # 3. Assign relevance based on ground truth order (sorted_data1)
    for i in range(k):
        # Get dataset name from the i-th row of the ground truth
        dataset_name = sorted_data1.iloc[i]['dataset']
        rel_value = k - i
        
        # Mark ground truth relevance
        sorted_data1.iloc[i, sorted_data1.columns.get_loc('relevance')] = rel_value
        
        # Mark the prediction's relevance score based on where it appears
        if dataset_name in sorted_data2.index:
            sorted_data2.loc[dataset_name, 'relevance'] = rel_value

    # 4. Prepare arrays
    y_truth = np.array([sorted_data1['relevance'].values])
    y_pred = np.array([sorted_data2['relevance'].values])

    # print(y_truth)
    # print(y_pred)

    # 5. Calculate Score
    score = ndcg_score(y_truth, y_pred, k=k)
    print(f"Score: {score:.4f}")
    return score



def calc_recall_at_k(truth_df, pred_df, k=5):
    # 1. Identify what the "True" top K datasets are
    # (Assuming truth_df is already sorted by actual relevance)
    true_relevant_set = set(truth_df.iloc[:k]['dataset'])
    
    # 2. Identify what the "Predicted" top K datasets are
    pred_top_k_set = set(pred_df.iloc[:k]['dataset'])
    
    # 3. Find the intersection (the "Hits")
    hits = len(true_relevant_set.intersection(pred_top_k_set))
    
    # 4. Calculate Recall
    recall = hits / len(true_relevant_set)
    
    print(f"Recall@{k}: {recall:.4f}")
    return recall

def calculate_ap(ground_truth, predicted_list, k=5):
    """
    Calculates Average Precision for a single ranked list.
    """

    ground_truth_set = set(ground_truth.iloc[:k]['dataset'])
    predicted_list = predicted_list.iloc[:k]['dataset'].to_list()
    hits = 0
    sum_precisions = 0
    
    # We only care about the top K positions
    for i, item in enumerate(predicted_list[:k]):
        if item in ground_truth_set:
            hits += 1
            # Precision at this specific rank i+1
            precision_at_i = hits / (i + 1)
            sum_precisions += precision_at_i
            
    # Divide by the total number of relevant items expected
    # If the ground truth is empty, return 0
    num_relevant = len(ground_truth_set)
    if num_relevant == 0:
        return 0.0
    
    print('Average Precision: ', sum_precisions / num_relevant)
    return sum_precisions / num_relevant


def calc_pearson_ranking(sorted_data1, sorted_data2):
    # 1. Properly set the index so we can look up by dataset name
    k = len(sorted_data1)
    sorted_data2 = sorted_data2.set_index("dataset")

    # 2. Initialize relevance columns
    sorted_data1['rank'] = 0
    sorted_data2['rank'] = 0

    # 3. Assign relevance based on ground truth order (sorted_data1)
    for i in range(k):
        # Get dataset name from the i-th row of the ground truth
        dataset_name = sorted_data1.iloc[i]['dataset']
        
        # Mark ground truth relevance
        sorted_data1.iloc[i, sorted_data1.columns.get_loc('rank')] = i
        
        # Mark the prediction's relevance score based on where it appears
        if dataset_name in sorted_data2.index:
            sorted_data2.loc[dataset_name, 'rank'] = i

    # 4. Prepare arrays
    y_truth = np.array([sorted_data1['rank'].values])
    y_pred = np.array([sorted_data2['rank'].values])

    print(y_truth)
    print(y_pred)
    breakpoint()

    corr, p_value = spearmanr(y_truth.flatten(), y_pred.flatten())

    # print(y_truth)
    # print(y_pred)

    # 5. Calculate Score
    print(f"Spearman Correlation: {corr:.4f}")

    return corr


def calculate_pearson_aligned(df1, df2, target_col, id_col='dataset'):
    """
    Aligns two DataFrames by a name/id column, then calculates 
    Pearson correlation for a specific numeric column.
    """
    # 1. Set the name column as index for perfect alignment
    df1_aligned = df1.set_index(id_col)
    df2_aligned = df2.set_index(id_col)
    
    # 2. Join on index to ensure we only compare rows present in both
    # (Inner join ensures alignment of the names)
    combined = df1_aligned[[target_col]].join(
        df2_aligned[[target_col]], 
        lsuffix='_1', 
        rsuffix='_2', 
        how='inner'
    )
    
    # 3. Calculate correlation between the two numeric columns
    corr = combined[f'{target_col}_1'].corr(combined[f'{target_col}_2'])

    print(corr)
    
    return corr



def distance_rmse_mae(aps, embedded, relevant_datasets):
    """
    Accepts the pandas dataframes for the sorted aps values and sorted embedded values according to their distances to the 
    left-out dataset. Compares the distances given by the APS space vs the Embedded space, to iteratively obtain the errors
    and eventually return the RMSE. 
    """

    merged = aps.merge(embedded, on='dataset', suffixes=('_1', '_2'))
    # print(merged)
    # merged.to_csv(f'merged_df.csv', index = False)
    # sys.exit()

    error = (merged['Distance_1'] - merged['Distance_2']).iloc[:relevant_datasets].tolist()

    rmse = np.sqrt(np.mean(np.array(error)**2))
    mae = np.mean(np.absolute(np.array(error)))

    return error, rmse, mae



def show_test_results_box_plots(test_num, y_limit = 100, metric = 'nmae_mean', ax_line = (10, 20, 40)):
    results_nmae = pd.DataFrame({})
    results_nmae['dataset'] = pd.read_csv(f'training_results/{test_num}th_test/test_results_0.csv')['dataset']

    for i in range(10):
        results_df = pd.read_csv(f'training_results/{test_num}th_test/test_results_{i}.csv')
        results_nmae[f'{metric}_{i}'] = results_df[metric]

    df_sorted = results_nmae.assign(row_mean= results_nmae.iloc[:,1:].mean(axis=1)).sort_values("row_mean")
    print(df_sorted['row_mean'].mean())
    print(df_sorted[(df_sorted['row_mean'] < 0.05)].reset_index())
    # print(df_sorted[(df_sorted['row_mean'] >= 10) & (df_sorted['row_mean'] < 20)].reset_index())
    # print(df_sorted[(df_sorted['row_mean'] >= 20) & (df_sorted['row_mean'] < 40)].reset_index())
    # print(df_sorted[(df_sorted['row_mean'] >= 40)].reset_index())
    # sys.exit()
    results_nmae = df_sorted.drop(columns="row_mean")


    labels = results_nmae['dataset'].to_list()
    data = []
    for i in range(len(labels)):
        data.append(results_nmae.iloc[i, 1:].astype(float).to_list())

    plt.figure(figsize=(20,10))
    plt.boxplot(x= data, tick_labels= labels, 
                patch_artist= True,
                boxprops=dict(facecolor = 'lightblue',color="blue"), 
                medianprops=dict(color="red", linewidth=2),
                flierprops=dict(marker='o', markerfacecolor='green', markersize=8),
                vert = False)


    # plt.axhline(y=ax_line[0], color='green', linestyle='--')
    # plt.axhline(y=ax_line[1], color='blue', linestyle='--')
    # plt.axhline(y=ax_line[2], color='red', linestyle='--')

    # plt.ayhline(y=ax_line[0], color='green', linestyle='--')
    plt.axvline(x=ax_line[0], color='green', linestyle='--')

    plt.xlabel('Mean Absolute Error', fontsize = 14)
    plt.xticks( ha='right', fontsize = 12)
    plt.yticks( ha='right', fontsize = 7)
    # plt.ylim(0, y_limit)
    plt.xlim(0, y_limit)
    plt.tight_layout()
    plt.savefig('my_plot.png', dpi=600, bbox_inches='tight')
    plt.show()
    



def show_test_results_box_plots_extra(test_num, y_limit = 100, metric = 'nmae_mean', ax_line = (10, 20, 40)):
    results_nmae = pd.DataFrame({})
    results_nmae['dataset'] = pd.read_csv(f'training_results/{test_num}th_test/test_results_0.csv')['dataset']
    results_nmae = results_nmae.set_index('dataset', drop= False)
    dataset_list = results_nmae['dataset'].to_list()


    for i in range(10):
        results_df = pd.read_csv(f'training_results/{test_num}th_test/validation_{i}.csv')

        for dataset in dataset_list:
            results_nmae.loc[dataset,f'{metric}_{i}'] = results_df[results_df['dataset'] == dataset].reset_index(drop=True).loc[19, metric]
            

    df_sorted = results_nmae.assign(row_mean= results_nmae.iloc[:,1:].mean(axis=1)).sort_values("row_mean")
    print(df_sorted['row_mean'].mean())

    results_nmae = df_sorted.drop(columns="row_mean").reset_index(drop=True)


    labels = results_nmae['dataset'].to_list()
    data = []
    for i in range(len(labels)):
        data.append(results_nmae.iloc[i, 1:].astype(float).to_list())

    plt.figure(figsize=(20,10))
    plt.boxplot(x= data, tick_labels= labels, 
                patch_artist= True,
                boxprops=dict(facecolor = 'lightblue',color="blue"), 
                medianprops=dict(color="red", linewidth=2),
                flierprops=dict(marker='o', markerfacecolor='green', markersize=8),
                vert = False)


    plt.axhline(y=ax_line[0], color='green', linestyle='--')
    plt.axhline(y=ax_line[1], color='blue', linestyle='--')
    plt.axhline(y=ax_line[2], color='red', linestyle='--')

    plt.ylabel('NMAE')
    plt.xticks(rotation=90, ha='right')
    plt.ylim(0, y_limit)
    plt.tight_layout()
    plt.show()





def show_metric_variation(test_num1, test_num2, dataset_idx, y_limit = 90, ran = 9):

    available_datasets = pd.read_csv(f'training_results/{test_num1}th_test/test_results_0.csv')['dataset'].to_list()

    
    plot_data_list_1 = []
    plot_data_list_2 = []

    for i in range(10):
        df_val = pd.read_csv(f'training_results/{test_num1}th_test/validation_{i}.csv')
        plot_data = df_val[df_val['dataset'] == available_datasets[dataset_idx]].reset_index().loc[:ran,'nmae_mean']

        plot_data_list_1.append(plot_data.to_list())

        df_val = pd.read_csv(f'training_results/{test_num2}th_test/validation_{i}.csv')
        plot_data = df_val[df_val['dataset'] == available_datasets[dataset_idx]].reset_index().loc[:ran, 'nmae_mean']

        plot_data_list_2.append(plot_data.to_list())


    plt.figure(figsize=(20,10))
    multi_plot_adv(data= plot_data_list_1, labels= [f'{i}' for i in range(10)], title= available_datasets[dataset_idx], y_label= 'NMAE (%)', y_lim= y_limit, data_extra= plot_data_list_2)

import pandas as pd
import numpy as np
import networkx as nx
import community.community_louvain as louvain
from pathlib import Path
from scipy.stats import entropy
from itertools import combinations
from collections import defaultdict
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from scipy.stats import entropy, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score, pairwise_distances, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean, cosine
from networkx.algorithms import bipartite
from kmodes.kmodes import KModes





class SVDFeatures():
    """
    Preprocesses the implicit dataset accordingly and uses SVD to extract requisite meta features from dataset.
    The feature extract method returns the following meta features.

    svd_top1_ratio : The value of largest singular value divided by the sum of all k singular values. Represents global popularity
    svd_top5_ratio : The sum of the first 5 singular values divided by the sum of all k sv. Shows how much of dataset is represented by the top
    svd_entropy    : Shannon Entropy calculated over normalized singular values. High entropy values show that the patterns are spread
    svd_energy_90  : The number of singular values require to capture 90% of spectral energy. A low value says that I only need a few dimensions 
    svd_slope      : A slope if fitted to the singular values in log space. Measures how fast the signal dies


    The item cohesion extractor method returns the following meta features.

    item_cohesion_mean  : A high value indicates that the items are mathematically very similar
    item_cohesion_std   : A high std reflects that the item dataset has clumps. Some items are very similar to each other. 
    item_cohesion_skew  : If positive most items are dissimilar, a few block buster clusters are there

    The reconstruction error extractor returns the following meta features.

    latent_error_ratio          : A high value indicates that the top k values do not capture all the user behavior
    latent_explained_variance   : Similar

    The centroid gap extractor returns the following features

    centroid_gap_euclidean  : Checks how the user and item latent spaces align with each other. 
    centroid_gap_cosine     : If Cosine gap is small and Euclidean is large, users and items interested in same topics but intensity is vastly different

    The method for latent dispersion returns the following features

    user_dispersion_mean    : The average magnitude of user preferences. A low value indicates a weak signal hence harder to infer
    user_dispersion_std     : Measures how much users deviate from the average profile. High value indicates preferences all over the place
    user_dispersion_cv      : The normalized dispersion. High cv indicates that the data is highly skewed

    item_dispersion_mean    : The average magnitude of item aspects. A low value indicates a weak signal hence harder to infer
    item_dispersion_std     : Measures how much items deviate from the average profile. High value indicates attributes all over the place
    item_dispersion_cv      : The normalized dispersion. High cv indicates that the data is highly skewed
    """

    def __init__(self, df):
        self.df = df
        self.u, self.d, self.vt = None, None, None
    
    def prepare_sparse_matrix(self):

        # Convert to categorical
        user_cat = self.df['user'].astype('category')
        item_cat = self.df['item'].astype('category')

        # Extract the sequential codes
        row_indices = user_cat.cat.codes
        col_indices = item_cat.cat.codes

        # Store the number of unique users/items for the matrix shape
        num_users = user_cat.nunique()
        num_items = item_cat.nunique()

        self.sparse_matrix = csr_matrix((np.ones(len(self.df)), (row_indices, col_indices)), 
                     shape=(num_users, num_items))
        

    def extract_svd_features(self, k=50):
        # Compute truncated SVD
        # u and vt contain the latent vectors for users and items. s contains the singular values
        self.prepare_sparse_matrix()
        u, s, vt = svds(self.sparse_matrix.asfptype(), k=k)
        self.u, self.s, self.vt = u, s, vt

        # Sort singular values in descending order
        s = np.sort(s)[::-1]

        # Normalizes singular values
        s_norm = s/ np.sum(s)

        features = {
            'svd_entropy'       : entropy(s_norm),
            'svd_top1_ratio'    : s_norm[0],
            'svd_top5_ratio'    : np.sum(s_norm[:5]),
            'svd_energy_90'     : np.searchsorted(np.cumsum(s_norm), 0.9) / k,
            'svd_slope'         : np.polyfit(np.log(np.arange(1, k+1)), np.log(s), 1)[0]
        }

        return features
    
    def extract_item_cohesion(self, sample_size = 1000):

        # Checks if vt is already calculated
        if self.vt is None:
            raise RuntimeError("You must call .extract_svd_features() before running this method")

        # Transpose so the items are the rows
        item_vectors = self.vt.T

        # Handles when the dataset is smaller than the sample size
        n_items = item_vectors.shape[0]
        actual_sample_size = min(n_items, sample_size)

        # Randomly samples indices from item vector
        indices = np.random.choice(a= n_items, size= actual_sample_size, replace= False)
        sampled_vectors = item_vectors[indices]

        # Calculates cosine similarity matrix shaped (sample_size, sample_size)
        sim_matrix = cosine_similarity(sampled_vectors)

        # Extract the upper triangle of matrix to avoid duplicates
        upper_tri_indices = np.triu_indices_from(sim_matrix, k=1)
        similarities = sim_matrix[upper_tri_indices]

        features = {
            'item_cohesion_mean'    : np.mean(similarities),
            'item_cohesion_std'     : np.std(similarities),
            'item_cohesion_skew'    : float(pd.Series(similarities).skew())
        }

        return features
    
    def extract_reconstruction_error(self):
        
        # Checks whether the necessary attributes are already calculated
        if self.sparse_matrix is None:
            self.prepare_sparse_matrix()
        
        if self.s is None:
            raise RuntimeError("You must call .extract_svd_features() before running this method")

        # Uses Fobenius Norm calculation to derive the energy or signal lost via the truncated SVD
        total_variance = np.sum(self.sparse_matrix.data**2)

        explained_variance = np.sum(self.s)

        reconstruction_error = 1 - (explained_variance / total_variance)

        features = {
            'latent_error_ratio'        : reconstruction_error,
            'latent_explained_variance'  : (explained_variance / total_variance)
        }

        return features
    
    def extract_centroid_gap(self):

        if (self.u is None) or (self.vt is None):
            raise RuntimeError("You must call .extract_svd_features() before running this method")

        # Calculates user centroid, ie mean of each column across all users
        user_centroid = np.mean(self.u, axis = 0)

        # Calculates the item centroid
        item_centroid = np.mean(self.vt.T, axis = 0)

        # Calculate distances Euclidean and Cosine
        gap_euclidean = euclidean(user_centroid, item_centroid)
        gap_cosine = cosine(user_centroid, item_centroid)

        features = {
            'centroid_gap_euclidean'    : gap_euclidean,
            'centroid_gap_cosine'       : gap_cosine
        }

        return features
    
    def extract_latent_dispersion(self):

        # Checks whether the svd calculation is already done
        if (self.u is None) or (self.vt is None):
            raise RuntimeError("You must call .extract_svd_features() before running this method")
        
        # Calculate the L2 norm for every user vector
        user_norms = np.linalg.norm(self.u, axis = 1)

        # Calculate the L2 norm for every item vector
        item_norms = np.linalg.norm(self.vt.T, axis = 1)

        features = {
            # User features
            'user_dispersion_mean'  : np.mean(user_norms),
            'user_dispersion_std'   : np.std(user_norms),
            'user_dispersion_cv'    : np.std(user_norms) / (np.mean(user_norms) + 1e-9),

            # Item features
            'item_dispersion_mean'  : np.mean(item_norms),
            'item_dispersion_std'   : np.std(item_norms),
            'item_dispersion_cv'    : np.std(item_norms) / (np.mean(item_norms) + 1e-9)
        }

        return features
    
class BipartiteGraphFeatures():
    """
    This class calculates a bipartite graph for the input implicit data for the use of extracting 
    requisite dataset meta-features. 

    The centrality and connectivity meta features extractor will return the following features

    'mean degrees'      : The average number of interactions per node. This indicates density.
    'degree_skew'       : A measure of asymmetry. A high skew indicates long tail distribution.
    'mean_pagerank'     : A high PageRank indicates a hierarchical network where a few hubs define the platform. 
    'avg_clustering'    : The degree to which users who share one time also share other items. Signals collaborative filtering
    'lcc_ratio'         : The fraction of nodes connected to the largest connected component

    The authorities and kurtosis extractor returns the following. HITS algorithm assign two scores to every node: hubs and authorities.
    Depending on how many high quality nodes interact with each node, the score is increased

    'auth_mean'         : It quantifies how much of your graph's connectivity is driven by extreme outliers. High value indicates a heavy tail.
    'auth_variance'     : If authority scores have a high variance, there is a good set of high quality items users interact with
    'auth_kurtosis'

    The community features extractor returns the following.

    'avg_bipartite_clustering'  : Measures the tendency of users to form co-consumption squares. If A and B likes 1, how likely to also like 2
    'clustering_variance'       : Tells if predictibility is uniform. A high value means that the dataset is 'patchy'
    'clustering_skewness'       : Measures the asymmetry in clustering.
    'modularity_score'          : Measures how easily network can be divided into groups.
    'num_communitites'          : The number of interest niches
    'comm_size_std'             : Measures the inequality between communities. There could be one giant community and many niche communities
    'comm_entropy'              : Measures balance of communities. A high number indicates users and items being divided equally across communities
    'avg_conductance'           : Measures leakage between communities
    'edge_node_ratio'           : A fundamental measure of density. 

    The structural distance method extracts the following meta features

    'avg_path_length'           : The average shortest path length between nodes. High APL indicates a complex stringy graph
    'max_path_length_approx'    : This is the longest "shortest path. A small value indicates a tight knit community.
    'path_length_std'           : Standard deviation of distances. Low value indicates a homogeneous graph.
    'path_length_skew'          : Measures if the distribution of distances tend to short or long side with or without certain outliers

    """


    def __init__(self, df):
        self.df = df
    
    def extract_centrality_connectivity_features(self):
        # Initialize Bipartite graph
        self.B = nx.Graph()
        

        # Add the respective nodes of the graph
        self.B.add_nodes_from(self.df['user'].unique(), bipartite = 0)
        self.B.add_nodes_from(self.df['item'].unique(), bipartite = 1)

        # Add the edges
        self.B.add_edges_from([
            (f"u_{u}", f"i_{i}") for u, i in zip(df['user'], df['item'])
        ])

        # Extract features
        # Degree Distribution
        degrees = [d for n, d in self.B.degree()]
        feat_mean_deg = np.mean(degrees)
        feat_skew_deg = pd.Series(degrees).skew()

        # PageRank
        pagerank = nx.pagerank(self.B)
        pr_values = list(pagerank.values())
        feat_mean_pr = np.mean(pr_values)

        # Clustering coefficient (Bipartite)
        # Average clustering across all nodes
        feat_clustering = nx.average_clustering(self.B)

        # Largest connected component (LCC) size
        lcc = self.B.subgraph(max(nx.connected_components(self.B), key= len))
        feat_lcc_ratio = lcc.number_of_nodes() / self.B.number_of_nodes()

        features = {
            'mean_degrees'  : feat_mean_deg,
            'degree_skew'   : feat_skew_deg,
            'mean_pagerank' : feat_mean_pr,
            'avg_clustering': feat_clustering,
            'lcc_ratio'     : feat_lcc_ratio
        }

        return features
    
    def extract_auth_and_kurtosis(self):

        hubs, authorities = nx.hits(self.B, max_iter= 100)

        auth_values = list(authorities.values())

        features = {
            'auth_mean'     : np.mean(auth_values),
            'auth_variance' : np.var(auth_values),
            'auth_kurtosis' : pd.Series(auth_values).kurtosis()
        }

        return features
    
    def extract_community_features(self):


        # Bipartite clustering coefficient
        cc_dict = bipartite.clustering(self.B)
        cc_values = list(cc_dict.values())
        
        # Community detection
        partition = louvain.best_partition(self.B)

        # Community statistics
        comm_assignments = pd.Series(partition.values())
        comm_counts = comm_assignments.value_counts()

        # 4. Conductance (Average across communities)
        # Lower conductance = more 'isolated' or 'bottlenecked' groups
        conductance_list = []
        for comm_id in comm_counts.index:
            nodes_in_comm = [n for n, c in partition.items() if c == comm_id]
            if 0 < len(nodes_in_comm) < len(self.B.nodes):
                conductance_list.append(nx.conductance(self.B, nodes_in_comm))

        features = {
            'avg_bipartite_clustering'  : np.mean(cc_values),
            'clustering_variance'       : np.var(cc_values),
            'clustering_skewness'       : pd.Series(cc_values).skew(),
            'modularity_score'          : louvain.modularity(partition, self.B),
            'num_communitites'          : len(comm_counts),
            'comm_size_std'             : comm_counts.std(),
            'comm_entropy'              : entropy(comm_counts),
            'avg_conductance'           : np.mean(conductance_list) if conductance_list else 0,
            'edge_node_ratio'           : self.B.number_of_edges() / self.B.number_of_nodes()
        }

        return features
    
    def extract_structural_distance_features(self, sample_size = 500):

        # Extract the largest connected component
        if not nx.is_connected(self.B):
            lcc_nodes = max(nx.connected_components(self.B), key=len)
            G_lcc = self.B.subgraph(lcc_nodes).copy()
        else:
            G_lcc = self.B

        # Sampling for efficiency in case graph is large
        nodes = list(G_lcc.nodes())
        if len(nodes) > sample_size:
            sample_nodes = np.random.choice(nodes, sample_size, replace=False)
        else:
            sample_nodes = nodes

        # Calculate distances
        path_lengths = []
        for node in sample_nodes:
            lengths = nx.single_source_shortest_path_length(G_lcc, node)
            path_lengths.extend(lengths.values())
        
        features = {
            'avg_path_length'       : np.mean(path_lengths),
            'max_path_length_approx': np.max(path_lengths),     # Approx Diameter
            'path_length_std'       : np.std(path_lengths),
            'path_length_skew'      : pd.Series(path_lengths).skew()
        }
    
        return features

class KModesClusteringFeatures():

    """
    This class runs a K-modes clustering algorithms to extract meta-features from the input implicit data. 

    The k-modes metafeatures extractor will return the following meta features

    'kmodes_total_cost'         :    Sum of all 'mismatches' between a user and its assigned cluster
    'kmodes_avg_cost'           :    The average cost of the users. Tell how much a user deviates on average from his cluster
    'kmodes_iterations'         :    The number of times the algorithm had to reshuffle users before the clusters stopped changing.
    'kmodes_normalized_cost'    :    The cost divided by the total number of cells in the matrix. Represents the error rate of assuming everyone fits into a few clusters
    
    The advanced k-modes features extractor could be used to extract the following. 
     
    'kmodes_cluster_size_mean'         :    Measures how evenly the users are distributed among the clusters     
    'kmodes_cluster_size_std_dev'      :    Measures how evenly the users are distributed. High std could mean that a model could be biased towards a majority clusters 
    'kmodes_cluster_size_gini'         :    A gini of 0 means that all clusters are of similar size while a 1 gini value means that a single cluster is massive.
    'kmodes_intra_dissimilarity_mean'  :    Measures the average dissimilarity between a cluster's users and its 'mode'. Measure of purity of a cluster
    'kmodes_intra_dissimilarity_std'   :    Similar to above in the result
    'kmodes_mode_entropy_mean'         :    Measures how diverse the typical interaction pattern is. High entropy signals a balanced mix of 1s and 0, while low usually means very sparse 
    'kmodes_mode_overlap_mean'         :    Measures how similar the clusters are to each other.
    

    The following 'quality' based meta features maybe extracted via the quality extractor

    'kmodes_silhouette_hamming_mean'    :   The sil score measures how similar a user is to their own cluster compared to the next closest cluster
    'kmodes_silhouette_hamming_median'  :   A high value for mean and median values imply that the cllusters are distinct and separated well
    'kmodes_cluster_purity_popularity'  :   Measures the overlap between the clusters and the global most popular items. Low purity means that the 'modes' are simply the top 10 items
    'kmodes_empty_clusters'             :   Finds out how many of the clusters ended up with 0 after cluster convergence. 

    The following meta features extract details related to the behavior of the dataset in question.     

    'kmodes_modes_sparsity_mean'    :   Mode sparsity shows what percentage of the mode's values are '1's  
    'kmodes_modes_sparsity_std'     :   A low sparsity means that the 'modal' items of the cetroids are actually only interacting with a handful of items
    'kmodes_rank_corr_mean'         :   The items in the clusters are sorted in accordance with their popularities and compared with the global popularity rankings
    'kmodes_rank_corr_std'          :   High correlations indicate that everyone just likes the same items. Low correlation is a massive indication that personalization is possible
    'kmodes_centroid_dist_mean'     :   Calculates the distribution of (hamming) distances between all pairs of centroids.  
    'kmodes_centroid_dist_skew'     :   A high Skew indicates that most clusters are similar but there are a few outliers

    The following features speak to the stability of calculated clusters. When run on 80-90% of sampled data, how much do the centroids and cost change. 
    If the stability is low, the patterns discovered are likely noise.

    'kmodes_stability_avg_drift'    :   A low drift indicates that the clusters are robust
    'kmodes_stability_variance'     :   High variance indicates that the models gets stuck on different local optima every time it runs


    """

    def __init__(self, df):
        self.df = df

        if self.df['user'].nunique() > 5000:
            top_indices = self.df['user'].value_counts().iloc[:5000].index.to_list()
            self.df = self.df.set_index('user', drop= False).loc[top_indices].reset_index(drop= True)
        
        if self.df['item'].nunique() > 10000:
            top_indices = self.df['item'].value_counts().iloc[:10000].index.to_list()
            self.df = self.df.set_index('item', drop= False).loc[top_indices].reset_index(drop= True)


    def extract_kmodes_meta_features(self, k = 5):
        
        self.df['value'] = 1 
        self.wide_df = self.df.pivot(index= 'user', columns= 'item', values= 'value').fillna(0)

        

        self.km = KModes(n_clusters=5, init='Huang', n_init=3, verbose=0)
        clusters = self.km.fit_predict(self.wide_df)


        features = {
        'kmodes_total_cost': self.km.cost_, # Sum of dissimilarities
        'kmodes_avg_cost':   self.km.cost_ / len(self.wide_df),
        'kmodes_iterations': self.km.n_iter_,
        # Ratio of cost to number of items (complexity measure)
        'kmodes_normalized_cost': self.km.cost_ / (len(self.wide_df) * self.wide_df.shape[1])
        }

        return features


    def extract_advanced_kmodes_meta_features(self):
        labels = self.km.labels_
        cluster_sizes = np.bincount(labels)

        # Gini Coefficient for cluster size balance
        def gini(x):
            sorted_x = np.sort(x)
            n = len(x)
            cum_x = np.cumsum(sorted_x)
            return (n + 1 - 2 * np.sum(cum_x) / cum_x[-1]) / n
        
        # Intra cluster dissimilarity
        centroids = self.km.cluster_centroids_
        individual_dissimilarities = []

        for i, row in enumerate(self.wide_df.values):
            centroid = centroids[labels[i]]
            # Count mismatches
            individual_dissimilarities.append(np.sum(row != centroid))

        # Mode (centroid) analysis
        mode_ratios = np.mean(centroids, axis=1)

        def binary_entropy(p):
            p = np.clip(p, 1e-9, 1-1e-9) # Avoid log(0)
            return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

        # Mode Overlap (Mean Jaccard-like similarity between centroids)
        overlap_matrix = cosine_similarity(centroids)

        # Exclude diagonal (self-overlap)
        np.fill_diagonal(overlap_matrix, 0)
        mean_overlap = np.sum(overlap_matrix) / (len(centroids) * (len(centroids) - 1))

        features = {
            'kmodes_cluster_size_mean':       np.mean(cluster_sizes),
            'kmodes_cluster_size_std_dev':    np.std(cluster_sizes),
            'kmodes_cluster_size_gini':       gini(cluster_sizes),
            'kmodes_intra_dissimilarity_mean': np.mean(individual_dissimilarities),
            'kmodes_intra_dissimilarity_std':  np.std(individual_dissimilarities),
            'kmodes_mode_entropy_mean':       np.mean([binary_entropy(p) for p in mode_ratios]),
            'kmodes_mode_overlap_mean':       mean_overlap
        }

        return features

    def extract_quality_kmodes_meta_features(self):
        labels = self.km.labels_
        centroids = self.km.cluster_centroids_

        # Silhouette Hamming metrics
        # Uses a try loop here since the silhouette algorithm can be unreliable and error prone

        try:
            if len(set(labels)) > 1:
                s_scores = silhouette_score(self.wide_df, labels, metric = 'hamming')
                sil_mean = np.mean(s_scores)
                sil_median = np.median(s_scores)

            else:
                sil_mean, sil_median = 0, 0
        except:
            sil_mean, sil_median = 0, 0
 
        # Cluster Purity Popularity
        # Calculates global top items

        global_pop = self.wide_df.sum(axis=0).sort_values(ascending= False).index[:10]
        purity_scores = []

        for mode in centroids:
            # Check how many of the cluster's 'Mode' items are just the global top items
            mode_items = np.where(mode == 1)[0]
            # Purity is high if the mode items are not the global top items
            overlap = len(set(mode_items).intersection(set(global_pop)))
            purity_scores.append(1 - (overlap / 10) if len(mode_items) > 0 else 0)


        # Empty clusters
        cluster_counts = np.bincount(labels, minlength = self.km.n_clusters)
        empty_clusters = np.sum(cluster_counts == 0)

        features = {
                'kmodes_silhouette_hamming_mean' : sil_mean,
                'kmodes_silhouette_hamming_median' : sil_median,
                'kmodes_cluster_purity_popularity' : np.mean(purity_scores),
                'kmodes_empty_clusters' : int(empty_clusters)

                }

        return features

    def extract_behavioral_meta_features(self):
        centroids = self.km.cluster_centroids_

        # Calculate the Mode sparsity
        mode_densities = np.mean(centroids, axis=1)

        # Calculate the cluster global correlations
        global_pop = self.wide_df.sum(axis=0).values
        correlations = []

        for mode in centroids:
           corr, _ = spearmanr(mode, global_pop)
           correlations.append(corr if not np.isnan(corr) else 0)
        
        # Calculate the Intercluster Separation via Hamming Distances between centroid pairs
        centroid_dist = pairwise_distances(centroids, metric = 'hamming')

        dist_values = centroid_dist[np.triu_indices(len(centroids), k=1)]


        features = {
                'kmodes_modes_sparsity_mean'    : np.mean(mode_densities),
                'kmodes_modes_sparsity_std'     : np.std(mode_densities),
                'kmodes_rank_corr_mean'         : np.mean(correlations),
                'kmodes_rank_corr_std'          : np.std(correlations),
                'kmodes_centroid_dist_mean'     : np.mean(dist_values),
                'kmodes_centroid_dist_skew'     : pd.Series(dist_values).skew() if len(dist_values) < 2 else 0  
                }
        
        return features

    def measure_kmodes_stability(self, k=5, iterations = 2, sub_sample_percentage = 0.9):
        original_centroids = self.km.cluster_centroids_

        stability_scores = []

        for _ in range(iterations):
            # Subsample the users
            sample_df = self.wide_df.sample(frac = sub_sample_percentage)

            # Fit a new kmodes model
            km_sub = KModes(n_clusters = k, init='Huang', n_init=1, max_iter= 20)
            km_sub.fit(sample_df)
            new_centroids = km_sub.cluster_centroids_

            # Compare the old and the new centroids to ascertain stability
            distances = pairwise_distances(original_centroids, new_centroids, metric = 'hamming')

            # For each original centroid find the closes matching new centroid and pair them up
            min_distances = np.min(distances, axis=1)
            stability_scores.append(np.mean(min_distances))

            features = {
                    'kmodes_stability_avg_drift'    : np.mean(stability_scores),
                    'kmodes_stability_variance'     : np.std(stability_scores)
                    }

            return features


class DecisionTreeFeatures():

    """
    This class runs a Decision Tree lightweight algorithm to extract meta features from implicit datasets. The class removes the top-item
    from the input dataset and runs a Decision Tree algorithm based on the remaining items for each user (the remaining items serve as features for each
    user) to accurately try and predict user's engagement with the top item.

    The extract meta-features method yields the following meta features on the decision tree.

    'tree_node_count'   : The number of total nodes in the decision tree. 
    'tree_leaf_count'   : The number of terminal nodes in the dataset
    'tree_max_depth'    : Shallow trees indicate simple relationship
    'tree_avg_depth'    : 
    'tree_impurity'     : Low purity indicates a dataset with blurry borders.

    The importance based meta feature extractor yields the following
    
    'tree_importance_skew'          : Importance measures the significance of each feature. A high skew indicates one or two items dominate the importance
    'tree_n_features_to_90'         : How many unique features does it take to reach 90% importance
    'tree_n_features_to_80'         :  
    'tree_n_features_to_70'         : 
    'tree_n_features_to_60'         : 
    'tree_n_features_to_50'         : 
    'tree_top_1p_concentration'     : The summed importance of the top 1% of the items/features
    'tree_top_5p_concentration'     : 
    'tree_importance_decay_ratio'   : Ratio of importance of 1st and 10th feature. Measures how quickly the signal drops. A slow decay indicates many important features
    
    The following advanced features give information pertaining to the structure, and node and path level metrics. 
    
    'tree_avg_leaf_size'        : The average number of users that end up at each terminal leaf. Tiny leaf sizes indicate that the tree had to find very specific niche rules
    'tree_leaf_size_cv'         : Indicates whether the leaves are roughly the same size, or that some are large while others are tiny
    'tree_balance_ratio'        : A high balance suggests that the tree has a main path, which the majority of the items take
    'tree_feature_utilization'  : The percentage of features in the mix that the tree actually used to make a split 
    'tree_feature_redundancy'   : How many unique items were used relative to the total number of nodes in the tree. High redundancy indicates that the tree keeps splitting on the same item again and again

    The impurity reduction method results in the following metric.

    'tree_total_impurity_reduction' : Tells how much information gain was possible. If the value is very low (close to 0) the chosen items have no predictive power
    'tree_normalized_improvement'   : A low value suggests that the data is unpredictable noise

    The quality and reliabilty metrics for the tree are given below.

    'tree_train_score'      : Measures the accuracy of the tree on the data it was trained on
    'tree_val_score'        : Tests the accuracy of the tree on the 'held out' dataset
    'tree_overfit_gap'      : Indicates how much the accuracy of train score differs from validation score. A low value indicates robustness.
    'tree_minmax_prob_gap'  : Measures the gaps between the most confident and least confident prediction
               
    The average path is extracted as follows

    'tree_avg_path_length'  : The tree may have multiple paths leading to leaves. This measures the average value of each.

    The following indicates the stability of the tree upong bootstrapping

    'tree_stability_bootstrap' : Measures how stable the model is upon bootstrapping

    """

    def __init__(self, df, num_features = 1000):
        self.df = df
        self.sparse_matrix = None

        # Find the most popular item
        item_counts = self.df['item'].value_counts()
        top_item_id = item_counts.index[0]


        # Find the user ids of those who interacted with the top item
        self.users_with_target = self.df[ self.df['item'] == top_item_id ]['user'].unique()

        # Remove the top item from the df and select only the predictor items
        predictor_items_indices = item_counts.index[1:(num_features+1)]


        self.filtered_df = self.df[ self.df['item'].isin(predictor_items_indices)].copy()


        self.filtered_df['value'] = 1

        # Make the sparse matrix
        self.prepare_sparse_matrix()


    def prepare_sparse_matrix(self):

        # Convert to categorical
        user_cat = self.filtered_df['user'].astype('category')
        item_cat = self.filtered_df['item'].astype('category')

        # Extract sequential codes
        user_indices = user_cat.cat.codes
        item_indices = item_cat.cat.codes

        # The user codes mapping
        user_mapping_df = pd.DataFrame({
            'original_user_id': user_cat.unique(),
            'mapped_user_codes': user_indices.unique()
            })


        # Store the number of unique users and unique items for sparse mat shape
        num_users = user_cat.nunique()
        num_items = item_cat.nunique()

        self.X_sparse = csr_matrix((np.ones(len(self.filtered_df)), (user_indices, item_indices)), shape = (num_users, num_items))

        # Make the y label array
        self.y = user_mapping_df['original_user_id'].isin(self.users_with_target).astype(int)


    def extract_tree_meta_features(self):
        
        self.dt = DecisionTreeClassifier(max_depth= 10, random_state= 42)
        self.dt.fit(self.X_sparse, self.y)

        self.tree = self.dt.tree_

        features = {
                'tree_node_count' : self.tree.node_count,
                'tree_leaf_count' : self.dt.get_n_leaves(),
                'tree_max_depth'  : self.dt.get_depth(),
                'tree_avg_depth'  : np.mean(self.tree.children_left != -1),
                'tree_impurity'   : np.mean(self.tree.impurity)
                }
        
        return features
    
    def extract_importance_meta_features(self):

        importances = self.dt.feature_importances_

        # Sort importances in descending order
        sorted_imp = np.sort(importances)[::-1]
        cumulative_imp = np.cumsum(sorted_imp)

        # Feature count to capture 90% importance
        n_to_90 = np.searchsorted(cumulative_imp,0.9)+1

        # Feature count to capture 80% importance
        n_to_80 = np.searchsorted(cumulative_imp, 0.8)+1

        # Feature count to capture 70% importance
        n_to_70 = np.searchsorted(cumulative_imp, 0.7)+1

        # Feature count to capture 60% importance
        n_to_60 = np.searchsorted(cumulative_imp, 0.6)+1

        # Feature count to capture 50% importance
        n_to_50 = np.searchsorted(cumulative_imp, 0.5)+1

        # Top-K concentration
        top_1p_idx = max(1, int(len(sorted_imp)*0.001))
        top_1p_concentration = np.sum(sorted_imp[:top_1p_idx])

        top_5p_idx = max(1, int(len(sorted_imp)*0.005))
        top_5p_concentration = np.sum(sorted_imp[:top_5p_idx])

        # Decay ratio
        decay_ratio = sorted_imp[0] / (sorted_imp[9] + 1e-9) if len(sorted_imp) > 9 else 0

        features = {
                    'tree_importance_skew' : pd.Series(importances).skew(),
                    'tree_n_features_to_90': n_to_90,
                    'tree_n_features_to_80': n_to_80,
                    'tree_n_features_to_70': n_to_70,
                    'tree_n_features_to_60': n_to_60,
                    'tree_n_features_to_50': n_to_50,
                    'tree_top_1p_concentration' : top_1p_concentration,
                    'tree_top_5p_concentration' : top_5p_concentration,
                    'tree_importance_decay_ratio' : decay_ratio
                }

        return features
    
        

    def extract_advanced_tree_meta_features(self):
        
        # Identify which nodes are leaves
        is_leaf = self.tree.children_left == -1
        leaf_samples = self.tree.n_node_samples[is_leaf]

        # Leaf Density
        avg_leaf_size = np.mean(leaf_samples)
        leaf_size_std = np.std(leaf_samples)

        # Structural Symmetry

        n_leaves = self.dt.get_n_leaves()
        theoretical_depth = np.log2(n_leaves) if n_leaves > 0 else 1
        balance_ratio = self.dt.get_depth() / theoretical_depth

        # Unique features used

        unique_features_used = len(np.unique(self.tree.feature[self.tree.feature != -2]))
        feature_redundancy = unique_features_used / self.tree.node_count

        features = {
                'tree_avg_leaf_size'        : avg_leaf_size,
                'tree_leaf_size_cv'         : leaf_size_std / (avg_leaf_size + 1e-9),
                'tree_balance_ratio'        : balance_ratio,
                'tree_feature_utilization'  : unique_features_used / self.X_sparse.shape[1],
                'tree_feature_redundancy'   : feature_redundancy

                }
        
        return features


    def extract_impurity_reduction(self):
        # Extract impurity and sample count for every node
        impurity = self.tree.impurity
        n_samples = self.tree.n_node_samples

        # Get indices for left and right children
        left_child = self.tree.children_left
        right_child = self.tree.children_right

        split_nodes = np.where(left_child != -1)[0]

        total_reduction = 0.0

        for i in split_nodes:
            parent_weight = n_samples[i] / n_samples[0]

            reduction = (impurity[i] - 
                    (n_samples[left_child[i]] / n_samples[i]) * impurity[left_child[i]] - 
                    (n_samples[right_child[i]] / n_samples[i]) * impurity[right_child[i]])
            
            total_reduction += reduction * parent_weight
            
        relative_reduction = total_reduction / self.tree.impurity[0] if (self.tree.impurity[0] > 0) else 0

        features = {
                'tree_total_impurity_reduction' : total_reduction,
                'tree_normalized_improvement'   : relative_reduction
                }

        return features

    def extract_performance_meta_features(self):

        X_train, X_val, y_train, y_val = train_test_split(
                self.X_sparse, self.y, test_size = 0.2, random_state= 42, stratify = self.y
                )
        

        # Fit the tree
        dt_n = DecisionTreeClassifier(max_depth = 10, random_state= 42)
        dt_n.fit(X_train, y_train)
        
        # Calculate the scores
        train_preds = dt_n.predict(X_train)
        val_preds = dt_n.predict(X_val)

        train_score = accuracy_score(y_train, train_preds)
        validation_score = accuracy_score(y_val, val_preds)

        # Calculate the overfit gap
        overfit_gap = train_score - validation_score

        # Calculate the minmax probability gap mean
        probs = dt_n.predict_proba(X_val)

        gaps = np.max(probs, axis=1) - np.min(probs, axis=1)
        minmax_prob_gap_mean = np.mean(gaps)

        features = {
                'tree_train_score'      : train_score,
                'tree_val_score'        : validation_score,
                'tree_overfit_gap'      : overfit_gap,
                'tree_minmax_prob_gap'  : minmax_prob_gap_mean
                }

        return features

    def extract_stability_bootstrap(self, n_iterations = 1):

        n_samples = self.X_sparse.shape[0]
        
        # Create two different bootstrap samples
        idx1 = np.random.choice(n_samples, n_samples, replace = True)
        idx2 = np.random.choice(n_samples, n_samples, replace = True)

        def get_subset(data, indices):
            return data[indices] if isinstance(data, np.ndarray) else data.tocsr()[indices]

        # Train tree A
        dt1 = DecisionTreeClassifier(max_depth = 10, random_state=1)
        dt1.fit(get_subset(self.X_sparse, idx1), self.y[idx1])
        imp1 = dt1.feature_importances_

        # Train tree A
        dt2 = DecisionTreeClassifier(max_depth = 10, random_state=2)
        dt2.fit(get_subset(self.X_sparse, idx2), self.y[idx2])
        imp2 = dt2.feature_importances_

        # Calculate the stability using spearman correlation
        stability, _ = spearmanr(imp1, imp2)

        if np.isnan(stability):
            stability = 0.0

        return {'tree_stability_bootstrap' : stability}

    def extract_avg_path_length(self):
        n_nodes = self.tree.node_count
        children_left = self.tree.children_left
        children_right = self.tree.children_right
        n_node_samples = self.tree.n_node_samples


        def get_path_lengths(node_id, current_depth):
            # If it's a leaf node
            if children_left[node_id] == -1:
                # Return (depth * samples_in_this_leaf, samples_in_this_leaf)
                return (current_depth * n_node_samples[node_id], n_node_samples[node_id])
            
            # If it's a split node, recurse down
            left_sum, left_samples = get_path_lengths(children_left[node_id], current_depth + 1)
            right_sum, right_samples = get_path_lengths(children_right[node_id], current_depth + 1)
            
            return (left_sum + right_sum), (left_samples + right_samples)

        total_weighted_depth, total_samples = get_path_lengths(0,0)

        return {'tree_avg_path_length': total_weighted_depth / total_samples}


def calculate_gini(interactions_df, item_col='item_id'):
    # 1. Get interaction counts per item
    counts = interactions_df[item_col].value_counts().values
    
    # 2. Sort the counts in ascending order
    counts = np.sort(counts)
    n = len(counts)
    
    # 3. Calculate index-weighted sum
    index = np.arange(1, n + 1)
    return (np.sum((2 * index - n - 1) * counts)) / (n * np.sum(counts))


def get_normalized_entropy(df, item_col='item_id'):
    # 1. Get the probability distribution (proportions)
    prob_dist = df[item_col].value_counts(normalize=True)
    
    # 2. Calculate the number of unique items
    n_unique = df[item_col].nunique()
    
    if n_unique <= 1:
        return 0.0
    
    # 3. Calculate Shannon Entropy
    actual_entropy = entropy(prob_dist, base=2)
    
    # 4. Calculate Maximum Entropy (log2 of number of items)
    max_entropy = np.log2(n_unique)
    
    # 5. Return the normalized ratio
    return actual_entropy / max_entropy


def top_bottom_metrics(df, item_col='item_id', top = True, num = 10):
    # 1. Get interaction counts
    counts = df[item_col].value_counts()
    
    # 2. Slice the top 10
    if top:
        top_bottom_num = counts.head(num).values
    else:
        top_bottom_num = counts.tail(num).values
    
    # 3. Calculate Stats
    variance = np.var(top_bottom_num)
    std_dev = np.std(top_bottom_num)
    mean_pop = np.mean(top_bottom_num)
    cv = std_dev / mean_pop
    
    return variance, std_dev, cv


def fill_df(df, fill = 'mean'):

    df['p_inter_top_user'] = df['p_inter_top_user'].fillna(df['p_inter_top_user'].mean())

    return df


def find_duplicates(df, threshold_similarity = 1):

    # Make adjacent list. The object is a series indexed by user_id and each element is a set of items
    # the respective user interacts with
    adj_list = df.groupby('user')['item'].apply(set)

    # Filters out users with less than 10 interactions
    mask = adj_list.str.len() > 10
    adj_dict = adj_list[mask].to_dict()
    adj_list = adj_list[mask]

    # Creates an inverted index dictionary, with the index being items and elements user collections
    inverted_index = defaultdict(list)

    for user, items in adj_list.items():
        for item in items:
            inverted_index[item].append(user)

    # Filters out items without at least a single interaction. 
    candidate_score = defaultdict(int)

    for item, users in inverted_index.items():
        if len(users) < 2:
            continue

        for u1, u2 in combinations(sorted(users), 2):
            candidate_score[(u1, u2)] += 1

    duplicate_sets = defaultdict(set)
    set_no = 0
    for (u1, u2), intersect_size in candidate_score.items():

        union_size = len(adj_dict[u1] | adj_dict[u2])
        score = intersect_size / union_size

        # If similarity is 1, adds the two elements to respective set or make new set
        if score >= threshold_similarity:
            found_key = next((k for k,s in duplicate_sets.items() if ((u1 in s) or (u2 in s))),None)

            if found_key:
                duplicate_sets[found_key].add(u1)
                duplicate_sets[found_key].add(u2)
            else:
                set_no += 1
                duplicate_sets[set_no] = {u1, u2}

    num_duplicate_sets = len(duplicate_sets)

    if num_duplicate_sets == 0:
        return 0, 0, 0, 0

    num_duplicates = sum([len(s) for s in duplicate_sets.values()])
    sorted_profiles = sorted([(len(s), k) for k,s in duplicate_sets.items()], reverse= True)
    freq_most_common_u_pattern = sorted_profiles[0][0]

    # Interaction mass of profile
    key = sorted_profiles[0][1]
    usr = next(iter(duplicate_sets[key]))
    i_mass_most_common_profile = len(adj_dict[usr])

    return num_duplicate_sets, num_duplicates, freq_most_common_u_pattern, i_mass_most_common_profile


def p_high_low_items(df, threshold = 5, high= True):
    """
    Returns the percentage value of total interactions accumulated by items with at least
    the given threshold. 
    """
    val_cnt = df['item'].value_counts()

    if high:
        mask = val_cnt > threshold
    else:
        mask = val_cnt <= threshold
    
    return val_cnt[mask]/df['item'].nunique()


def users_inter_with_low_prob_items(df, inter_threshold = 2):
    """
    Returns the number of users that have interacted with items with low interaction numbers, and
    a percentage figure against the total number of users
    """
    adj_list = df.groupby('item')['user'].apply(set)

    # Filters for items with low interactions and turns their indices into a list
    temp_df = df['item'].value_counts()
    low_inter_item_indices = temp_df[ temp_df <= inter_threshold].index.to_list()

    # Iterates through the indices list and include each new set into the user set
    user_set = set()

    for i in low_inter_item_indices:
        user_set.update(adj_list[i])
    
    return len(user_set), len(user_set)/(df['user'].nunique())


def get_top_n_percentage( df, col, threshold = 0.8):
    """
    Calculates the number of top n-percentage (80%) users or items in terms of the accumulated 
    interactions. Given a column label and a threshold(percentage), returns the total number of users/items
    that fall within that percentage.
    """
    cum_weights = df[col].value_counts().cumsum() / df['user'].value_counts().sum()
    top_indices = (cum_weights[ cum_weights <= threshold].index.to_list())

    return len(top_indices)





file_path_meta = Path('datasets_info.csv')
meta_df = pd.read_csv(file_path_meta).set_index('dataset', drop= False)
available_data = meta_df[meta_df['downloaded'] == 1]['dataset'].to_list()
meta_df = meta_df.loc[available_data]
meta_df = meta_df.iloc[:,1:2]

meta_features_labels = ['num_users', 'num_items', 'num_inter', 'density', 'user_item_ratio', 'item_user_ratio', 'highest_num_rating_u', 
                        'lowest_num_rating_u', 'highest_num_rating_i', 'lowest_num_rating_i', 'mean_num_rating_u', 'mean_num_rating_i', 'p_inter_top_user', 
                        'p_top10_to_overall', 'p_top5_to_overall', 'p_low5_to_overall', 'num_users_top_10_p', 'num_users_top_20_p', 'num_users_top_30_p', 
                        'num_users_top_40_p', 'num_users_top_50_p', 'num_users_top_60_p', 'num_users_top_70_p', 'num_users_top_80_p', 'num_users_top_90_p',
                        'num_items_top_10_p', 'num_items_top_20_p', 'num_items_top_30_p', 'num_items_top_40_p', 'num_items_top_50_p', 'num_items_top_60_p', 
                        'num_items_top_70_p', 'num_items_top_80_p', 'num_items_top_90_p', 'mean_inter_top10_user', 'mean_inter_top5_user', 
                        'mean_inter_low10_user', 'mean_inter_low5_user', 'top10tobottom10_user', 'top10tobottom5_user', 'gini_coefficient', 'shannon_entropy', 
                        'head_to_tail_ratio', 'inter_to_unique_items', 'num_coldstart_users', 'p_coldstart_users', 'num_low_inter_items', 'p_low_inter_items', 
                        'num_u_i_low_inter_items', 'p_u_i_low_inter_items', 'num_low_inter_items_5', 'p_low_inter_items_5', 'num_high_inter_items_5', 
                        'p_high_inter_items_5', 'overall_items_variance', 'overall_items_std_dev', 'overall_items_CV', 'top10_items_variance', 
                        'top10_items_std_dev', 'top10_items_CV', 'bottom10_items_variance', 'bottom10_items_std_dev', 'bottom10_items_CV', 
                        'num_duplicate_u_profiles', 'num_duplicate_u_profile_sets', 'ratio_duplicate_users', 'freq_common_u_pattern', 'num_onetime_patterns', 
                        'i_mass_most_common_prof', 'num_similar_0_9_u_profiles', 'num_similar_0_9_u_profile_sets', 'ratio_similar_0_9_users', 
                        'freq_similar_0_9_u_pattern', 'svd_top1_ratio', 'svd_top5_ratio', 'svd_entropy', 'svd_energy_90', 'svd_slope', 'item_cohesion_mean', 
                        'item_cohesion_std', 'item_cohesion_skew', 'latent_error_ratio', 'latent_explained_variance', 'centroid_gap_euclidean', 
                        'centroid_gap_cosine', 'user_dispersion_mean', 'user_dispersion_std', 'user_dispersion_cv', 'item_dispersion_mean', 
                        'item_dispersion_std', 'item_dispersion_cv', 'kmodes_total_cost', 'kmodes_avg_cost', 'kmodes_iterations', 'kmodes_normalized_cost', 
                        'kmodes_cluster_size_mean', 'kmodes_cluster_size_std_dev', 'kmodes_cluster_size_gini', 'kmodes_intra_dissimilarity_mean', 
                        'kmodes_intra_dissimilarity_std', 'kmodes_mode_entropy_mean', 'kmodes_mode_overlap_mean', 'kmodes_silhouette_hamming_mean', 
                        'kmodes_silhouette_hamming_median', 'kmodes_cluster_purity_popularity', 'kmodes_empty_clusters', 'kmodes_modes_sparsity_mean', 
                        'kmodes_modes_sparsity_std', 'kmodes_rank_corr_mean', 'kmodes_rank_corr_std', 'kmodes_centroid_dist_mean', 'kmodes_centroid_dist_skew', 
                        'kmodes_stability_avg_drift', 'kmodes_stability_variance', 'bip_g_mean degrees', 'bip_g_degree_skew', 'bip_g_mean_pagerank', 
                        'bip_g_avg_clustering', 'bip_g_lcc_ratio', 'bip_g_auth_mean', 'bip_g_auth_variance', 'bip_g_auth_kurtosis', 
                        'bip_g_avg_bipartite_clustering', 'bip_g_clustering_variance', 'bip_g_clustering_skewness', 'bip_g_modularity_score', 
                        'bip_g_num_communitites', 'bip_g_comm_size_std', 'bip_g_comm_entropy', 'bip_g_avg_conductance', 'bip_g_edge_node_ratio', 
                        'bip_g_avg_path_length', 'bip_g_max_path_length_approx', 'bip_g_path_length_std', 'bip_g_path_length_skew', 'tree_node_count',
                        'tree_leaf_count','tree_max_depth','tree_avg_depth','tree_impurity','tree_importance_skew','tree_n_features_to_90',
                        'tree_n_features_to_80','tree_n_features_to_70','tree_n_features_to_60','tree_n_features_to_50','tree_top_1p_concentration',
                        'tree_top_5p_concentration','tree_importance_decay_ratio','tree_avg_leaf_size','tree_leaf_size_cv','tree_balance_ratio',
                        'tree_feature_utilization','tree_feature_redundancy','tree_total_impurity_reduction','tree_normalized_improvement','tree_train_score',
                        'tree_val_score','tree_overfit_gap','tree_minmax_prob_gap','tree_avg_path_length', 'tree_stability_bootstrap'
]

empty_df = pd.DataFrame(columns= meta_features_labels, index= meta_df.index)

meta_df = pd.concat([meta_df, empty_df], axis= 1)



for i in (available_data[20:21]):
    dataset_name = i
    file_path = Path(f'datasets/{dataset_name}/processed/interactions.csv')

    if file_path.exists():
        df = pd.read_csv(file_path)
    else:
        print("Error. No data")


    # Original features
    meta_df.loc[i, 'num_users'] = df['user'].nunique()
    meta_df.loc[i, 'num_items'] = df['item'].nunique()
    meta_df.loc[i, 'num_inter'] = len(df)
    meta_df.loc[i, 'density'] = round((len(df)/(df['user'].nunique() * df['item'].nunique())),6)
    meta_df.loc[i, 'user_item_ratio'] = round((df['user'].nunique()/df['item'].nunique()),6)
    meta_df.loc[i, 'item_user_ratio'] = round((df['item'].nunique()/df['user'].nunique()),6)
    meta_df.loc[i, 'highest_num_rating_u'] = df['user'].value_counts().iloc[0]
    meta_df.loc[i, 'lowest_num_rating_u'] = df['user'].value_counts().iloc[-1]
    meta_df.loc[i, 'highest_num_rating_i'] = df['item'].value_counts().iloc[0]
    meta_df.loc[i, 'lowest_num_rating_i'] = df['item'].value_counts().iloc[-1]
    meta_df.loc[i, 'mean_num_rating_u'] = df['user'].value_counts().mean()
    meta_df.loc[i, 'mean_num_rating_i'] = df['item'].value_counts().mean()


    # Calculates the percentage of interactions against the overall number of interactions
    meta_df.loc[i, 'p_inter_top_user'] = round(((df['user'].value_counts().iloc[0])/(len(df))), 6)
    meta_df.loc[i, 'p_top10_to_overall'] = round(((df['user'].value_counts().iloc[:10].sum())/(len(df))), 6)
    meta_df.loc[i, 'p_top5_to_overall'] = round(((df['user'].value_counts().iloc[:5].sum())/(len(df))), 6)
    meta_df.loc[i, 'p_low5_to_overall'] = round(((df['user'].value_counts().iloc[-5:].sum())/(len(df))), 6)

    # Calculates the total number of users within the top n-percentage of interactions
    meta_df.loc[i, 'num_users_top_10_p'] = get_top_n_percentage(df, col= 'user', threshold= 0.1)
    meta_df.loc[i, 'num_users_top_20_p'] = get_top_n_percentage(df, col= 'user', threshold= 0.2)
    meta_df.loc[i, 'num_users_top_30_p'] = get_top_n_percentage(df, col= 'user', threshold= 0.3)
    meta_df.loc[i, 'num_users_top_40_p'] = get_top_n_percentage(df, col= 'user', threshold= 0.4)
    meta_df.loc[i, 'num_users_top_50_p'] = get_top_n_percentage(df, col= 'user', threshold= 0.5)
    meta_df.loc[i, 'num_users_top_60_p'] = get_top_n_percentage(df, col= 'user', threshold= 0.6)
    meta_df.loc[i, 'num_users_top_70_p'] = get_top_n_percentage(df, col= 'user', threshold= 0.7)
    meta_df.loc[i, 'num_users_top_80_p'] = get_top_n_percentage(df, col= 'user', threshold= 0.8)
    meta_df.loc[i, 'num_users_top_90_p'] = get_top_n_percentage(df, col= 'user', threshold= 0.9)

    # Calculates the total number of items within the top n-percentage of interactions
    meta_df.loc[i, 'num_items_top_10_p'] = get_top_n_percentage(df, col= 'item', threshold= 0.1)
    meta_df.loc[i, 'num_items_top_20_p'] = get_top_n_percentage(df, col= 'item', threshold= 0.2)
    meta_df.loc[i, 'num_items_top_30_p'] = get_top_n_percentage(df, col= 'item', threshold= 0.3)
    meta_df.loc[i, 'num_items_top_40_p'] = get_top_n_percentage(df, col= 'item', threshold= 0.4)
    meta_df.loc[i, 'num_items_top_50_p'] = get_top_n_percentage(df, col= 'item', threshold= 0.5)
    meta_df.loc[i, 'num_items_top_60_p'] = get_top_n_percentage(df, col= 'item', threshold= 0.6)
    meta_df.loc[i, 'num_items_top_70_p'] = get_top_n_percentage(df, col= 'item', threshold= 0.7)
    meta_df.loc[i, 'num_items_top_80_p'] = get_top_n_percentage(df, col= 'item', threshold= 0.8)
    meta_df.loc[i, 'num_items_top_90_p'] = get_top_n_percentage(df, col= 'item', threshold= 0.9)

    
    # Takes the interaction numbers of top and bottom users to calculate the mean
    meta_df.loc[i, 'mean_inter_top10_user'] = round(((df['user'].value_counts().iloc[:10].mean())/(len(df))), 6)
    meta_df.loc[i, 'mean_inter_top5_user'] = round(((df['user'].value_counts().iloc[:5].mean())/(len(df))), 6)
    meta_df.loc[i, 'mean_inter_low10_user'] = round(((df['user'].value_counts().iloc[-10:].mean())/(len(df))), 6)
    meta_df.loc[i, 'mean_inter_low5_user'] = round(((df['user'].value_counts().iloc[-5:].mean())/(len(df))), 6)

    # Takes top and bottom users's interaction numbers and compare them against each other
    meta_df.loc[i, 'top10tobottom10_user'] = round(((df['user'].value_counts().iloc[:10].mean())/(df['user'].value_counts().iloc[-10:].mean())), 6)
    meta_df.loc[i, 'top10tobottom5_user'] = round(((df['user'].value_counts().iloc[:10].mean())/(df['user'].value_counts().iloc[-5:].mean())), 6)

    # Gini coefficient and shannon entropy
    meta_df.loc[i, 'gini_coefficient'] = round(calculate_gini(df, item_col= 'item'), 6)
    meta_df.loc[i, 'shannon_entropy'] = round(get_normalized_entropy(df, item_col= 'item'),6)

    # The head to tail ratio for items
    top_p = int(df['item'].nunique() * 0.2)
    meta_df.loc[i, 'head_to_tail_ratio'] = round((df['item'].value_counts().iloc[:top_p].sum())/( df['item'].value_counts().iloc[top_p:].sum()), 6)

    # No. of total interactions divided by number of unique items
    meta_df.loc[i, 'inter_to_unique_items'] = round((len(df)/df['item'].nunique()), 6)

    # Number and Percentage of coldstart users against all unique items
    temp_df = df['user'].value_counts()
    meta_df.loc[i, 'num_coldstart_users'] = len( temp_df[ temp_df <= 2])
    meta_df.loc[i, 'p_coldstart_users'] = round((len( temp_df[ temp_df <= 2])/ df['user'].nunique()), 6)

    # Number and Percentage of low interaction items (below 3 interactions)
    temp_df = df['item'].value_counts()
    meta_df.loc[i, 'num_low_inter_items'] = len( temp_df[ temp_df <= 2])
    meta_df.loc[i, 'p_low_inter_items'] = round((len( temp_df[ temp_df <= 2])/ df['user'].nunique()), 6)

    # Number and percentage of users that interacted with low inter(<=2) items
    low_i_user_data = users_inter_with_low_prob_items(df, inter_threshold= 2)
    meta_df.loc[i, 'num_u_i_low_inter_items'] = low_i_user_data[0]
    meta_df.loc[i, 'p_u_i_low_inter_items'] = low_i_user_data[1]

    # Number and percentage of low interaction items (below 5)
    temp_df = df['item'].value_counts()
    meta_df.loc[i, 'num_low_inter_items_5'] = len( temp_df[ temp_df <= 5])
    meta_df.loc[i, 'p_low_inter_items_5'] = round((len( temp_df[ temp_df <= 5])/ df['user'].nunique()), 6)

    # Number and percentage of items above 5 interactions
    temp_df = df['item'].value_counts()
    meta_df.loc[i, 'num_high_inter_items_5'] = len( temp_df[ temp_df > 5])
    meta_df.loc[i, 'p_high_inter_items_5'] = round((len( temp_df[ temp_df > 5])/ df['user'].nunique()), 6)

    # Get overall standard deviation, variance and coefficient of variance for item interactions
    overall_items_metrics = top_bottom_metrics(df, item_col= 'item', top= True, num= df['item'].nunique())
    meta_df.loc[i, 'overall_items_variance'] = round(overall_items_metrics[0], 6)               
    meta_df.loc[i, 'overall_items_std_dev'] = round(overall_items_metrics[1], 6)                       
    meta_df.loc[i, 'overall_items_CV'] = round(overall_items_metrics[2], 6)                            

    # Get item variance std deviation for top
    top10_items_metrics = top_bottom_metrics(df, item_col= 'item', top= True, num= 10)
    meta_df.loc[i, 'top10_items_variance'] = round(top10_items_metrics[0], 6)
    meta_df.loc[i, 'top10_items_std_dev'] = round(top10_items_metrics[1], 6)
    meta_df.loc[i, 'top10_items_CV'] = round(top10_items_metrics[2], 6)

    # Get item variance std deviation for bottom
    bottom10_items_metrics = top_bottom_metrics(df, item_col= 'item', top= False, num= 10)
    meta_df.loc[i, 'bottom10_items_variance'] = round(bottom10_items_metrics[0], 6)
    meta_df.loc[i, 'bottom10_items_std_dev'] = round(bottom10_items_metrics[1], 6)
    meta_df.loc[i, 'bottom10_items_CV'] = round(bottom10_items_metrics[2], 6)

    # Get number of duplicate profiles( profiles with similar item interaction patterns)
    num_duplicate_sets, num_duplicates, freq_common_u_pattern, i_mass_most_common_profile = find_duplicates(df, threshold_similarity= 1)
    meta_df.loc[i, 'num_duplicate_u_profiles'] = num_duplicates
    meta_df.loc[i, 'num_duplicate_u_profile_sets'] = num_duplicate_sets
    meta_df.loc[i, 'ratio_duplicate_users'] = round((num_duplicates/df['user'].nunique()),6)
    meta_df.loc[i, 'freq_common_u_pattern'] = freq_common_u_pattern
    meta_df.loc[i, 'num_onetime_patterns'] = df['user'].nunique() - num_duplicates
    meta_df.loc[i, 'i_mass_most_common_prof'] = i_mass_most_common_profile

    # Get number of user profiles above a similarity threshold of 0.9
    num_similar_0_9_sets, num_similar_0_9_profiles, freq_similar_0_9_u_pattern, _ = find_duplicates(df, threshold_similarity= 0.9)
    meta_df.loc[i, 'num_similar_0_9_u_profiles'] = num_similar_0_9_profiles
    meta_df.loc[i, 'num_similar_0_9_u_profile_sets'] = num_similar_0_9_sets
    meta_df.loc[i, 'ratio_similar_0_9_users'] = round((num_similar_0_9_profiles/df['user'].nunique()),6)
    meta_df.loc[i, 'freq_similar_0_9_u_pattern'] = freq_similar_0_9_u_pattern

    # Extract the singular value decomp features
    svd = SVDFeatures(df)
    meta_features = svd.extract_svd_features()
    meta_df.loc[i, 'svd_top1_ratio'] = round(meta_features['svd_top1_ratio'], 6)
    meta_df.loc[i, 'svd_top5_ratio'] = round(meta_features['svd_top5_ratio'], 6)
    meta_df.loc[i, 'svd_entropy'] = round(meta_features['svd_entropy'], 6)
    meta_df.loc[i, 'svd_energy_90'] = round(meta_features['svd_energy_90'], 6)  
    meta_df.loc[i, 'svd_slope'] = round(meta_features['svd_slope'], 6)

    # Extract the item cohesion meta features
    meta_features = svd.extract_item_cohesion(sample_size= 1000)
    meta_df.loc[i, 'item_cohesion_mean'] = round(meta_features['item_cohesion_mean'],6)
    meta_df.loc[i, 'item_cohesion_std'] = round(meta_features['item_cohesion_std'],6)
    meta_df.loc[i, 'item_cohesion_skew'] = round(meta_features['item_cohesion_skew'],6)

    # Extract the reconstruction error
    meta_features = svd.extract_reconstruction_error()
    meta_df.loc[i, 'latent_error_ratio'] = round(meta_features['latent_error_ratio'],6)
    meta_df.loc[i, 'latent_explained_variance'] = round(meta_features['latent_explained_variance'],6)

    # Extract cetroid gap between latent spaces
    meta_features = svd.extract_centroid_gap()
    meta_df.loc[i, 'centroid_gap_euclidean'] = round(meta_features['centroid_gap_euclidean'],6)
    meta_df.loc[i, 'centroid_gap_cosine'] = round(meta_features['centroid_gap_cosine'],6)

    # Extract latent dispersion for users and items
    meta_features = svd.extract_latent_dispersion()
    meta_df.loc[i, 'user_dispersion_mean'] = round(meta_features['user_dispersion_mean'],6)
    meta_df.loc[i, 'user_dispersion_std'] = round(meta_features['user_dispersion_std'],6)
    meta_df.loc[i, 'user_dispersion_cv'] = round(meta_features['user_dispersion_cv'],6)

    meta_df.loc[i, 'item_dispersion_mean'] = round(meta_features['item_dispersion_mean'],6)                   
    meta_df.loc[i, 'item_dispersion_std'] = round(meta_features['item_dispersion_std'],6)
    meta_df.loc[i, 'item_dispersion_cv'] = round(meta_features['item_dispersion_cv'],6)


    # Extract meta features from k-modes clustering


    k_modes = KModesClusteringFeatures(df= df)
    meta_features = k_modes.extract_kmodes_meta_features()
    meta_df.loc[i, 'kmodes_total_cost'] = round(meta_features['kmodes_total_cost'],6)
    meta_df.loc[i, 'kmodes_avg_cost'] = round(meta_features['kmodes_avg_cost'],6)
    meta_df.loc[i, 'kmodes_iterations'] = round(meta_features['kmodes_iterations'],6)
    meta_df.loc[i, 'kmodes_normalized_cost'] = round(meta_features['kmodes_normalized_cost'],6)

    # Extract meta features measuring distribution of the clusters themselves
    meta_features = k_modes.extract_advanced_kmodes_meta_features()
    meta_df.loc[i, 'kmodes_cluster_size_mean'] = round(meta_features['kmodes_cluster_size_mean'],6)
    meta_df.loc[i, 'kmodes_cluster_size_std_dev'] = round(meta_features['kmodes_cluster_size_std_dev'],6)              
    meta_df.loc[i, 'kmodes_cluster_size_gini'] = round(meta_features['kmodes_cluster_size_gini'],6) 
    meta_df.loc[i, 'kmodes_intra_dissimilarity_mean'] = round(meta_features['kmodes_intra_dissimilarity_mean'],6) 
    meta_df.loc[i, 'kmodes_intra_dissimilarity_std'] = round(meta_features['kmodes_intra_dissimilarity_std'],6)               
    meta_df.loc[i, 'kmodes_mode_entropy_mean'] = round(meta_features['kmodes_mode_entropy_mean'],6) 
    meta_df.loc[i, 'kmodes_mode_overlap_mean'] = round(meta_features['kmodes_mode_overlap_mean'],6) 

    # Extract meta features indicating the quality and reliability of the clusters from kmodes
    meta_features = k_modes.extract_quality_kmodes_meta_features()
    meta_df.loc[i, 'kmodes_silhouette_hamming_mean'] = round(meta_features['kmodes_silhouette_hamming_mean'],6) 
    meta_df.loc[i, 'kmodes_silhouette_hamming_median'] = round(meta_features['kmodes_silhouette_hamming_median'],6) 
    meta_df.loc[i, 'kmodes_cluster_purity_popularity'] = round(meta_features['kmodes_cluster_purity_popularity'],6) 
    meta_df.loc[i, 'kmodes_empty_clusters'] = round(meta_features['kmodes_empty_clusters'],6) 

    # Extract meta features describing the behavior of the kmodes clusters
    meta_features = k_modes.extract_behavioral_meta_features()
    meta_df.loc[i, 'kmodes_modes_sparsity_mean'] = round(meta_features['kmodes_modes_sparsity_mean'],6) 
    meta_df.loc[i, 'kmodes_modes_sparsity_std'] = round(meta_features['kmodes_modes_sparsity_std'],6)                    
    meta_df.loc[i, 'kmodes_rank_corr_mean'] = round(meta_features['kmodes_rank_corr_mean'],6) 
    meta_df.loc[i, 'kmodes_rank_corr_std'] = round(meta_features['kmodes_rank_corr_std'],6)   
    meta_df.loc[i, 'kmodes_centroid_dist_mean'] = round(meta_features['kmodes_centroid_dist_mean'],6) 
    meta_df.loc[i, 'kmodes_centroid_dist_skew'] = round(meta_features['kmodes_centroid_dist_skew'],6) 

    # Extract meta features indicating the robustness of the kmodes clusters
    meta_features = k_modes.measure_kmodes_stability()
    meta_df.loc[i, 'kmodes_stability_avg_drift'] = round(meta_features['kmodes_stability_avg_drift'],6) 
    meta_df.loc[i, 'kmodes_stability_variance'] = round(meta_features['kmodes_stability_variance'],6) 



    # # Extract meta-features from Bipartite graph
    # bp_graph = BipartiteGraphFeatures(df)
    # meta_features = bp_graph.extract_centrality_connectivity_features()
    # meta_df.loc[i, 'mean degrees' ] = round(meta_features['mean_degrees'],6)
    # meta_df.loc[i, 'degree_skew' ] = round(meta_features['degree_skew'],6)
    # meta_df.loc[i, 'mean_pagerank' ] = round(meta_features['mean_pagerank'],6)
    # meta_df.loc[i, 'avg_clustering' ] = round(meta_features['avg_clustering'],6)
    # meta_df.loc[i, 'lcc_ratio' ] = round(meta_features['lcc_ratio'],6)

    # meta_features = bp_graph.extract_auth_and_kurtosis()
    # meta_df.loc[i, 'auth_mean' ] = round(meta_features['auth_mean'],6)
    # meta_df.loc[i, 'auth_variance' ] = round(meta_features['auth_variance'],6)
    # meta_df.loc[i, 'auth_kurtosis' ] = round(meta_features['auth_kurtosis'],6)
    


    # Extract meta features via Decision Trees
    tree = DecisionTreeFeatures(df, num_features= -2)
    meta_features = tree.extract_tree_meta_features()

    # Extract meta features on the tree structure
    meta_df.loc[i, 'tree_node_count'] = round(meta_features['tree_node_count'],10)
    meta_df.loc[i, 'tree_leaf_count'] = round(meta_features['tree_leaf_count'],10)
    meta_df.loc[i, 'tree_max_depth'] = round(meta_features['tree_max_depth'],10)
    meta_df.loc[i, 'tree_avg_depth'] = round(meta_features['tree_avg_depth'],10)
    meta_df.loc[i, 'tree_impurity'] = round(meta_features['tree_impurity'],10)

    # Extract meta features based on importance of tree features  
    meta_features = tree.extract_importance_meta_features()
    meta_df.loc[i, 'tree_importance_skew'] = round(meta_features['tree_importance_skew'],10)
    meta_df.loc[i, 'tree_n_features_to_90'] = round(meta_features['tree_n_features_to_90'],10)
    meta_df.loc[i, 'tree_n_features_to_80'] = round(meta_features['tree_n_features_to_80'],10)
    meta_df.loc[i, 'tree_n_features_to_70'] = round(meta_features['tree_n_features_to_70'],10)
    meta_df.loc[i, 'tree_n_features_to_60'] = round(meta_features['tree_n_features_to_60'],10)
    # MISSING A ROW
    meta_df.loc[i, 'tree_top_1p_concentration'] = round(meta_features['tree_top_1p_concentration'],10)
    meta_df.loc[i, 'tree_top_5p_concentration'] = round(meta_features['tree_top_5p_concentration'],10)
    meta_df.loc[i, 'tree_importance_decay_ratio'] = round(meta_features['tree_importance_decay_ratio'],10)

    # Extract meta features pertaining to the structure of tree
    meta_features = tree.extract_advanced_tree_meta_features()
    meta_df.loc[i, 'tree_avg_leaf_size'] = round(meta_features['tree_avg_leaf_size'],10)
    meta_df.loc[i, 'tree_leaf_size_cv'] = round(meta_features['tree_leaf_size_cv'],10)
    meta_df.loc[i, 'tree_balance_ratio'] = round(meta_features['tree_balance_ratio'],10)
    meta_df.loc[i, 'tree_feature_utilization'] = round(meta_features['tree_feature_utilization'],10)
    meta_df.loc[i, 'tree_feature_redundancy'] = round(meta_features['tree_feature_redundancy'],10)

    # Extract meta features pertaining to impurity reduction of tree
    meta_features = tree.extract_impurity_reduction()
    meta_df.loc[i, 'tree_total_impurity_reduction'] = round(meta_features['tree_total_impurity_reduction'],10)
    meta_df.loc[i, 'tree_normalized_improvement'] = round(meta_features['tree_normalized_improvement'],10)

    # Extract quality and reliability metrics of trees
    meta_features = tree.extract_performance_meta_features()
    meta_df.loc[i, 'tree_train_score'] = round(meta_features['tree_train_score'],10)
    meta_df.loc[i, 'tree_val_score'] = round(meta_features['tree_val_score'],10)
    meta_df.loc[i, 'tree_overfit_gap'] = round(meta_features['tree_overfit_gap'],10)
    meta_df.loc[i, 'tree_minmax_prob_gap'] = round(meta_features['tree_minmax_prob_gap'],10)

    # Extract average path length of tree
    meta_features = tree.extract_avg_path_length()
    meta_df.loc[i, 'tree_avg_path_length' ] = round(meta_features['tree_avg_path_length' ],10)

    # Extracts the bootstrapping stability of tree
    meta_features = tree.extract_stability_bootstrap()
    meta_df.loc[i, 'tree_stability_bootstrap'] = round(meta_features['tree_stability_bootstrap'],10)

                                                    
    print(meta_df)
    meta_df.to_csv('temp_df.csv', index=False)



    
meta_df = meta_df.reset_index(drop= True)

print(meta_df)
meta_df.to_csv('unpruned_updated_df.csv', index=False)

# filled_meta_df = fill_df(meta_df, fill= 'mean')
# filled_meta_df.to_csv('unpruned_filled_df.csv', index= False)



# Copyright (c) 2023 Maike Sonnewald
# Modified from the original https://github.com/maikejulie/NEMI
# by Laique Djeutchouang

import re
import pandas as pd
import numpy as np

import umap
import pickle
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import DBSCAN


def apply_umap(dfn:np.ndarray, min_dist:float=0.5, umap_neighbors:int=200,
               learning_rate:float=1.0, n_epochs: int=None, init:str='random', n_jobs:int=-1) -> np.ndarray:
    """
    Apply UMAP dimensionality reduction.

    Args:
        dfn (np.ndarray): Standardized dataframe/ndarray (n_samples, n_features).
        min_dist (float): Minimum distance between points in the embedding. Defaults to 0.5.
        umap_neighbors (int): Number of neighbors for UMAP. Defaults to 200.
        learning_rate (float): Number of neighbors for UMAP. Defaults to 1.0.
        n_epochs (int): Optional, defaults to None. The number of training epochs to be
                        used in optimizing the low dimensional embedding.
        init (str): Initialization method for UMAP. Defaults to 'random'.
        n_jobs (int): Number of parallel jobs to run. Defaults to -1 (use all processors).

    Returns:
        np.ndarray: UMAP-transformed data.
    """
    
    # print('Embedding ...')
    model_umap = umap.UMAP(min_dist=min_dist, n_components=3, n_neighbors=umap_neighbors,
                           learning_rate=learning_rate, n_epochs=n_epochs, init=init, n_jobs=n_jobs)
    df_umap = model_umap.fit_transform(dfn)

    return df_umap


def get_sorted_clusters(df_umap, n_clusters=3, hclust_neighbors=40, n_jobs=-1):

    # Clustering
    knn_graph = kneighbors_graph(df_umap, n_neighbors=hclust_neighbors, n_jobs=n_jobs, include_self=False)
    model = AgglomerativeClustering(linkage='ward', connectivity=knn_graph, n_clusters=n_clusters)    
    clusters = model.fit_predict(df_umap)

    # Number of clusters (also the same as the label name in the agglomerated cluster dict)
    n_clusters = np.max(clusters) + 1
    
    # Create a histogram of the different clusters
    hist, _ = np.histogram(clusters, np.arange(n_clusters+1))
    
    # Clusters sorted by size (largest to smallest)
    sorted_clusters = np.argsort(hist)[::-1]
    
    # Assign new labels where labels 0,...,k go in decreasing member size 
    new_labels = np.empty(clusters.shape)
    new_labels.fill(np.nan)
    for new_label, old_label in enumerate(sorted_clusters):
        new_labels[clusters == old_label] = new_label
    
    return new_labels


def get_clusters(df_umap, n_clusters=9, hclust_neighbors=40, n_jobs=-1):

    # Clustering
    knn_graph = kneighbors_graph(df_umap, n_neighbors=hclust_neighbors, n_jobs=n_jobs, include_self=False)
    model = AgglomerativeClustering(linkage='ward', connectivity=knn_graph, n_clusters=n_clusters)    
    clusters = model.fit_predict(df_umap)
    
    return clusters


def run_agglom(df_umap, n_clusters=9, hclust_neighbors=40):
    """ Run Agglomerative Clustering on the UMAP-transformed data.
    Args:
        df_umap (np.ndarray): UMAP-transformed data.
        n_clusters (int): Number of clusters to form. Defaults to 9.
        hclust_neighbors (int): Number of neighbors for clustering. Defaults to 40.
    Returns:
        np.ndarray: Cluster labels for each point in the UMAP-transformed data.
    """
    knn_graph = kneighbors_graph(df_umap, n_neighbors=hclust_neighbors, n_jobs=-1, include_self=False)
    model = AgglomerativeClustering(linkage='ward', connectivity=knn_graph, n_clusters=n_clusters)    
    clusters = model.fit_predict(df_umap)
    
    return clusters


def run_dbscan(df_umap, eps=0.5, min_samples=5, metric='euclabel_idean'):
    """ Run DBSCAN clustering on the UMAP-transformed data.
    Args:
        df_umap (np.ndarray): UMAP-transformed data.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other. Defaults to 0.5.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point. Defaults to 5.
        metric (str): The metric to use when calculating distance between instances in a feature array. Defaults to 'euclabel_idean'.
    Returns:
        np.ndarray: Cluster labels for each point in the UMAP-transformed data.
    """
    print(f'DBSCAN: epsilon = {eps}')
    print(f'DBSCAN: min samples = {min_samples}')
    print(f'DBSCAN: metric = {metric}')

    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(df_umap)   
    clusters = model.labels_

    return clusters


def sort_clusters(clusters):
    """ Sort clusters by area and assign new labels.
    Args:
        clusters (np.ndarray): Cluster labels.
    Returns:
        np.ndarray: New labels for clusters sorted by size.
    """
    # Number of clusters (also the same as the label name in the agglomerated cluster dict)
    n_clusters = np.unique(clusters)
    
    # Create a histogram of the different clusters
    hist, _ = np.histogram(clusters, np.arange(n_clusters+1))
    
    # Clusters sorted by size (largest to smallest)
    sorted_clusters = np.argsort(hist)[::-1]
    
    # Assign new labels where labels 0,...,k go in decreasing member size 
    new_labels = np.empty(clusters.shape)
    new_labels.fill(np.nan)
    for new_label, old_label in enumerate(sorted_clusters):
        new_labels[clusters == old_label] = new_label
    
    return new_labels


def nemi_func(dfn, n_clusters=9, min_dist=0.5, umap_neighbors=200, hclust_neighbors=40):

    print('UMAP: min_dist = '+str(min_dist))
    print('UMAP: n_neighbors = '+str(umap_neighbors))
    print('HCLUST: n_neighbors = '+str(hclust_neighbors))
    print('HCLUST: no of clusters = '+str(n_clusters))

    # print('Embedding ...')
    df_umap = umap.UMAP(min_dist=min_dist, n_components=3, n_neighbors=umap_neighbors).fit_transform(dfn)
        
    # print('clustering..')
    knn_graph = kneighbors_graph(df_umap, n_neighbors=hclust_neighbors, include_self=False)
    model = AgglomerativeClustering(linkage='ward', connectivity=knn_graph, n_clusters=n_clusters)    
    clusters = model.fit_predict(df_umap)

    # print('Sorting ...')
    # Number of clusters (also the same as the label name in the agglomerated cluster dict)
    n_clusters = np.max(clusters) + 1
    
    # Create a histogram of the different clusters
    hist, _ = np.histogram(clusters, np.arange(n_clusters+1))
    
    # Clusters sorted by size (largest to smallest)
    sorted_clusters= np.argsort(hist)[::-1]
    
    # Assign new labels where labels 0,...,k go in decreasing member size 
    new_labels = np.empty(clusters.shape)
    new_labels.fill(np.nan)
    for new_label, old_label in enumerate(sorted_clusters):
        new_labels[clusters == old_label] = new_label

    print('Realisation finished')
    # q.put(new_labels)
    
    return df_umap, new_labels


# Finds clusters in different ensembles that has largest overlap with clusters in the base_label (ensemble 0 in this case)
def assess_overlap(ens_labels, label_id=0, num_members=50, max_clusters=None):

    base_id = copy.deepcopy(label_id)
    base_labels = ens_labels[base_id]
    compare_ids = [i for i in range(num_members)]
    compare_ids.pop(base_id)
    num_clusters = int(np.max(base_labels) + 1)

    # If not pre-set, set max number of clusters to total number of clusters in the base
    if max_clusters is None:
        max_clusters = copy.deepcopy(num_clusters)

    sortedOverlap = np.zeros((len(compare_ids)+1, max_clusters, base_labels.shape[0]))*np.nan

    print(num_clusters, max_clusters)
    summaryStats = np.zeros((num_clusters, max_clusters))

    # Compile sorted cluster data
    # TODO: add assert statement to make sure that the clusters have been sorted?

    # dataVector = [nemi.clusters for id, nemi in enumerate(self.nemi_pack) if id != base_id]
    dataVector = [ens_labels[id] for id, _ in enumerate(ens_labels) if id != base_id]

    # Loop over ensemble members, not including the base member
    for compare_cnt, compare_id in enumerate(compare_ids):
        
        # Grab clusters of ensemble member
        compare_labels = dataVector[compare_cnt]

        # Go through each cluster in the base and assess the percentage overlap
        # for every cluster in the ensemble member (overlap / total coverage area) 
        for c1 in range(max_clusters): 
            # Initialize dummy array to mark location of the cluster for the base member
            data1_M = np.zeros(base_labels.shape, dtype=int)
            
            # Mark where the considered cluster is in the member that is being used as the baseline
            data1_M[np.where(c1==base_labels)] = 1 
            
            # Count numer of entries [Why?] 
            summaryStats[0, c1] = np.sum(data1_M) 

            # Go through each cluster
            # k = 0
            for c2 in range(num_clusters):
                # Initialize dummy array to mark where the cluster is in the comparison member
                data2_M = np.zeros(base_labels.shape, dtype=int) 

                # Mark where the considered cluster is in the member that is being used as the comparison
                data2_M[np.where(c2==compare_labels)] = 1    

                # Sum of flags where the two datasets of that cluster are both present
                num_overlap = np.sum(data1_M*data2_M)       

                # Sum of where they overlap
                num_total = np.sum(data1_M | data2_M)       

                # Collect the number that is largest of k and the num_overlap/num_total
                # k = max(k, num_overlap / num_total)       
                summaryStats[c2, c1] = (num_overlap / num_total) * 100 # Add percentage of coverage

            # Filled in 'summaryStatistics' matrix results of percentage overlaps

        usedClusters = set() # Used to mak sure clusters don't get selected twice
        
        # Clusters are already sorted by size
        sortedOverlapForOneCluster = np.zeros(base_labels.shape, dtype=int)*np.nan
        
        # Go through clusters from (biggest to smallest since they are sorted)
        for c1 in range(max_clusters):  
            sortedOverlapForOneCluster = np.zeros(base_labels.shape, dtype=int)*np.nan
            #print('cluster number ', c1, summaryStats.shape, summaryStats[1:,c1-1].shape)

            # Find biggest cluster in first column, making sure it has not been used
            sortedClusters = np.argsort(summaryStats[:, c1])[::-1]
            biggestCluster = [ele for ele in sortedClusters if ele not in usedClusters][0]

            # Record it for later
            usedClusters.add(biggestCluster)

            # Initialize dummy array
            data2_M = np.zeros(base_labels.shape, dtype=int)

            # Select which country is being assessed
            data2_M[np.where(biggestCluster == compare_labels)]=1 # Select cluster being assessed

            sortedOverlapForOneCluster[np.where(data2_M==1)]=1
            sortedOverlap[compare_id, c1, :] = sortedOverlapForOneCluster

    # Fill in the base entry in the sorted overlap
    for c1 in range(max_clusters):  
        sortedOverlap[base_id, c1, :] = 1 * (base_labels == c1)

    # Majority vote
    aggOverlaps = np.nansum(sortedOverlap, axis=0)
    voteOverlaps = np.argmax(aggOverlaps, axis=0)

    # Save clusters estimated from the ensemble
    return voteOverlaps


# Sorts clusters by area, 0 => largest ..
def sort_by_area(voteOverlaps):
    
    clusters = copy.deepcopy(voteOverlaps)
    
    # print('sorting..')
    # Number of clusters (also the same as the label name in the agglomerated cluster dict)
    n_clusters = np.max(clusters) + 1
    
    # Create a histogram of the different clusters
    hist, _ = np.histogram(clusters, np.arange(n_clusters+1))
    
    # Clusters sorted by size (largest to smallest)
    sorted_clusters = np.argsort(hist)[::-1]
    
    # Assign new labels where labels 0,...,k go in decreasing member size 
    new_labels = np.empty(clusters.shape)
    new_labels.fill(np.nan)
    for new_label, old_label in enumerate(sorted_clusters):
        new_labels[clusters == old_label] = new_label

    return new_labels


# Overlap with NEMI clusters
def overlap_with_nemi_clusters(voteOverlaps, ensembles):
   
    # base_labels = sort_by_area(voteOverlaps)
    base_labels = copy.deepcopy(voteOverlaps)
    
    dataVector = [ensembles[id] for id, nemi in enumerate(ensembles)]
    alist = []
    npts = len(ensembles[0])
    max_clusters = int(np.max(base_labels) + 1)
    summaryStats = np.zeros((max_clusters, max_clusters))

    for compare_cnt, compare_id in enumerate([i for i in range(len(ensembles))]):
        # Grab clusters of ensemble member
        compare_labels = dataVector[compare_cnt]
    
        # Go through each cluster in the base and assess the percentage overlap
        # for every cluster in the ensemble member (overlap / total coverage area) 
        for c1 in range(max_clusters): 
            # Initialize dummy array to mark location of the cluster for the base member
            data1_M = np.zeros(base_labels.shape, dtype=int)
            
            # Mark where the considered cluster is in the member that is being used as the baseline
            data1_M[np.where(c1==base_labels)] = 1 # Locations whuch are in the 'c1' cluster (in base_label) are marked 1, others are zero
            
            # Count numer of entries [Why?] 
            summaryStats[0, c1] = np.sum(data1_M) # Counts no of samples/locations that are 'c1' cluster as in base_label
    
            # Go through each cluster
            # k = 0
            for c2 in range(max_clusters):
                # Initialize dummy array to mark where the cluster is in the comparison member
                data2_M = np.zeros(base_labels.shape, dtype=int) 
    
                # Mark where the considered cluster is in the member that is being used as the comparison
                data2_M[np.where(c2==compare_labels)] = 1 # locations which are recognised as 'c2' cluster in one of the ensembles are marked 1
    
                # Check locations that are marked, say, 0 th cluster in baseline (data1_M) are also marked 0th cluster in the other ensemble (compare_cnt)
                # Find the overlap
    
                # Sum of flags where the two datasets of that cluster are both present
                num_overlap = np.sum(data1_M*data2_M)       # when overlap both marked same cluster 1 * 1 = 1, else 1 * 0 = 0' then count 1's, i.e., count overlaps
    
                # Sum of where they overlap
                num_total = np.sum(data1_M | data2_M)       
    
                # Collect the number that is largest of k and the num_overlap/num_total
                # k = max(k, num_overlap / num_total)       
                # summaryStats[c2, c1] = (num_overlap / num_total)*100 # Add percentage of coverage
                summaryStats[c2, c1] = num_overlap
    
                # print(summaryStats[c2, c1])    
                # summaryStats, for each ensembles (excluding base_labels, ens0 in this case) gives the amount of overlap between clusters in that ensemble and the base_labels
    
    
    
            # np.save('ens'+str(compare_id),summaryStats)
            overlap_df = pd.DataFrame(summaryStats)
            overlap_df.index.rename('ens' + str(compare_id), inplace=True)
            overlap_df['max_intersection'] = overlap_df.max(axis=1)
            overlap_df = overlap_df.astype('int')
            print(overlap_df.max_intersection.sum() / npts)
            # display(overlap_df)
            # store_summaryStats[compare_id] = overlap_df 
            
            # Clusters are already sorted by size
            # alist.append(overlap_df.max_intersection.sum()/2816)
            # alist[compare_cnt] = overlap_df.max_intersection.sum()/2816
            alist.append(overlap_df.max_intersection.sum() / npts)
    
        return alist


# Return sorted overlap for entropy calculation
def get_sortedOverlap(ensembles, id=0, num_members=3, max_clusters=None):

    base_id = copy.deepcopy(id)
    base_labels = ensembles[base_id]
    compare_ids = [i for i in range(num_members)]
    compare_ids.pop(base_id)
    num_clusters = int(np.max(base_labels) + 1)

    # If not pre-set, set max number of clusters to total number of clusters in the base
    if max_clusters is None:
        max_clusters = num_clusters

    sortedOverlap = np.zeros((len(compare_ids) + 1, max_clusters, base_labels.shape[0]))*np.nan

    print(num_clusters, max_clusters)
    summaryStats = np.zeros((num_clusters, max_clusters))

    # Compile sorted cluster data
    # TODO: add assert statement to make sure that the clusters have been sorted?

    # dataVector = [nemi.clusters for id, nemi in enumerate(self.nemi_pack) if id != base_id]
    dataVector = [ensembles[id] for id, nemi in enumerate(ensembles) if id != base_id]

    # Loop over ensemble members, not including the base member
    for compare_cnt, compare_id in enumerate(compare_ids):
        # Grab clusters of ensemble member
        compare_labels = dataVector[compare_cnt]

        # Go through each cluster in the base and assess the percentage overlap
        # for every cluster in the ensemble member (overlap / total coverage area) 
        for c1 in range(max_clusters): 
            # Initialize dummy array to mark location of the cluster for the base member
            data1_M = np.zeros(base_labels.shape, dtype=int)
            
            # Mark where the considered cluster is in the member that is being used as the baseline
            data1_M[np.where(c1==base_labels)] = 1 
            
            # Count numer of entries [Why?] 
            summaryStats[0, c1] = np.sum(data1_M) 

            # Go through each cluster
            # k = 0
            for c2 in range(num_clusters):
                # Initialize dummy array to mark where the cluster is in the comparison member
                data2_M = np.zeros(base_labels.shape, dtype=int) 

                # Mark where the considered cluster is in the member that is being used as the comparison
                data2_M[np.where(c2==compare_labels)] = 1    

                # Sum of flags where the two datasets of that cluster are both present
                num_overlap = np.sum(data1_M*data2_M)       

                # Sum of where they overlap
                num_total = np.sum(data1_M | data2_M)       

                # Collect the number that is largest of k and the num_overlap/num_total
                # k = max(k, num_overlap / num_total)       
                summaryStats[c2, c1] = (num_overlap / num_total) * 100 # Add percentage of coverage

            # Filled in 'summaryStatistics' matrix results of percentage overlaps
            
        usedClusters = set() # Used to mak sure clusters don't get selected twice
        
        # Clusters are already sorted by size
        sortedOverlapForOneCluster = np.zeros(base_labels.shape, dtype=int)*np.nan
        
        # Go through clusters from (biggest to smallest since they are sorted)
        for c1 in range(max_clusters):  
            sortedOverlapForOneCluster = np.zeros(base_labels.shape, dtype=int)*np.nan
            #print('cluster number ', c1, summaryStats.shape, summaryStats[1:,c1-1].shape)

            # find biggest cluster in first column, making sure it has not been used
            sortedClusters = np.argsort(summaryStats[:, c1])[::-1]
            biggestCluster = [ele for ele in sortedClusters if ele not in usedClusters][0]

            # record it for later
            usedClusters.add(biggestCluster)

            # Initialize dummy array
            data2_M = np.zeros(base_labels.shape, dtype=int)

            # Select which country is being assessed
            data2_M[np.where(biggestCluster == compare_labels)] = 1 # Select cluster being assessed

            sortedOverlapForOneCluster[np.where(data2_M==1)] = 1
            sortedOverlap[compare_id, c1, :] = sortedOverlapForOneCluster

    # Fill in the base entry in the sorted overlap
    for c1 in range(max_clusters):  
        sortedOverlap[base_id, c1, :] = 1 * (base_labels == c1)

    return sortedOverlap


def time_now():

    from datetime import datetime
    now = datetime.now()

    return now.strftime("%H:%M:%S")


# Addtion to the above functions
# By Laique Djeutchouang

# Signed log-transformer
class SignedLogTransformer():
    """
    This transformer applies a signed log transformation to the input data.
    """
    def __init__(self, epsilon=1e-20):
        self.epsilon = epsilon # An infinitesimal positive value (offset) to avoid log(0)

    def fit_transform(self, x):
        scaled_x = np.sign(x) * np.log10(np.abs(x) + self.epsilon)  # Add tiny offset to avoid log(0)
        
        return scaled_x


# Power10 transformer
class Power10Transformer():
    """
    This transformer scales the input data by a factor of 10. It is useful for 
    transforming data that is close to zero to a more manageable range. This is 
    particularly useful for data that is very small or very large. The scaling 
    factor can be adjusted to suit the specific needs of the data.
    """
    def __init__(self, scale_factor=1e10):
        self.scale_factor = scale_factor # An infinitesimal positive value to bring values close to (-1, 1).

    def fit_transform(self, x):
        scaled_x =  self.scale_factor * x # Scale the data
        
        return scaled_x


# Dictionary of scalers
scalers = {'Quantile-Normal': [QuantileTransformer, 'qtnormalscaled'],
           'Quantile-Uniform': [QuantileTransformer, 'qtuniformscaled'],
           'Robust': [RobustScaler, 'robustscaled'],
           'Standard': [StandardScaler, 'standardscaled'], 
           'Signed-Log': [SignedLogTransformer, 'signedlogscaled'],
           'Power-10': [Power10Transformer, 'power10scaled'],}


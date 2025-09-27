"""
This mudule exports a single dataframe with entropy for all base label IDs.
It dentifies the base label ID with the least entropy. Finally, it exports entropy
files with respect to that base_label
"""

import os
import sys
import re
import glob
import copy
import time
import numpy as np
import pandas as pd
from collections import OrderedDict

# Add path to local modules and import them
source = os.path.abspath('/home/djeutsch/Projects/seaLevelRegimes')
if source not in sys.path:
    sys.path.insert(1, source)

from src import nemi_func as nf # Importing the nemi_func module
from src import aux_func as af # Importing the aux_func module




def load_cluster_based_entropy(entropy_dir:str, nc:int) -> OrderedDict:
    """
    Load ncluster-based entropy data for all base label IDs.

    Args:
        entropy_dir (str): Directory from which the entropy files are loaded.
        nc (int): Number of clusters.

    Returns:
        OrderedDict: A dictionary of minimum entropy data sorted by keys.
    """
    filenames = glob.glob(f'{entropy_dir}/nclusters_{nc}/base_label_id_*.csv')
    
    min_ent = {}
    for filename in filenames:
        num = af.get_numbers_from_filename(filename.split('/')[-1])
        md = float(num[2] + '.' + num[3])
        knn = int(num[1])
        bid = int(num[0])
        min_ent[(bid, knn, md)] = pd.read_csv(filename)
        
    # Sort the dictionary by keys
    sorted_min_entropy_dict = OrderedDict(sorted(min_ent.items(), key=lambda t: t[0]))

    return sorted_min_entropy_dict



def concatenate_entropy_data(nc:int, emb_params:list[tuple], entropy_dir:str):
    """
    Concatenate loaded ncluster-based entropy data for all UMAP key 
    parameter combinations (emb_params) and save to CSV.

    Args:
        nc (int): Number of clusters.
        emb_params (list[tuple]): List of UMAP key parameter combinations - minimum distances & nearest neighhbors.
        entropy_dir (str): Directory where the entropy files are stored.
    """
    # Load ncluster-based entropy data for all base label IDs.
    base_label_dict = load_cluster_based_entropy(entropy_dir, nc)
    
    # Define output directory
    output_dir = f'{entropy_dir}/nclusters_{nc}'
    os.makedirs(output_dir, exist_ok=True)
    
    for knn, md in emb_params:
        
        # Filter the base label dictionary for the current knn and md
        filtered_values = [
            value for (a, b, c), value in base_label_dict.items()
            if b == knn and c == md]

        # Concatenate the filtered values into a single DataFrame and save to CSV
        df = pd.concat(filtered_values)
        output_filename = f'{output_dir}/n_neighbors_{knn}_min_dist_{md}_nclusters_{nc}_mean_entropy.csv'
        df.to_csv(output_filename, index=False)
    print(f"\nAll concatenated entropy data for knn x md cominations saved at: {output_dir}")



def export_min_entropy_labels(nc:int, entropy_dir:str):
    """
    Export base labels with minimum entropy to a text file.

    Args:
        nc (int): Number of clusters.
        entropy_dir (str): Base directory where the entropy files are stored.
    """
    # Define data directory and the filenames to process
    data_dir = f'{entropy_dir}/nclusters_{nc}'
    filenames = glob.glob(f'{data_dir}/*_nclusters_{nc}_mean_entropy.csv')
    
    # Extract base labels with minimum entropy and save to a text file
    # Initialize an empty dictionary to store the minimum entropy data and
    # the corresponding base labels for each parameter combination (knn, md)
    min_ent = {}
    for filename in filenames:
        num = af.get_numbers_from_filename(filename.split('/')[-1])
        md = float(num[1] + '.' + num[2])
        knn = int(num[0])

        df = pd.read_csv(filename, on_bad_lines='skip')
        min_ent[(knn, md)] = df.mean_entropy.idxmin()

    ordered_base_label_dict = OrderedDict(sorted(min_ent.items(), key=lambda t: t[0]))
    lst = list(ordered_base_label_dict.values())
    
    output_filename = f'{data_dir}/base_label_min_entropy_nclusters_{nc}.txt'
    with open(output_filename, 'w') as file:
        file.write('\n'.join(str(itm) for itm in lst))
    print(f"Exported base labels with minimum entropy to {output_filename}\n")



def process_clusters_entropy(nc:int, lst:list, emb_params:list[tuple],
                             base_dir:str, n_ens:int=20):
    """
    Process clusters and calculate entropy for each parameter combination.

    Args:
        nc (int): Number of clusters.
        lst (list): List of base labels with minimum entropy.
        emb_params (list[tuple]): List of UMAP key parameter combinations - minimum distances & nearest neighhbors.
        base_dir (str): Base directory where the clustering and entropy files are stored.
        n_ens  (int, optional): Number of ensembles. Defaults to 50.
    """
    # Define data base directory and load clusters data
    clusters_dir = f'{base_dir}/clusterings'
    sorted_nclusters_dict, ncluster_size = af.load_clusters(nc=nc, clusters_dir=clusters_dir)

    # Initialize parameters
    n_param = len(emb_params)
    n_pts = copy.deepcopy(ncluster_size)
    
    # Initialize labels array
    lab = np.zeros((n_param, n_ens, n_pts)) * np.nan
    for k1, k2 in enumerate(range(1, n_ens + 1)):
        for idx, (knn, md) in enumerate(emb_params):
            lab[idx, k1, :] = sorted_nclusters_dict[(k2, knn, md)]

    # Calculate entropy
    entropy_list = []
    print(f"Entropy for embedded clustering (#clusters = {nc}) with:")
    for id in range(n_param):
        labels = lab[id, :, :]
        sorted_overlap = af.get_overlap(labels, id=lst[id], n_ens=n_ens)
        sorted_overlap_ = np.nan_to_num(sorted_overlap)
        df = pd.DataFrame(np.argmax(sorted_overlap_, axis=1)).T
        df_c = af.get_ent(df)
        entropy_list.append(df_c.entropy)
        print(f"{' '*3}{id}: n_neighbors = {emb_params[id][0]}, min_dist = {emb_params[id][1]}")

    # Save entropy data
    output_dir = f'{base_dir}/entropy/nclusters_{nc}'
    os.makedirs(output_dir, exist_ok=True)
    for id, entropy in enumerate(entropy_list):
        knn, md = emb_params[id]
        output_filename = f'{output_dir}/entropy_n_neighbors_{knn}_min_dist_{md}_nclusters_{nc}.npy'
        if not os.path.exists(output_filename):
            np.save(output_filename, entropy)
        else:
            # If the file already exists, skip saving
            print(f"File {output_filename} already exists. Skipping save.")
            continue     

    print(f"\nAll entropy data array for ncluster-{nc}-based knn x md cominations saved at: {output_dir}")



def run_entropy_file(nc:int, n_ens:int=20, region:str='Global', scaler:str='Standard'):
    """
    Main function to execute the entropy calculation for a given cluster.
    
    Args:
        nc (int): Number of clusters.
        n_ens (int, optional): Number of ensembles. Defaults to 20.
        region (str, optional): Region of interest. Defaults to 'Global'.
        scaler (str, optional): Data scaler used. Defaults to 'Standard'.
    """ 
    # List of UMAP key parameter combinations - minimum distances & nearest neighhbors.
    emb_params = af.UMAP_KNN_MIN_DIST
    
    print("\n-------------- STARTING PROCESSING ENTROPY FILES ----------------\n")

    # Record the start time
    start_now = nf.time_now()
    print(f"Process started at: {start_now}")

    print(f"Ensembles = {n_ens}")
    print(f'NEMI Param: n_cluster = {nc}')
    print(f"Region of interest: {region}")
    print(f"Data scaler used: {scaler}")  

    # Define directories of interest
    base_dir = f'/group/maikesgrp/laique/NOAA/nemis/{region}/{scaler}'
    entropy_dir = f'{base_dir}/entropy' 

    # Load and concatenate ncluster-based entropy data for all base label IDs.
    concatenate_entropy_data(nc=nc, emb_params=emb_params, entropy_dir=entropy_dir)
    
    # Export base labels with minimum entropy to a text file
    export_min_entropy_labels(nc=nc, entropy_dir=entropy_dir)

    # Load clusters and process entropy for each parameter combination
    with open(f'{entropy_dir}/nclusters_{nc}/base_label_min_entropy_nclusters_{nc}.txt', 'r') as file:
        lst = [int(line.strip()) for line in file.readlines()]
    process_clusters_entropy(nc=nc, lst=lst, emb_params=emb_params, base_dir=base_dir)
    
    # Record the end time
    end_now = nf.time_now()
    print(f"Process finished at: {end_now}")
    
    print("\n-------------- FINISHED PROCESSING ENTROPY FILES ----------------\n")


if __name__ == "__main__":
    
    print("\n-------------- READING COMMAND-LINE ARGUMENTS ----------------\n")
    
    # Record the start time
    start_time = time.time()
    
    # Check if the correct number of command-line arguments are provided
    if len(sys.argv) != 4:
        print("Usage: python entropy_files.py <n_clusters> <region> <scaler>")
        sys.exit(1)
    
    # Parse command-line arguments
    nc = int(sys.argv[1])
    region = sys.argv[2]
    scaler = sys.argv[3]
    n_ens = 20 # Number of ensembles. Defaults to 20.
    
    # Call and run the entropy file processing
    run_entropy_file(nc=nc, n_ens=n_ens, region=region, scaler=scaler)
    
    # Record the end time
    end_time = time.time()
    
    # Calculate and print the execution time
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds\n")

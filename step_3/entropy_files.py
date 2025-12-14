"""
This mudule exports a single dataframe with entropy for all base label IDs.
It dentifies the base label ID with the least entropy. Finally, it exports entropy
files with respect to that base_label
"""

import io
import os
import sys
import re
import glob
import copy
import time
import numpy as np
import pandas as pd
from collections import OrderedDict

# Ensure UTF-8 encoding for stdout
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8") 

# Add path to local modules and import them
source = os.path.abspath('/home/Laique.Djeutchouang/DEVs/BV-Regimes/NEMI/seaLevelRegimes')
if source not in sys.path:
    sys.path.insert(1, source)

from src import nemi_func as nf # Importing the nemi_func module
from src import aux_func as af # Importing the aux_func module


# Define base directory for static files
base_dir = '/work/lnd/CM4X/NEMI'


def load_cluster_based_entropy(entropy_dir:str, n_clusters:int) -> OrderedDict:
    """
    Load ncluster-based entropy data for all base label IDs.

    Args:
        entropy_dir (str): Directory from which the entropy files are loaded.
        n_clusters (int): Number of clusters.

    Returns:
        OrderedDict: A dictionary of minimum entropy data sorted by keys.
    """
    filenames = glob.glob(f'{entropy_dir}/nclusters_{n_clusters}/base_labels/entropy_base_label_id_*.csv')
    
    min_ent = {}
    for filename in filenames:
        num = af.get_numbers_from_filename(filename.split('/')[-1])
        md = float(num[2] + '.' + num[3])
        nn = int(num[1])
        bid = int(num[0])
        min_ent[(bid, nn, md)] = pd.read_csv(filename)
        
    # Sort the dictionary by keys
    sorted_min_entropy_dict = OrderedDict(sorted(min_ent.items(), key=lambda t: t[0]))

    return sorted_min_entropy_dict



def concatenate_entropy_data(n_clusters:int, emb_params:list[tuple], entropy_dir:str):
    """
    Concatenate loaded ncluster-based entropy data for all UMAP key 
    parameter combinations (emb_params) and save to CSV.

    Args:
        n_clusters (int): Number of clusters.
        emb_params (list[tuple]): List of UMAP key parameter combinations - minimum distances & nearest neighhbors.
        entropy_dir (str): Directory where the entropy files are stored.
    """
    # Load ncluster-based entropy data for all base label IDs.
    af.log_info(f"Loading entropy CSV data associated with #clusters = {n_clusters} ...")
    base_label_dict = load_cluster_based_entropy(entropy_dir, n_clusters)
    
    # Define output directory
    output_dir = f'{entropy_dir}/nclusters_{n_clusters}/mean_entropy'
    print(f"\nCreating <{output_dir}> if it doesn't exist ...")
    af.make_dirs(output_dir)
    
    af.log_info(f"Concatenating loaded entropy CSV data ...")
    for nn, md in emb_params:
        
        # Filter the base label dictionary for the current knn and md
        filtered_values = [
            value for (a, b, c), value in base_label_dict.items()
            if b == nn and c == md]

        # Concatenate the filtered values into a single DataFrame and save to CSV
        df = pd.concat(filtered_values)
        output_filename = f'{output_dir}/mean_entropy_nn_{nn}_md_{md}_nc_{n_clusters}.csv'
        df.to_csv(output_filename, index=False)
    


def export_min_entropy_labels(n_clusters:int, entropy_dir:str):
    """
    Export base labels with minimum entropy to a text file.

    Args:
        n_clusters (int): Number of clusters.
        entropy_dir (str): Base directory where the entropy files are stored.
    """
    # Define data directory and the filenames to process
    data_dir = f'{entropy_dir}/nclusters_{n_clusters}/mean_entropy'
    filenames = glob.glob(f'{data_dir}/mean_entropy_*_nc_{n_clusters}.csv')
    
    # Extract base labels with minimum entropy and save to a text file
    # Initialize an empty dictionary to store the minimum entropy data and
    # the corresponding base labels for each parameter combination (nn, md)
    af.log_info(f"Exporting base labels with minimum entropy associated with #clusters = {n_clusters} ...")
    min_ent = {}
    for filename in filenames:
        num = af.get_numbers_from_filename(filename.split('/')[-1])
        md = float(num[1] + '.' + num[2])
        nn = int(num[0])

        df = pd.read_csv(filename, on_bad_lines='skip')
        min_ent[(nn, md)] = df.mean_entropy.idxmin()

    ordered_base_label_dict = OrderedDict(sorted(min_ent.items(), key=lambda t: t[0]))
    lst = list(ordered_base_label_dict.values())
    
    # Define output directory
    output_dir = f'{entropy_dir}/nclusters_{n_clusters}/min_entropy'
    af.log_info(f"\nCreating <{output_dir}> if it doesn't exist ...")
    af.make_dirs(output_dir)
    
    # Save the list of base labels with minimum entropy to a text file
    output_filename = f'{output_dir}/base_label_min_entropy_nc_{n_clusters}.txt'
    with open(output_filename, 'w') as file:
        file.write('\n'.join(str(itm) for itm in lst))



def process_clusters_entropy(clust_dir:str, n_clusters:int, lst:list,
                             emb_params:list[tuple], entropy_dir:str, num_members:int=20):
    """
    Process clusters and calculate entropy for each parameter combination.

    Args:
        clust_dir (str): Directory where the cluster files are stored.
        n_clusters (int): Number of clusters.
        lst (list): List of base labels with minimum entropy.
        emb_params (list[tuple]): List of UMAP key parameter combinations - minimum distances & nearest neighhbors.
        entropy_dir (str): Directory to save the entropy output files.
        num_members  (int, optional): Number of ensemble members. Defaults to 20.
    """
    # Load clusters data
    af.log_info(f"Loading embedded clusters associated with: #cluters={n_clusters} ...")
    sorted_nclusters_dict, ncluster_size = af.load_clusters(clust_dir=clust_dir, n_clusters=n_clusters)

    # Initialize parameters
    n_params = len(emb_params)
    n_pts = copy.deepcopy(ncluster_size)
    
    # Initialize labels array
    af.log_info("Initializing and filling cluster labels array ...")
    lab = np.zeros((n_params, num_members, n_pts)) * np.nan
    for k1, k2 in enumerate(range(1, num_members + 1)):
        for idx, (nn, md) in enumerate(emb_params):
            lab[idx, k1, :] = sorted_nclusters_dict[(k2, nn, md)]

    # Calculate entropy
    output_dir = f'{entropy_dir}/nclusters_{n_clusters}'
    for id in range(n_params):
        nn, md = emb_params[id]
        filename = f"entropy_emb_nn_{nn}_md_{md}_nc_{n_clusters}.npy"
        output_filename = f'{output_dir}/{filename}'
        if not os.path.exists(output_filename):
            af.log_info(f"Calculating entropy for: n_neighbors={nn}, min_dist={md}, #clusters={n_clusters}  ...")
            labels = lab[id, :, :]
            
            print(f"{' '*5}• Getting overlap of associated cluster labels: base label ID = {lst[id]} ...")
            sorted_overlap = af.get_overlap(ens_labels=labels, id=lst[id], num_members=num_members)
            sorted_overlap_ = np.nan_to_num(sorted_overlap)
            
            print(f"{' '*5}• Getting entropy for the sorted overlap ...")
            df = pd.DataFrame(np.argmax(sorted_overlap_, axis=1)).T
            df_c = af.get_ent(df)
            entropy = df_c.entropy
            
            print(f"{' '*5}• Saving the computed entropy ...")
            np.save(output_filename, entropy)
        else:
            af.log_info(f" File <{filename}> already exists. Skipped entropy calculation.")
            continue
        
    af.log_info(f"Completed processing embedded clusters associated with #clusters = {n_clusters}.") 



def run_entropy_file(data_res:str, data_field:str, n_clusters:int, num_members:int=20):
    """
    Main function to execute the entropy calculation for a given cluster.
    
    Args:
        data_res (str): Resolution of the data.
        data_field (str): 'statics' (mean) or 'dynamics' (e.g.; monthly climatology).
        n_clusters (int): Number of clusters.
        num_members (int, optional): Number of ensembles. Defaults to 20.
    """ 
    # List of UMAP key parameter combinations - minimum distances & nearest neighhbors.
    emb_params = af.UMAP_NNS_MDS
    
    print("\n-------------- STARTING PROCESSING ENTROPY FILES ----------------\n")

    # Record the start time
    start_now = nf.time_now()
    print(f"Process started at: {start_now}")
    
    print(f"{'='*7} Key Parameters Used {'='*7}")
    print(f"{' '*5}• Number of ensemble members = {num_members}")
    print(f"{' '*5}• Cluster complexity level = {n_clusters}")
    
    # Declare embedding and clustering directories
    if data_field == 'statics':
        clust_dir = f'{base_dir}/CM4X-{data_res}/outputs/statics/clusterings'
        entropy_dir = f'{base_dir}/CM4X-{data_res}/outputs/statics/entropy'
    elif data_field == 'dynamics':
        clust_dir = f'{base_dir}/CM4X-{data_res}/outputs/dynamics/clusterings'
        entropy_dir = f'{base_dir}/CM4X-{data_res}/outputs/dynamics/entropy'
    else:
        raise ValueError(f"Unknown field: {data_field}. Must be 'statics (mean)' or 'dynamics' (e.g.; monthly climatology).")

    # Load and concatenate ncluster-based entropy data for all base label IDs.
    concatenate_entropy_data(n_clusters=n_clusters, emb_params=emb_params, entropy_dir=entropy_dir)
    
    # Export base labels with minimum entropy to a text file
    export_min_entropy_labels(n_clusters=n_clusters, entropy_dir=entropy_dir)

    # Load clusters and process entropy for each parameter combination
    filename = f"base_label_min_entropy_nc_{n_clusters}.txt"
    input_filename = f"{entropy_dir}/nclusters_{n_clusters}/min_entropy/{filename}"
    with open(input_filename, 'r') as file:
        lst = [int(line.strip()) for line in file.readlines()]
    process_clusters_entropy(clust_dir=clust_dir, n_clusters=n_clusters, lst=lst,
                             emb_params=emb_params, entropy_dir=entropy_dir, num_members=num_members)
    
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
        print("Usage: python entropy_files.py <data_res> <data_field> <n_clusters>")
        sys.exit(1)
    
    # Parse command-line arguments
    data_res = sys.argv[1]
    data_field = sys.argv[2]
    n_clusters = int(sys.argv[3])
    num_members = 20 # Number of ensemble members. Defaults to 20.
    
    # Call and run the entropy file processing
    run_entropy_file(data_res=data_res, data_field=data_field,
                     n_clusters=n_clusters, num_members=num_members)
    
    # Record the end time
    end_time = time.time()
    
    # Calculate and print the execution time
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds\n")

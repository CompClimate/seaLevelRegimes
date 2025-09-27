"""
This file calculates entropy for one base label
Exports to dataframe
This script is run in parallel for all base label IDs using a bash script
"""

import os
import sys
import time
import copy

# Add path to local modules and import them
source = os.path.abspath('/home/djeutsch/Projects/seaLevelRegimes')
if source not in sys.path:
    sys.path.insert(1, source)

from src import nemi_func as nf # Importing the nemi_func module
from src import aux_func as af # Importing the aux_func module




def save_entropy_to_csv(lab, emb_params, nc, blid, n_ens, entropy_dir):
    """
    Calculate entropy and save it to CSV files.

    Args:
        lab (np.ndarray): Labels array.
        emb_params (list): List of combinations of used parameters - minimum distances & UMAP nearest neighhbors.
        nc (int): Number of clusters.
        blid (int): Base ID.
        n_ens (int): Number of ensembles.
        entropy_dir (str): Directory to save the entropy output files.
    """
    # Create output directory if it doesn't exist
    output_dir = f'{entropy_dir}/nclusters_{nc}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nEntropy output directory created at: {output_dir}")
    
    for id, nkey in enumerate(emb_params):
        outfile = f'{output_dir}/base_label_id_{blid}_n_neighbors_{nkey[0]}_min_dist_{nkey[1]}_entropy.csv'

        if not os.path.exists(outfile):
            df_ent = af.get_ent_for_all_params(lab, param_id=id, baselab_id=blid, n_ens=n_ens)
            df_ent.to_csv(outfile, index=False)
            print(f"Entropy saved to: {outfile}")
        else:
            print(f"File {outfile} already exists. Skipping ...")



def run_entropy_blabel(nc:int, blid:int, region:str='Global', scaler:str='Standard', n_ens:int=20):
    """
    Main function to execute the entropy calculation for a given cluster and a specific base label ID
    
    Args:
        nc (int): Number of clusters.
        blid (int): Base ID.
        region (str, optional): Region of interest. Defaults to 'Global'.
        scaler (str, optional): Data scaler used. Defaults to 'Standard'.
        n_ens (int, optional): Number of ensembles. Defaults to 20.
    """ 
    # List of UMAP key parameter combinations - minimum distances & nearest neighhbors.
    emb_params = af.UMAP_KNN_MIN_DIST
    
    print("\n-------------- STARTING COMPUTING ENTROPY ----------------\n")

    # Record the start time
    start_now = nf.time_now()
    print(f"Process started at: {start_now}")
    
    print(f"Ensembles = {n_ens}")
    print(f"Base label ID = {blid}")
    print(f'NEMI Param: n_cluster = {nc}')
    print(f'NEMI Param: Ensemble as base_label_id = {blid}')
    print(f"Region of interest: {region}")
    print(f"Data scaler used: {scaler}") 

    # Define the base directory
    base_dir = f'/group/maikesgrp/laique/NOAA/nemis/{region}/{scaler}'
    
    # Define data base directory and load clusters data
    clusters_dir = f'{base_dir}/clusterings'
    sorted_nclusters_dict, ncluster_size = af.load_clusters(nc=nc, clusters_dir=clusters_dir)

    # Define constants
    n_pts = copy.deepcopy(ncluster_size)
    
    # Create and fill a labels array with cluster data
    lab = af.fill_labels_array(sorted_nclusters_dict=sorted_nclusters_dict,
                               emb_params=emb_params, n_ens=n_ens, n_pts=n_pts)
    
    # Define entropy directory and save entropy to CSV
    entropy_dir = f'{base_dir}/entropy'
    save_entropy_to_csv(lab=lab, emb_params=emb_params, nc=nc, blid=blid,
                        n_ens=n_ens, entropy_dir=entropy_dir)
    
    # Record the end time
    end_now = nf.time_now()
    print(f"Process finished at: {end_now}")

    print("\n-------------- FINISHED COMPUTING ENTROPY ----------------\n\n")




if __name__ == "__main__":
    
    print("\n-------------- READING COMMAND-LINE ARGUMENTS ----------------\n")
    
    # Record the start time
    start_time = time.time()
    
    # Parse command-line arguments
    if len(sys.argv) != 5:
        print("Usage: python entropy_baselabels.py <n_clusters> <base_id> <region> <scaler>")
        sys.exit(1)

    nc = int(sys.argv[1])
    blid = int(sys.argv[2])
    region = sys.argv[3]
    scaler = sys.argv[4]
    n_ens = 20 # Number of ensembles. Defaults to 20.
    
    # Call and run the entropy based label main function
    run_entropy_blabel(nc=nc, blid=blid, region=region, scaler=scaler, n_ens=n_ens)
    
    # Record the end time
    end_time = time.time()
    
    # Calculate and print the execution time
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds\n")
       

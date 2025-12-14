"""
This file calculates entropy for one base label
Exports to dataframe
This script is run in parallel for all base label IDs using a bash script
"""
import io
import os
import sys
import time
import copy

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


def save_entropy_to_csv(labels, emb_params, n_clusters, blid, num_members, entropy_dir):
    """
    Calculate entropy and save it to CSV files.

    Args:
        labels (np.ndarray): Ensemble labels array.
        emb_params (list): List of combinations of used parameters - minimum distances & UMAP nearest neighhbors.
        n_clusters (int): Number of clusters.
        blid (int): Base ID.
        num_members (int): Number of ensemble members.
        entropy_dir (str): Directory to save the entropy output files.
    """
    # Create output directory if it doesn't exist
    output_dir = f'{entropy_dir}/nclusters_{n_clusters}/base_labels'
    print(f"\nCreating <{output_dir}> if it doesn't exist ...")
    af.make_dirs(output_dir)
    
    for id, nkey in enumerate(emb_params):
        filename = "entropy_base_label_id_{blid}_nn_{nkey[0]}_md_{nkey[1]}.csv"
        output_filename = f'{output_dir}/{filename}'

        if not os.path.exists(output_filename):
            df_ent = af.get_ent_for_all_params(labels=labels, param_id=id, baselab_id=blid, num_members=num_members)
            df_ent.to_csv(output_filename, index=False)
            print(f"Entropy saved as: <{filename}>")
        else:
            print(f"File <{output_filename}> already exists. Skipping ...")



def run_entropy_blabel(data_res:str, data_field:str, blid:int, n_clusters:int, num_members:int=20):
    """
    Main function to execute the entropy calculation for a given cluster and a specific base label ID
    
    Args:
        data_res (str): Resolution of the data.
        data_field (str): 'statics' (mean) or 'dynamics' (e.g.; monthly climatology).
        blid (int): Base ID.
        n_clusters (int): Number of clusters.
        num_members (int, optional): Number of ensemble members. Defaults to 20.
    """ 
    # List of UMAP key parameter combinations - minimum distances & nearest neighhbors.
    emb_params = af.UMAP_NNS_MDS
    
    print("\n-------------- STARTING COMPUTING ENTROPY ----------------\n")

    # Record the start time
    start_now = nf.time_now()
    print(f"Process started at: {start_now}")
    
    print(f"{'='*7} Key Parameters Used {'='*7}")
    print(f"{' '*5}• Number of ensemble members = {num_members}")
    print(f"{' '*5}• Member base label ID = {blid}")
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
    
    # Load clusters data
    af.log_info(f"Loading embedded clusters associated with: #cluters={n_clusters} ...")
    sorted_nclusters_dict, ncluster_size = af.load_clusters(clust_dir=clust_dir, n_clusters=n_clusters)
    
    # Create and fill a labels array with cluster data
    af.log_info("Creating and filling cluster labels array ...")
    n_pts = copy.deepcopy(ncluster_size)
    labels = af.fill_labels_array(sorted_nclusters_dict=sorted_nclusters_dict,
                                  emb_params=emb_params, num_members=num_members, n_pts=n_pts)
    
    # Ensure entropy output directory exists
    af.log_info(f"Creating <{entropy_dir}> if it doesn't exist ...")
    af.make_dirs(entropy_dir)
    
    # Compute and save entropy to CSV
    af.log_info("Calculating and saving entropy to CSV files ...")
    save_entropy_to_csv(labels=labels, emb_params=emb_params, n_clusters=n_clusters, blid=blid,
                        num_members=num_members, entropy_dir=entropy_dir)
    
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
        print("Usage: python entropy_baselabels.py <data_res> <data_field> <base_id> <n_clusters>")
        sys.exit(1)

    data_res = sys.argv[1]
    data_field = sys.argv[2]
    blid = int(sys.argv[3])
    n_clusters = int(sys.argv[4])
    num_members = 20 # Number of ensemble members. Defaults to 20.
    
    # Call and run the entropy based label main function
    run_entropy_blabel(data_res=data_res, data_field=data_field,
                       blid=blid, n_clusters=n_clusters, num_members=num_members)
    
    # Record the end time
    end_time = time.time()
    
    # Calculate and print the execution time
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds\n")
       

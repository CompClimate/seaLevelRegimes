# Script to create and export sorted clusters

import os
import sys
import time
import numpy as np

# Add path to local modules and import them
source = os.path.abspath('/home/djeutsch/Projects/seaLevelRegimes')
if source not in sys.path:
    sys.path.insert(1, source)

from src import nemi_func as nf # Importing the nemi_func module
from src import aux_func as af # Importing the aux_func module


# Define base directory for static files
base_dir = '/group/maikesgrp/laique/NOAA/nemis'




def load_embeddings(embedding_filename):
    """Load embeddings from a .npy file."""
    
    if not os.path.exists(embedding_filename):
        raise FileNotFoundError(f"File <{embedding_filename}> not found!")
    
    return np.load(embedding_filename)



def run_clustering(member:int, resolution:str, field:str, umap_kwargs:dict, clust_kwargs:dict):
    """
    Main function to execute the clustering process. 
    
    Args:
        ens (int): Ensemble number.
        min_dist (float): Minimum distance for UMAP.
        umap_knn (int): UMAP nearest neighbors.
        n_clusters (int): Number of clusters.
        hclust_knn (int): HCLUST nearest neighbors.
        region (str): Region of interest.
        scaler (str): Data scaler used.
    """
    
    print("\n-------------- STARTING CLUSTERING ----------------\n")

    # Record the start time
    start_now = nf.time_now()
    print(f"Process started at: {start_now}")
    
    # Extract kwargs with defaults
    umap_md = umap_kwargs.get("umap_md", 0.5)
    umap_nn = umap_kwargs.get("umap_nn", 200)
    n_clusters = clust_kwargs.get("n_clusters", 3)
    hclust_n = clust_kwargs.get("hclust_n", 40)
    
    print(f"{'='*7} Key Parameters Used (with defaults for others) {'='*7}")
    print(f"{' '*5}• Ensemble member = {member}")
    print(f"{' '*5}• UMAP min_dist = {umap_md}")
    print(f"{' '*5}• UMAP n_neighbors = {umap_nn}")
    print(f"{' '*5}• AGGLOMEROTIVE n_clusters = {n_clusters}")
    print(f"{' '*5}• AGGLOMEROTIVE hclust_neighbors  = {hclust_n}") 

    # Declare embedding directory
    if field == 'mean':
        embed_dir = f'{base_dir}/CM4X-{resolution}/outputs/statics/embeddings'
        clust_dir = f'{base_dir}/CM4X-{resolution}/outputs/statics/clusterings/nclusters_{n_clusters}'
    else:
        embed_dir = f'{base_dir}/CM4X-{resolution}/outputs/dynamics/embeddings'
        clust_dir = f'{base_dir}/CM4X-{resolution}/outputs/dynamics/clusterings/nclusters_{n_clusters}'
    
    # Generate input data filename
    preffix = f"emb_{member:02d}th_ensemble_md_{umap_md}_nn_{umap_nn}"
    input_filename = f"{embed_dir}/{preffix}.npy"
    def embedding_exists():
        """
        Check if an embedding file exists under prescribed configuration.
        """
        if os.path.exists(input_filename):
            return True
        else:
            return False
    
    # Ensure the output directory exists
    af.log_info(f"Creating <{embed_dir}> if it doesn't exist ...")
    af.make_dirs(clust_dir)
    
    # Generate output data filename
    suffix = f"{preffix}_hclustn_{hclust_n}_nc_{n_clusters}"
    output_filename = f"{clust_dir}/{suffix}.npy"
    
    if embedding_exists():
        if not os.path.exists(output_filename):
            
            embedding = load_embeddings(input_filename) # Load embedding
            clusters = nf.get_sorted_clusters(df_umap=embedding, n_clusters=n_clusters, hclust_neighbors=hclust_n) # Perform clustering
            np.save(output_filename, clusters) # Save clusters
            
        else:
            af.log_info(f'Desired clusters already exist, and can be loaded from: <{output_filename}>')
            
    else:
        af.log_info(f"Embedding data to cluster not found! It hasn't been created yet.")

    # Record the end time
    end_now = nf.time_now()
    print(f"Process finished at: {end_now}")

    print("\n-------------- CLUSTERING COMPLETED ----------------\n\n")



if __name__ == "__main__":
    
    print("\n-------------- READING COMMAND-LINE ARGUMENTS ----------------\n")
    
    # Record the start time
    start_time = time.time()

    # Parse command-line arguments
    if len(sys.argv) != 6:  # 6 arguments + script name
        print("Usage: python hac_clusters.py <member> <umap_md> <umap_nn> <n_clusters> <hclust_n>")
        sys.exit(1)

    # Parse command-line arguments
    member = int(sys.argv[1])
    umap_kwargs = {"umap_md": float(sys.argv[2]),
                   "umap_nn": int(sys.argv[3])}
    clust_kwargs = {"n_clusters": float(sys.argv[4]),
                    "hclust_n": int(sys.argv[5])}
    resolution = 'p125' 
    field = 'mean'  # for 'statics' or 'dynamics', depending on your data
    
    # Call and run the clustering main function
    run_clustering(member=member, resolution=resolution,
                   umap_kwargs=umap_kwargs, clust_kwargs=clust_kwargs)
    
    # Record the end time
    end_time = time.time()
    
    # Calculate and print the execution time
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds\n")
    
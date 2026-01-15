# Script to create and export sorted clusters

import io
import os
import sys
import time
import numpy as np

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

def load_embeddings(embedding_filename):
    """Load embeddings from a .npy file."""
    
    if not os.path.exists(embedding_filename):
        raise FileNotFoundError(f"File <{embedding_filename}> not found!")
    
    return np.load(embedding_filename)



def run_clustering(input_file:str, clust_kwargs:dict, output_file:str):
    """
    Main function to execute the clustering process. 
    
    Args:
        input_file (str): Path to the input embedding file.
        clust_kwargs : Keyword arguments to passed to clustering method.
            Clustering methods available: 'Agglomerative'.
            Agglomerative options:
                - n_clusters (int) defaults to 3
                - hclust_n (int) defaults to 40
                - n_jobs (int): number of parallel jobs to run (defaults to -1)
        output_file (str): Path to save the output clustering file.
    Returns:
        None
    """
    
    print("\n-------------- STARTING CLUSTERING ----------------\n")

    # Record the start time
    start_now = nf.time_now()
    print(f"Process started at: {start_now}")
    
    # Extract kwargs with defaults
    n_clusters = clust_kwargs.get("n_clusters", 3)
    hclust_n = clust_kwargs.get("hclust_n", 40)
    n_jobs = clust_kwargs.get("n_jobs", -1)
    
    print(f"{'='*7} Key Parameters Used (with defaults for others) {'='*7}")
    print(f"{' '*5}• AGGLOMEROTIVE n_clusters = {n_clusters}")
    print(f"{' '*5}• AGGLOMEROTIVE hclust_neighbors  = {hclust_n}") 

    # Define the input checkpoint file
    def embedding_exists():
        """
        Check if an embedding file exists under prescribed configuration.
        """
        if os.path.exists(input_file):
            return True
        else:
            return False
    
    if embedding_exists():
        if not os.path.exists(output_file):
            
            embedding = load_embeddings(input_file) # Load embedding
            clusters = nf.get_sorted_clusters(df_umap=embedding, n_clusters=n_clusters, hclust_neighbors=hclust_n, n_jobs=n_jobs) # Perform clustering
            np.save(output_file, clusters) # Save clusters

        else:
            af.log_info('Desired clusters already exist')
            
    else:
        af.log_info("Embedding data to cluster not found! It hasn't been created yet.")

    # Record the end time
    end_now = nf.time_now()
    print(f"Process finished at: {end_now}")

    print("\n-------------- CLUSTERING COMPLETED ----------------\n\n")



if __name__ == "__main__":
    import argparse
    import os
    import time
    
    
    print("\n-------------- READING COMMAND-LINE ARGUMENTS ----------------\n")

    parser = argparse.ArgumentParser(description="UMAP embedding for BV budget (HPC-safe)")

    parser.add_argument("input_file", type=str)
    parser.add_argument("hclust_n", type=int)
    parser.add_argument("n_clusters", type=int)
    parser.add_argument("output_file", type=str)

    args = parser.parse_args()
    start = time.time()
    
    clust_kwargs = {"hclust_n": args.hclust_n,
                    "n_clusters": args.n_clusters,
                    "n_jobs": int(os.environ.get("SLURM_CPUS_PER_TASK", 1)), # Use allocated CPUs or number of threads
                    }

    
    # Record the start time
    start_time = time.time()

    # Call and run the clustering main function
    af.log_info('Started clustering of learned Manifold Representation.')
    run_clustering(input_file=args.input_file, 
                   clust_kwargs=clust_kwargs, 
                   output_file=args.output_file)
    af.log_info('Completed clustering of learned Manifold Representation.')
    
    # Record the end time
    end_time = time.time()
    
    # Calculate and print the execution time
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds\n")
    
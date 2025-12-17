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



def run_clustering(data_res:str, data_field:str, member:int, umap_kwargs:dict, clust_kwargs:dict):
    """
    Main function to execute the clustering process. 
    
    Args:
        data_res (str): Resolution of the data.
        data_field (str): 'statics' (mean) or 'dynamics' (e.g.; monthly climatology).
        member (int): Ensemble number.
        
        umap_kwargs: Keyword arguments to passed to UMAP.
            UMAP options:
                - umap_md (float): min_dist defaults to 0.5
                - umap_nn (int): n_neighbors defaults to 200
                - n_components (int): number of embedding dimensions (defaults to 3)
                - n_epochs (int): number of epochs (defaults to None)
                - learning_rate (float): learning rate (defaults to 1.0)

        clustering_kwargs : Keyword arguments to passed to clustering method.
            Clustering methods available: 'Agglomerative'.
            Agglomerative options:
                - n_clusters (int) defaults to 3
                - hclust_n (int) defaults to 40
                - n_jobs (int): number of parallel jobs to run (defaults to -1)
    Returns:
        None
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

    # Declare embedding and clustering directories
    if data_field == 'statics':
        embed_dir = f'{base_dir}/CM4X-{data_res}/outputs/statics/embeddings'
        clust_dir = f'{base_dir}/CM4X-{data_res}/outputs/statics/clusterings/nclusters_{n_clusters}'
    elif data_field == 'dynamics':
        embed_dir = f'{base_dir}/CM4X-{data_res}/outputs/dynamics/embeddings'
        clust_dir = f'{base_dir}/CM4X-{data_res}/outputs/dynamics/clusterings/nclusters_{n_clusters}'
    else:
        raise ValueError(f"Unknown field: {data_field}. Must be 'statics (mean)' or 'dynamics' (e.g.; monthly climatology).")
    
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
    af.log_info(f"Creating <{clust_dir}> if it doesn't exist ...")
    af.make_dirs(clust_dir)
    
    # Generate output data filename
    suffix = f"{preffix}_nc_{n_clusters}_hclustn_{hclust_n}"
    output_filename = f"{clust_dir}/{suffix}.npy"
    
    if embedding_exists():
        if not os.path.exists(output_filename):
            
            embedding = load_embeddings(input_filename) # Load embedding
            clusters = nf.get_sorted_clusters(df_umap=embedding, n_clusters=n_clusters, hclust_neighbors=hclust_n, n_jobs=n_jobs) # Perform clustering
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
    import sys
    import os
    import time
    
    print("\n-------------- READING COMMAND-LINE ARGUMENTS ----------------\n")
    
    # Record the start time
    start_time = time.time()

    # Parse command-line arguments
    if len(sys.argv) != 8:  # 8 arguments + script name
        print("Usage: python hac_clusters.py <data_res> <data_field> <member> <umap_md> <umap_nn> <hclust_n> <n_clusters>")
        sys.exit(1)

    # Parse command-line arguments
    data_res = sys.argv[1]
    data_field = sys.argv[2]
    member = int(sys.argv[3])
    umap_kwargs = {"umap_md": float(sys.argv[4]),
                   "umap_nn": int(sys.argv[5])}
    clust_kwargs = {"hclust_n": int(sys.argv[6]),
                    "n_clusters": int(sys.argv[7]),
                    "n_jobs": int(os.environ.get("SLURM_CPUS_PER_TASK", 1)), # Use allocated CPUs or number of threads
                    }

    # Call and run the clustering main function
    af.log_info('Started clustering of learned Manifold Representation.')
    run_clustering(data_res=data_res, data_field=data_field, member=member,
                   umap_kwargs=umap_kwargs, clust_kwargs=clust_kwargs)
    af.log_info('Completed clustering of learned Manifold Representation.')
    
    # Record the end time
    end_time = time.time()
    
    # Calculate and print the execution time
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds\n")
    
# Export embedding

import os
import sys
import time
import numpy as np
import pandas as pd
import copy

# Add path to local modules and import them
source = os.path.abspath('/home/djeutsch/Projects/seaLevelRegimes')
if source not in sys.path:
    sys.path.insert(1, source)

from src import nemi_func as nf # Importing the nemi_func module
from src import aux_func as af # Importing the aux_func module


# Define base directory for static files
base_dir = '/group/maikesgrp/laique/NOAA/nemis'

def generate_output_filename(output_dir, member, umap_md, umap_nn): 
    """Generate the output filename based on parameters."""
    output_filename = f"{output_dir}/{member:02d}th_ensemble_md_{umap_md}_nn_{umap_nn}"
    
    return output_filename


def process_embedding(data, member, umap_kwargs, embed_dir, scaler=None):
    """
    Process embeddings and save them to a file.
    """
    # Extract UMAP kwargs with defaults
    umap_md = umap_kwargs.get("umap_md", 0.5)
    umap_nn = umap_kwargs.get("umap_nn", 200)
    n_epochs = umap_kwargs.get("n_epochs", None)
    init = umap_kwargs.get("init", 'spectral')
    learning_rate = umap_kwargs.get("learning_rate", 1.0)
    
    print(f"{'='*7} Key Parameters Used (with defaults for others) {'='*7}")
    print(f"{' '*5}• Ensemble member = {member}")
    print(f"{' '*5}• UMAP min_dist = {umap_md}")
    print(f"{' '*5}• UMAP n_neighbors = {umap_nn}")
    print(f"{' '*5}• UMAP learning_rate = {learning_rate}")
    print(f"{' '*5}• UMAP n_epochs  = {n_epochs}")
    print(f"{' '*5}• UMAP init = {init}")
    print(f"{' '*5}• Data scaler used = {scaler}")
    
    # Generate output filename
    suffix = f"emb_{member:02d}th_ensemble_md_{umap_md}_nn_{umap_nn}.npy"
    output_filename = f"{embed_dir}/{suffix}"
    
    def embedding_exists():
        """
        Check if an embedding file exists under prescribed configuration.
        """
        if os.path.exists(output_filename):
            return True
        else:
            return False

    # Manifold representation learning
    if embedding_exists():
        af.log_info(f'Embedding <{suffix}> already exists.')
        return
    else:
        # --- Preprocessing ---
        if scaler is not None:
            valid_scalers = ['Quantile-Normal', 'Quantile-Uniform', 'Robust', 'Standard', 'Signed-Log', 'Power-10']
            if scaler not in valid_scalers:
                raise FileNotFoundError(
                    f"Unknown scaler: {scaler}. Scaler must be one of <{valid_scalers}>."
                )
            scaler_class = nf.scalers[scaler][0]
            scaler_args = {}
            if scaler == 'Quantile-Normal':
                scaler_args['output_distribution'] = 'normal'
            elif scaler == 'Quantile-Uniform':
                scaler_args['output_distribution'] = 'uniform'
            data_scaler = scaler_class(**scaler_args)
            af.log_info(f'Scaling data with {scaler} scaler ...')
            data = data_scaler.fit_transform(data)
        
        # Apply UMAP
        af.log_info(f'Computing embedding: {member:02d}th member of the ensemble ...')
        embedding = nf.apply_umap(dfn=data, min_dist=umap_md, umap_neighbors=umap_nn,
                                  learning_rate=learning_rate, n_epochs=n_epochs, init=init)
        
        af.log_info('Saving embedding ...')
        np.save(output_filename, embedding)
        
        return embedding


def run_embedding(parquet_file:str, member:int, umap_kwargs:dict, scaler:str):
    """
    Main function to execute the embedding process. This function reads
    the input data, applies UMAP for dimmemberionality reduction, and saves 
    the embeddings to a specified output directory. It also handles the data 
    scaling based on the provided scaler type.
    
    Args:
        parquet_file (str): Path to the input parquet file.
        resolution (str): Resolution of the CM4X data, e.g., 'p125' for CM4X-p125.
        member (int): memberemble number.
        scaler: sklearn-like transformer, optional
        A scaler (e.g., StandardScaler) to normalize the data.
        
        umap_kwargs: Keyword arguments to passed to UMAP.
            UMAP options:
                - umap_md (float): min_dist defaults to 0.5
                - umap_nn (int): n_neighbors defaults to 200
                - n_components (int): number of embedding dimensions (defaults to 3)
                - n_epochs (int): number of epochs (defaults to None)
                - learning_rate (float): learning rate (defaults to 1.0)
    """
    
    print("\n-------------- STARTING NEMI RUN ----------------\n")
    
    # Record and print the start time
    start = nf.time_now()
    print(f"Process started at {start}\n")
    
    # Read the input data
    af.log_info("Loading data ...")
    data = pd.read_parquet(parquet_file)
    data = data.values
    
    # Get resolution from the file path
    resolution = parquet_file.split('/')[-1].split('-')[-1].split('_')[0]

    # Define embedding output directory
    field = parquet_file.split('/')[-1].split('.')[0].split('_')[-1]
    if field == 'mean':
        embed_dir = f'{base_dir}/CM4X-{resolution}/outputs/statics/embeddings'
    else:
        embed_dir = f'{base_dir}/CM4X-{resolution}/outputs/dynamics/embeddings'
    
    # memberure the output directory exists
    af.log_info(f"Creating <{embed_dir}> if it doesn't exist ...")
    af.make_dirs(embed_dir)
    
    # Process embeddings
    embedding = process_embedding(data, member, umap_kwargs, embed_dir, scaler)

    # Record and print the completion time
    end = nf.time_now()
    print(f"\nProcess finished at {end}")
    
    return embedding
    


if __name__ == "__main__":
    
    print("\n-------------- READING COMMAND-LINE ARGUMENTS ----------------\n")
    
    # Record the start time
    start_time = time.time()
    
    # Check command-line arguments
    if len(sys.argv) != 5:
        print("Usage: python export_embedding.py <csv_file/parquet_file> <member> <min_dist> <n_neighbors> <scaler>")
        sys.exit(1)

    # Parse command-line arguments
    parquet_file = sys.argv[1]
    member = int(sys.argv[2])
    umap_kwargs = {"umap_md": float(sys.argv[3]),
                   "umap_nn": int(sys.argv[4])}
    scaler = None
    
    # Call and run the embedding main function
    _ = run_embedding(parquet_file=parquet_file, member=member, umap_kwargs=umap_kwargs, scaler=scaler)
    
    # Record the end time
    end_time = time.time()
    
    # Calculate and print the execution time
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds\n")
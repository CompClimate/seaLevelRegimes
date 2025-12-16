# Export embedding

import os
import io
import sys
import time
import numpy as np
import pandas as pd

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
    init = umap_kwargs.get("init", 'random')
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


def run_embedding(parquet_file:str, data_res:str, data_field:str, member:int, umap_kwargs:dict, scaler:str):
    """
    Main function to execute the embedding process. This function reads
    the input data, applies UMAP for dimmemberionality reduction, and saves 
    the embeddings to a specified output directory. It also handles the data 
    scaling based on the provided scaler type.
    
    Args:
        parquet_file (str): Path to the input parquet file.
        data_res (str): Resolution of the CM4X data, e.g., 'p125' for CM4X-p125.
        data_field (str): 'statics' (mean) or 'dynamics' (e.g.; monthly climatology).
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

    # Define embedding output directory
    if data_field == 'statics':
        embed_dir = f'{base_dir}/CM4X-{data_res}/outputs/statics/embeddings'
    elif data_field == 'dynamics':
        embed_dir = f'{base_dir}/CM4X-{data_res}/outputs/dynamics/embeddings'
    else:
        raise ValueError(f"Unknown field: {data_field}. Must be 'statics (mean)' or 'dynamics' (e.g.; monthly climatology).")
    
    # memberure the output directory exists
    af.log_info(f"Creating <{embed_dir}> if it doesn't exist ...")
    af.make_dirs(embed_dir)
    
    # Process embeddings
    af.log_info('Started the Manifold Representation Learning.')
    embedding = process_embedding(data, member, umap_kwargs, embed_dir, scaler)
    af.log_info('Completed the Manifold Representation Learning.')

    # Record and print the completion time
    end = nf.time_now()
    print(f"\nProcess finished at {end}")
    
    return embedding
    


if __name__ == "__main__":
    
    print("\n-------------- READING COMMAND-LINE ARGUMENTS ----------------\n")
    
    # Record the start time
    start_time = time.time()
    
    # Check command-line arguments
    if len(sys.argv) != 6:
        print("Usage: python export_embedding.py <data_res> <data_field> <member> <min_dist> <n_neighbors>")
        sys.exit(1)

    # Parse command-line arguments
    data_res = sys.argv[1]
    data_field = sys.argv[2]
    member = int(sys.argv[3])
    umap_kwargs = {"umap_md": float(sys.argv[4]),
                   "umap_nn": int(sys.argv[5]),
                   "init": 'random'}
    scaler = None
    parquet_file = f"{base_dir}/CM4X-{data_res}/inputs/global_BVB_mclim_fields_scaled.parquet" # Path to your data file
    
    # Call and run the embedding main function
    _ = run_embedding(parquet_file=parquet_file, data_res=data_res, data_field=data_field,
                      member=member, umap_kwargs=umap_kwargs, scaler=scaler)
    
    # Record the end time
    end_time = time.time()
    
    # Calculate and print the execution time
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds\n")
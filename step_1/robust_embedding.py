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

def process_embedding(data:np.ndarray, member:int, umap_kwargs:dict, output_file:str) -> np.ndarray:
    """
    Process embeddings and save them to a file.
    
    Args:
        data: Input data for embedding.
        member (int): memberure of the ensemble.
        umap_kwargs (dict): Keyword arguments for UMAP.
        output_file (str): Path to save the output embedding.
    Returns:
        embedding: The computed embedding. 
    """
    # Extract UMAP kwargs with defaults
    umap_md = umap_kwargs.get("umap_md", 0.5)
    umap_nn = umap_kwargs.get("umap_nn", 200)
    n_epochs = umap_kwargs.get("n_epochs", None)
    init = umap_kwargs.get("init", 'random')
    n_jobs = umap_kwargs.get("n_jobs", -1)
    learning_rate = umap_kwargs.get("learning_rate", 1.0)
    
    print(f"{'='*7} Key Parameters Used (with defaults for others) {'='*7}")
    print(f"{' '*5}• Ensemble member = {member}")
    print(f"{' '*5}• UMAP min_dist = {umap_md}")
    print(f"{' '*5}• UMAP n_neighbors = {umap_nn}")
    print(f"{' '*5}• UMAP n_jobs = {n_jobs}")
    print(f"{' '*5}• UMAP learning_rate = {learning_rate}")
    print(f"{' '*5}• UMAP n_epochs  = {n_epochs}")
    print(f"{' '*5}• UMAP init = {init}")
    
    # Manifold representation learning: apply UMAP
    if not os.path.exists(output_file):
        af.log_info(f'Computing embedding for - ENS: {member:02d}, UMAP min_dist: {umap_md}, UMAP n_neighbors: {umap_nn}')
        embedding = nf.apply_umap(dfn=data, 
                                  min_dist=umap_md, 
                                  umap_neighbors=umap_nn,
                                  learning_rate=learning_rate, 
                                  n_epochs=n_epochs, 
                                  init=init,
                                  n_jobs=n_jobs)
        
        af.log_info('Saving embedding ...')
        np.save(output_file, embedding)
    else:
        af.log_info(f'FYI: embedding for - ENS={member:02d}, UMAP min_dist={umap_md}, UMAP n_neighbors={umap_nn} already exists.')
        print(f'It can be loaded at:\n<{output_file}>\n')
        return


def run_embedding(input_file:str, member:int, umap_kwargs:dict, output_file:str):
    """
    Main function to execute the embedding process. This function reads
    the input data, applies UMAP for dimensionality reduction, and saves 
    the embeddings to a specified output directory. It also handles the data 
    scaling based on the provided scaler type.
    
    Args:
        input_file (str): Path to the input data file.
        member (int): memberemble number.
        umap_kwargs: Keyword arguments to passed to UMAP.
            UMAP options:
                - umap_md (float): min_dist defaults to 0.5
                - umap_nn (int): n_neighbors defaults to 200
                - n_components (int): number of embedding dimensions (defaults to 3)
                - n_epochs (int): number of epochs (defaults to None)
                - learning_rate (float): learning rate (defaults to 1.0)
        output_file (str): Path to save the output embedding.
    """
    
    print("\n-------------- STARTING NEMI RUN ----------------\n")
    
    # Record and print the start time
    start = nf.time_now()
    print(f"Process started at {start}\n")
    
    # Read the input data
    af.log_info("Loading data ...")
    data = pd.read_parquet(input_file)
    data = data.values
    
    # Process embeddings
    af.log_info('Started the Manifold Representation Learning.')
    embedding = process_embedding(data, member, umap_kwargs, output_file)
    af.log_info('Completed the Manifold Representation Learning.')

    # Record and print the completion time
    end = nf.time_now()
    print(f"\nProcess finished at {end}")
    
    return embedding
    
    
# -----------------------------------------------------------------------------
# CLI entry point (job-array aligned)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import os
    import time

    parser = argparse.ArgumentParser(description="UMAP embedding for BV budget (HPC-safe)")

    parser.add_argument("input_file", type=str)
    parser.add_argument("min_dist", type=float)
    parser.add_argument("n_neighbors", type=int)
    parser.add_argument("output_file", type=str)
    parser.add_argument("member", type=int)

    args = parser.parse_args()
    start = time.time()
    
    umap_kwargs = {"umap_md": args.min_dist,
                   "umap_nn": args.n_neighbors,
                   "init": "random",  # Critical for HPC robustness
                   "n_jobs": int(os.environ.get("SLURM_CPUS_PER_TASK", 1)), # Use allocated CPUs or number of threads
                   }
    
    # Call and run the embedding main function
    _ = run_embedding(input_file=args.input_file, 
                      member=args.member,
                      umap_kwargs=umap_kwargs,
                      output_file=args.output_file)

    elapsed = time.time() - start
    print(f"\n✔ Total execution time: {elapsed:.2f} s\n")

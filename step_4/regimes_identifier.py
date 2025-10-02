# Export NEMI clusters

import os
import sys
import time
import copy
import numpy as np
import xarray as xr

# Add path to local modules and import them
source = os.path.abspath('/home/djeutsch/Projects/seaLevelRegimes')
if source not in sys.path:
    sys.path.insert(1, source)

from src import nemi_func as nf # Importing the nemi_func module
from src import aux_func as af # Importing the aux_func module





# Define base directory for static files
base_dir = '/group/maikesgrp/laique/NOAA/nemis'
bvb_dir = '/group/maikesgrp/laique/NOAA/CM4X-BV-budgets'

def save_regimes_as_DataArray(regimes:list[np.ndarray], n_clusters:int,
                              bvb_ds:xr.Dataset, output_dir: str):
    """
    Save NEMI regimes to NetCDF files.

    Args:
        regimes (list): List of identified clusters arrays.
        n_clusters (int): Number of clusters.
        bvb_ds (xr.Dataset): Original BVD Dataset to reconstruct the clusters.
        output_dir (str): Directory to save the NetCDF output files.
    """
    
    # Get the UMAP key parameter combinations
    emb_params = af.UMAP_NNS_MDS
    print(f"NEMI embedded clustering (#clusters = {n_clusters}) with:")
    for id, clusters in enumerate(regimes):
        # Get the UMAP parameters
        umap_nn, umap_md = emb_params[id]
        
        # Define the output filename
        filename = f"identified_regimes_md_{umap_md}_nn_{umap_nn}_nc_{n_clusters}.nc"
        output_filename = f'{output_dir}/{filename}'
        
        # Create a DataArray from the clusters
        if not os.path.exists(output_filename):
            
            clusters_ds = af.reconstruct_DataArray(bvb_ds=bvb_ds, embedded_nclusters_array=clusters)
            clusters_ds = clusters_ds.rename({'geo_cluster': 'regimes'})
            
            # Save the DataArray to a NetCDF file
            clusters_ds.to_netcdf(output_filename)
            af.log_info(f"Regimes DataArray created and saved as: <{filename}>")
        else:
            af.log_info(f"File <{output_filename}> already exists. Skipping.")
            continue
        
    af.log_info(f"Regimes ({n_clusters}) DataArray created and saved at: {output_dir}")



def run_regimes_identifier(n_clusters:int, resolution:str, field:str, num_members:int=20):
    """
    Main function to process, identify, and export geographical regimes.
    
    Args:
        n_clusters (int): Number of clusters.
        resolution (str): Resolution of the data.
        field (str): 'mean' for statics and anything else for dynamics.
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

     # Declare entropy and clustering directories
    if field == 'mean':
        clust_dir = f'{base_dir}/CM4X-{resolution}/outputs/statics/clusterings'
        entropy_dir = f'{base_dir}/CM4X-{resolution}/outputs/statics/entropy'
        regimes_dir = f'{base_dir}/CM4X-{resolution}/outputs/statics/regimes'
    else:
        clust_dir = f'{base_dir}/CM4X-{resolution}/outputs/dynamics/clusterings'
        entropy_dir = f'{base_dir}/CM4X-{resolution}/outputs/dynamics/entropy'
        regimes_dir = f'{base_dir}/CM4X-{resolution}/outputs/dynamics/regimes'
    
    # Create output directory if it doesn't exist
    output_dir = f'{regimes_dir}/nclusters_{n_clusters}'
    af.log_info(f"\nCreating <{output_dir}> if it doesn't exist ...")
    af.make_dirs(output_dir)

    # Load base label ids
    input_dir = f'{entropy_dir}/nclusters_{n_clusters}/min_entropy'
    label_ids = np.loadtxt(f'{input_dir}/base_label_min_entropy_nc_{n_clusters}.txt').astype('int')
    
    # Load clusters data
    sorted_nclusters_dict, ncluster_size = af.load_clusters(clust_dir=clust_dir, n_clusters=n_clusters)

    # Create an empty labels array and fill it with sorted clusters
    n_pts = copy.deepcopy(ncluster_size)
    n_param = len(emb_params)
    labels = af.fill_labels_array(sorted_nclusters_dict=sorted_nclusters_dict,
                                  emb_params=emb_params, num_members=num_members, n_pts=n_pts)

    # Get NEMI clusters
    regimes = [nf.assess_overlap(ens_labels=labels[id, :, :], label_id=label_ids[id], num_members=num_members) for id in range(n_param)]

    # Read the original Dataset file (netCDF)
    bvb_path = f'{bvb_dir}/CM4X-{resolution}_BVB_2005_2014_time_mean.nc'
    bvb_ds = xr.open_dataset(bvb_path)

    # Reconstruct NEMI clusters as a xarray DataArray and save
    save_regimes_as_DataArray(regimes=regimes, n_clusters=n_clusters,
                              bvb_ds=bvb_ds, output_dir=output_dir)
    
    # Record the end time 
    end_now = nf.time_now()
    print(f"Process finished at: {end_now}")
    
    print('\n-------------- FINISHED PROCESSING GEOGRAPHYCAL REGIMES ----------------\n')
    


if __name__ == "__main__":
    
    print("\n-------------- READING COMMAND-LINE ARGUMENTS ----------------\n")
    
    # Record the start time
    start_time = time.time()
    
    # Check if the correct number of command-line arguments are provided
    if len(sys.argv) != 2:
        print("Usage: python regimes_identifier.py <n_clusters>")
        sys.exit(1)
    
    # Parse command-line arguments
    n_clusters = int(sys.argv[1])
    num_members = 20 # Number of ensemble members. Defaults to 20.
    resolution = 'p125' 
    field = 'mean'  # for 'statics' or 'dynamics', depending on your data
    
    # Call and run the regimes file identifier main function
    run_regimes_identifier(n_clusters=n_clusters, resolution=resolution,
                           field=field, num_members=num_members)
    
    # Record the end time
    end_time = time.time()
    
    # Calculate and print the execution time
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds\n")







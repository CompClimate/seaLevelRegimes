# Export NEMI clusters

import os
import sys
import time
import copy
import numpy as np
import xarray as xr
from collections import OrderedDict

# Add path to local modules and import them
source = os.path.abspath('/home/djeutsch/Projects/seaLevelRegimes')
if source not in sys.path:
    sys.path.insert(1, source)

from src import nemi_func as nf # Importing the nemi_func module
from src import aux_func as af # Importing the aux_func module





def save_nemi_nclusters_DataArray(nemi_nclusters:list[np.ndarray], nc:int,
                                  original_ds:xr.Dataset, output_dir: str):
    """
    Save NEMI clusters to NetCDF files.

    Args:
        nemi_nclusters (list): List of NEMI cluster arrays.
        nc (int): Number of clusters.
        original_ds (xr.Dataset): Original xarray Dataset to reconstruct the clusters.
        output_dir (str): Directory to save the NetCDF output files.
    """
    
    # Get the UMAP key parameter combinations
    emb_params = af.UMAP_KNN_MIN_DIST
    print(f"NEMI embedded clustering (#clusters = {nc}) with:")
    for id, nclusters in enumerate(nemi_nclusters):
        # Get the UMAP parameters
        umap_knn, mini_dist = emb_params[id]
        
        # Reshape the NEMI clusters to match the original dataset dimensions
        nclusters_ds = af.reconstruct_DataArray(original_ds=original_ds, embedded_nclusters_array=nclusters)
        nclusters_ds = nclusters_ds.rename({'geo_cluster': 'nemi_nclusters'})
        
        # Save the DataArray to a NetCDF file
        output_path = f'{output_dir}/nemi_nclusters_{nc}_mini_dist_{mini_dist}_n_neighbors_{umap_knn}.nc'
        if not os.path.exists(output_path):
            nclusters_ds.to_netcdf(output_path)
        else:
            print(f"File {output_path} already exists. Skipping save.")
            continue
        
        print(f"{' '*3}{id}: n_neighbors = {emb_params[id][0]}, min_dist = {emb_params[id][1]}")
        
    print(f"NEMI nclusters ({nc}) saved at: {output_dir}\n")



def run_nemi_clusters(nc:int, n_ens:int=20, region:str='Global', scaler:str='Standard'):
    """
    Main function to process and export NEMI clusters.
    
    Args:
        nc (int): Number of clusters.
        n_ens (int, optional): Number of ensembles. Defaults to 20.
        region (str, optional): Region of interest. Defaults to 'Global'.
        scaler (str, optional): Data scaler used. Defaults to 'Standard'.
    """
    
    # List of UMAP key parameter combinations - minimum distances & nearest neighhbors.
    emb_params = af.UMAP_KNN_MIN_DIST
    
    print("\n-------------- STARTING PROCESSING NEMI CLUSTERS ----------------\n")

    # Record the start time
    start_now = nf.time_now()
    print(f"Process started at: {start_now}")
    
    print(f"Ensembles = {n_ens}")
    print(f'NEMI Param: n_cluster = {nc}')
    print(f"Region of interest: {region}")
    print(f"Data scaler used: {scaler}")  

    # Define directories of interest
    base_dir = f'/group/maikesgrp/laique/NOAA'
    data_dir = f'{base_dir}/nemis/{region}/{scaler}/entropy/nclusters_{nc}'

    # Load base labels
    list_nc = np.loadtxt(f'{data_dir}/base_label_min_entropy_nclusters_{nc}.txt').astype('int')
    
    # Define clusters data directory and load clusters data
    clusters_dir = f'{base_dir}/nemis/{region}/{scaler}/clusterings'
    sorted_nclusters_dict, ncluster_size = af.load_clusters(nc=nc, clusters_dir=clusters_dir)

    # Create an empty labels array and fill it with sorted clusters
    n_pts = copy.deepcopy(ncluster_size)
    n_param = len(emb_params)
    labels_nc = af.fill_labels_array(sorted_nclusters_dict=sorted_nclusters_dict,
                                     emb_params=emb_params, n_ens=n_ens, n_pts=n_pts)

    # Get NEMI clusters
    nemi_nclusters = [nf.assess_overlap(ensembles=labels_nc[id, :, :], lid=list_nc[id], n_ens=n_ens) for id in range(n_param)]

    # Read the original xarray Dataset file
    ds_path = f'{base_dir}/outputs/OM4p25_CM4_Barotropic_Vorticity_Budget_2005_2014_Mean.nc'
    original_ds = xr.open_dataset(ds_path)
    
    # Create output directory if it doesn't exist
    output_dir = f'{base_dir}/nemis/Global/Standard/nemi_nclusters/nclusters_{nc}'
    os.makedirs(output_dir, exist_ok=True)

    # Reconstruct NEMI clusters as a xarray DataArray and save
    save_nemi_nclusters_DataArray(nemi_nclusters=nemi_nclusters, nc=nc,
                                  original_ds=original_ds, output_dir=output_dir)
    
    # Record the end time 
    end_now = nf.time_now()
    print(f"Process finished at: {end_now}")
    
    print('\n-------------- FINISHED PROCESSING NEMI CLUSTERS ----------------\n')
    


if __name__ == "__main__":
    
    print("\n-------------- READING COMMAND-LINE ARGUMENTS ----------------\n")
    
    # Record the start time
    start_time = time.time()
    
    # Check if the correct number of command-line arguments are provided
    if len(sys.argv) != 4:
        print("Usage: python nemi_clusters.py <n_clusters> <n_ens> <region> <scaler>")
        sys.exit(1)
    
    # Parse command-line arguments
    nc = int(sys.argv[1])
    region = sys.argv[2]
    scaler = sys.argv[3]
    n_ens = 20 # Number of ensembles. Defaults to 20.
    
    # Call and run the entropy file processing
    run_nemi_clusters(nc=nc, region=region, scaler=scaler, n_ens=n_ens)
    
    # Record the end time
    end_time = time.time()
    
    # Calculate and print the execution time
    execution_time = end_time - start_time
    print(f"\nExecution time: {execution_time:.2f} seconds\n")







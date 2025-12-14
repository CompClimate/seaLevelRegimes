import io
import os
import sys
import re
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from collections import OrderedDict

# Ensure UTF-8 encoding for stdout
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8") 

# Add path to local modules and import them
source = os.path.abspath('/home/Laique.Djeutchouang/DEVs/BV-Regimes/NEMI/seaLevelRegimes')
if source not in sys.path:
    sys.path.insert(1, source)
from src import nemi_func as nf  # Importing the nemi_func module



def _entropy(row, i=0):

    data = row['counts']
    L = sum(data)
    n = len(data)
    
    if n != 1:
        ress = 0
        for i in range(n):
            ress =  ress - (data[i]/L * np.log2(data[i]/L))
                            
    else:
                            
        ress = - (data[i]/L * np.log2(data[i]/L))
    
    return ress


def get_ent(df):

    df_c = pd.DataFrame(df.stack().groupby(level=0).apply(lambda x: np.unique(x, return_inverse=True, return_counts=True)[2]))
    df_c.columns = ['counts']
    df_c['entropy'] = df_c.apply(_entropy, axis=1)

    return df_c


def get_overlap(ens_labels, id=0, num_members=3, max_clusters=None):

    base_id = id
    base_labels = ens_labels[base_id]
    compare_ids = [i for i in range(num_members)]
    compare_ids.pop(base_id)

    num_clusters = int(np.max(base_labels) + 1)

    # If not pre-set, set max number of clusters to total number of clusters in the base
    if max_clusters is None:
        max_clusters = num_clusters

    sortedOverlap = np.zeros((len(compare_ids)+1, max_clusters, base_labels.shape[0])) * np.nan

    # print(num_clusters, max_clusters)
    summaryStats = np.zeros((num_clusters, max_clusters))

    # Compile sorted cluster data
    # TODO: add assert statement to make sure that the clusters have been sorted?

    # dataVector = [nemi.clusters for id, nemi in enumerate(self.nemi_pack) if id != base_id]
    dataVector = [ens_labels[id] for id, nemi in enumerate(ens_labels) if id != base_id]

    # Loop over ensemble members, not including the base member
    for compare_cnt, compare_id in enumerate(compare_ids):
        # Grab clusters of ensemble member
        compare_labels = dataVector[compare_cnt]

        # go through each cluster in the base and assess the percentage overlap
        # for every cluster in the ensemble member (overlap / total coverage area) 
        for c1 in range(max_clusters): 
            # Initialize dummy array to mark location of the cluster for the base member
            data1_M = np.zeros(base_labels.shape, dtype=int)
            
            # Mark where the considered cluster is in the member that is being used as the baseline
            data1_M[np.where(c1==base_labels)] = 1 
            
            # Count number of entries [Why?] 
            summaryStats[0, c1] = np.sum(data1_M) 

            # Go through each cluster
            # k = 0
            for c2 in range(num_clusters):
                # Initialize dummy array to mark where the cluster is in the comparison member
                data2_M = np.zeros(base_labels.shape, dtype=int) 

                # Mark where the considered cluster is in the member that is being used as the comparison
                data2_M[np.where(c2==compare_labels)] = 1    

                # Sum of flags where the two datasets of that cluster are both present
                num_overlap = np.sum(data1_M * data2_M)       

                # Sum of where they overlap
                num_total = np.sum(data1_M | data2_M)       

                # Collect the number that is largest of k and the num_overlap/num_total
                # k = max(k, num_overlap / num_total)       
                summaryStats[c2, c1] = (num_overlap / num_total) * 100 # Add percentage of coverage

            # Filled in 'summaryStatistics' matrix results of percentage overlaps

        usedClusters = set() # Used to mak sure clusters don't get selected twice
        # Clusters are already sorted by size
        
        sortedOverlapForOneCluster = np.zeros(base_labels.shape, dtype=int) * np.nan
        
        # Go through clusters from (biggest to smallest since they are sorted)
        for c1 in range(max_clusters):  
            sortedOverlapForOneCluster = np.zeros(base_labels.shape, dtype=int) * np.nan
            # print('cluster number ', c1, summaryStats.shape, summaryStats[1:,c1-1].shape)

            # find biggest cluster in first column, making sure it has not been used
            sortedClusters = np.argsort(summaryStats[:, c1])[::-1]
            biggestCluster = [ele for ele in sortedClusters if ele not in usedClusters][0]

            # record it for later
            usedClusters.add(biggestCluster)

            # Initialize dummy array
            data2_M = np.zeros(base_labels.shape, dtype=int)

            # Select which country is being assessed
            data2_M[np.where(biggestCluster == compare_labels)] = 1 # Select cluster being assessed

            sortedOverlapForOneCluster[np.where(data2_M==1)] = 1
            sortedOverlap[compare_id, c1, :] = sortedOverlapForOneCluster

    # Fill in the base entry in the sorted overlap
    for c1 in range(max_clusters):  
        sortedOverlap[base_id, c1, :] = 1 * (base_labels == c1)

    return sortedOverlap

def get_ent_for_all_params(labels, param_id=0, baselab_id=0, num_members=50):

    """
    param_id - index for umap param combination
    baselab_id - ensemble index to use for baselabel
    """

    now = datetime.now()
    # print("base_label = "+str(baselab_id), now.strftime("%H:%M:%S"))

    labs = labels[param_id, :, :]
    sortedOverlap = get_overlap(ens_labels=labs, id=baselab_id, num_members=num_members)
    sortedOverlap_ = np.nan_to_num(sortedOverlap) # nan to zero
    df = pd.DataFrame(np.argmax(sortedOverlap_, axis=1)).T
    df = df.astype('int64')

    df_c = get_ent(df)
    # print('params, base_label, entropy', emb_params[id],k,df_c.entropy.mean())
    df_ent =  pd.DataFrame({'base_label': [baselab_id], 'mean_entropy': [df_c.entropy.mean()]})


    # ff = pd.concat(df_ent, axis=0)
    # ff.set_index('base_label', inplace=True)
    # nkey = emb_params[id]
    # ff.to_csv('/data/'+str(nkey[0])+'_'+str(nkey[1])+'_mean_entropy.csv')
    # print(ff)

    return df_ent


# Addtion to the above functions
# By Laique Djeutchouang


def get_numbers_from_filename(filename):
    """
    This function uses regex to find all sequences of digits in the filename and returns 
    them as a list of strings. For example, if the filename is "base_id_5_50_0.3_entropy_.csv",
    the function will return ['5', '50', '0', '3'].

    Args:
        filename (str): The filename to parse.

    Returns:
        list[str]: A list of digit sequences found in the filename.
    """
    import re
    return re.findall(r'\d+', filename)



def load_clusters(clust_dir:str, n_clusters:int) -> tuple[OrderedDict, int]:
    """
    Load cluster data from the specified directory.

    Args:
        n_clusters (int): Number of clusters.
        clust_dir (str): Path to the directory containing cluster files.

    Returns:
        OrderedDict: A dictionary of cluster data sorted by keys.
        int: The size of the cluster data array.
    """
    # Define data directory and the filenames to process
    nclusters_dir = f'{clust_dir}/nclusters_{n_clusters}'
    
    # Initialize an empty dictionary to store cluster data
    nclusters_dict = {}
    
    # Walk through the directory and load cluster data into the dictionary
    for root, dirs, files in os.walk(nclusters_dir):
        for file in files:
            cluster = np.load(os.path.join(root, file))
            num = get_numbers_from_filename(file)

            if int(num[0][0]) == 0:
                ens = int(num[0][1])
            else:
                ens = int(num[0])

            md = float(num[1] + '.' + num[2])
            nn = int(num[3])

            nclusters_dict[(ens, nn, md)] = cluster
    
    # Get the cluster data array size
    if cluster is None:
        raise ValueError("cluster was not assigned a value")
    ncluster_size = cluster.shape[0]
    
    # Sort the dictionary by keys
    sorted_nclusters_dict = OrderedDict(sorted(nclusters_dict.items(), key=lambda t: t[0]))

    return sorted_nclusters_dict, ncluster_size



def fill_labels_array(sorted_nclusters_dict:OrderedDict,
                      emb_params:list, num_members:int, n_pts:int) -> np.ndarray:
    """
    Fill a labels array with cluster data.

    Args:
        sorted_nclusters_dict (OrderedDict): Dictionary of cluster data as output of load_clusters.
        emb_params (list[tuple]): List of UMAP key parameter combinations - minimum distances & nearest neighhbors.
        num_members (int): Number of ensemble members.
        n_pts (int): Number of points.
        
    Returns:
        np.ndarray: A filled labels array.
    """
    
    # Initialize parameters
    n_param = len(emb_params)
    
    # Initialize labels array
    labels = np.zeros((n_param, num_members, n_pts)) * np.nan
    
    # Fill the labels array with clusters
    for k1, k2 in enumerate(range(1, num_members + 1)):
        for idx, (nn, md) in enumerate(emb_params):
            labels[idx, k1, :] = sorted_nclusters_dict[(k2, nn, md)]
            
    return labels


# List of UMAP key parameter combinations - minimum distances & nearest neighhbors.
UMAP_NNS_MDS = [(5, 0.1), (5, 0.3), (5, 0.5), (5, 0.7), (5, 0.9),
                (10, 0.1), (10, 0.3), (10, 0.5), (10, 0.7), (10, 0.9),
                (50, 0.1), (50, 0.3), (50, 0.5), (50, 0.7), (50, 0.9),
                (100, 0.1), (100, 0.3), (100, 0.5), (100, 0.7), (100, 0.9),
                (200, 0.1), (200, 0.3), (200, 0.5), (200, 0.7), (200, 0.9)]

bvb_terms = ['beta_V', 'BPT', 'Mass_flux', 'eta_dt', 'Curl_dudt', 'Curl_taus', 'Curl_taub', 'Curl_Adv', 'Curl_diff']


def reconstruct_DataArray(bvb_ds, embedded_nclusters_array) -> xr.Dataset:
    """
    Reconstruct the dataset with embedded clusters while preserving original 
    NaN patterns. Create an xarray Dataset from the cluster array data. 
    
    Args:
        bvb_ds (xarray.Dataset): Original xarray Dataset with NaN patterns.
        embedded_nclusters_array (numpy.ndarray): Array of embedded clusters.
        
    Returns:
        xarray.Dataset: Dataset containing the reconstructed cluster data.
    """
    # Make a copy of the original xarray Dataset before further preprocessing
    bvb_ds = bvb_ds[bvb_terms].copy()
    baseline_var = bvb_terms[0]
    da = bvb_ds[baseline_var].copy() 
    
    # Create mask of complete cases (no NaNs in any predictor)
    complete_mask = xr.full_like(da, True, dtype=bool)
    for var in bvb_terms:
        complete_mask = complete_mask & bvb_ds[var].notnull()
        
    # Create full array with NaNs
    full_shape = da.shape
    reconstructed = np.full(full_shape, np.nan)
    
    # Fill in predictions where we had complete cases
    reconstructed[complete_mask.values] = embedded_nclusters_array
    
    # Create DataArray with original structure
    nclusters_da = xr.DataArray(reconstructed,
                                dims=da.dims,
                                coords=da.coords)
    
    # Return cluster dataset with the original dataset structure
    nclusters_ds = xr.Dataset()
    nclusters_ds["geo_cluster"] = nclusters_da
    nclusters_ds["geo_cluster"].attrs['standard_name'] = "Geospatial ocean regimes"
    
    return nclusters_ds


def log_info(message):
    import logging 
    from datetime import datetime, timezone
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(asctime)s - %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%SZ'
    )

    # Create logger
    logger = logging.getLogger(__name__)

    # Set timestamp to UTC
    logging.Formatter.converter = lambda *args: datetime.now(timezone.utc).timetuple()
    
    # Log the message
    logger.info(message)


# Function to create directories if they do not exist
def make_dirs(data_dir):
    """Create a directory if it doesn't exist."""
    from pathlib import Path
    Path(data_dir).mkdir(parents=True, exist_ok=True)

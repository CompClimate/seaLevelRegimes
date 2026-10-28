"""
This utility will open cluster and entropy (.npy/.nc) files and apply dominant balance identification to for a given cluster 
"""

import numpy as np
import pandas as pd
import scipy
import xarray as xr
from itertools import combinations
import numpy as np
import xarray as xr
from scipy.optimize import linear_sum_assignment      # Hungarian


# BV budget terms display names (LaTeX)
bvb_terms = ["beta_V", "BPT", "Curl_Adv", "Curl_taus", "Curl_taub", "Curl_diff", "Mass_flux", "eta_dt", "Curl_dudt"]
bvb_tlabs = [r"$\beta\,\overline{V}$",
             r"$\overline{J(p_b,H)}/\rho_o$",
             r"$\hat{z}\cdot(\nabla\wedge\overline{\mathcal{A}})$",
             r"$\hat{z}\cdot(\nabla\wedge\overline{\tau_s})/\rho_o$",
             r"$\hat{z}\cdot(-\nabla\wedge\overline{\tau_b})/\rho_o$",
             r"$\hat{z}\cdot(\nabla\wedge\overline{\mathcal{B}})$",
             r"$-f\,\overline{Q_m}/\rho_o$",
             r"$f\,\partial_t\overline{\eta}$",
             r"$\hat{z}\cdot(-\nabla\wedge\overline{U_t})$",]


def dominant_balance(ds_bvb, nemi_labels, da_ent, single_instance=True, qt=None, percent_cutoff=10, nf=9, 
                     bias_flag='unbiased', return_all_balances=False, penalise_flag=True):
    """
    Main function that returns dominant balance following Kaiser et al. (2022)
    Offers option to confine region based on entropy cutoff - over that the balance is computed
    
    ds_bvb: budget file; should have same time dim name (not xofyear)
    nemi_labels: clusters
    da_ent: entropy
    qt: quantile to use as entropy cutoff; fraction
    percent_cutoff: to use as entropy cutoff
    bias_flag: 'unbiased'/'biases'
    nf: number of features/budget terms
    """

    if 'month' in ds_bvb.coords:
        ds_bvb = ds_bvb.rename({"month": "time"})
    if 'year' in ds_bvb.coords:
        ds_bvb = ds_bvb.rename({"year": "time"})   

    if single_instance:
        lab = nemi_labels.copy() # lab = nemi_labels.mean('time')
    else:
        lab = mode(nemi_labels, 'time') # Shape : (lat,lon); mode over time

    nc = int(max(np.unique(lab))) + 1 # Number of clusters
    
    if single_instance:
        ent = da_ent.copy()
    else:
        ent = da_ent.mean('time')   # Shape : (lat,lon); entropy over the time

    lst = [] # To contain the budget terms
    bal = np.zeros((nc, nf)) # To contain the dominant balance
    
    for i in range(nc):
        ent_ = ent.where(lab==i) # Entropy masked by cluster label
        
        if qt is not None:
            cf = ent_.quantile(qt,dim=('lat','lon',)) # qt'th percentile of entropy; shape : scalar
        else:
            cf = percent_cutoff # If using a percentage cutoff
            
        ent_masked = ent_.where(ent_<cf) # Entropy further masked by cutoff value
        mask = ent_masked.notnull() # Notnull entropy - mask
        
        e1 = ds_bvb.beta_V.where(mask).mean().to_numpy() # Average magnitude of budget term over the region        
        e2 = ds_bvb.BPT.where(mask).mean().to_numpy()
        e3 = ds_bvb.Curl_Adv.where(mask).mean().to_numpy()
        e4 = ds_bvb.Curl_taus.where(mask).mean().to_numpy()
        e5 = ds_bvb.Curl_taub.where(mask).mean().to_numpy()
        e6 = ds_bvb.Curl_diff.where(mask).mean().to_numpy()
        e7 = ds_bvb.Mass_flux.where(mask).mean().to_numpy()
        e8 = ds_bvb.eta_dt.where(mask).mean().to_numpy()
        e9 = ds_bvb.Curl_dudt.where(mask).mean().to_numpy()
        
        rsd = e1 - e2 - e3 - e4 - e5 - e6 - e7 - e8 - e9 # Residual term
        _ = pd.DataFrame(np.vstack([e1, e2, e3, e4, e5, e6, e7, e8, e9, rsd]))
        lst.append(_)

        # Dominant balance identification following Kaiser et al. (2022)
        balances = generate_balances(nf=nf, bias_flag=bias_flag)
        ngb = np.shape(balances)[0]
        # print('Number of hypothesis = ', ngb)
        features = np.vstack([e1, e2, e3, e4, e5, e6, e7, e8, e9]).T[0]
    
        M = np.zeros((ngb, ))
        for mm in range(ngb):
            M[mm] = get_m_score(balances[mm,:], features, bias=bias_flag, penalise=penalise_flag)    
    
        bal[i,:], _ = get_optimal_balance_from_score(M, features, balances)    

    df_dominant_balance = pd.DataFrame(bal)
    df_magnitudes = pd.concat(lst, axis = 1).T
    
    if nf==9:
        tlst = bvb_tlabs + [r"Residual"]
        df_magnitudes.columns = tlst
        df_dominant_balance.columns = bvb_tlabs

    if not return_all_balances:
        return df_magnitudes, df_dominant_balance.astype('int')
    else:
        return bal



def open_files(path_clusters, path_entropy, single_instance=False):

    """
    Opens clusters and entropy files
    """

    ds = xr.open_dataset(path_clusters)
    if 'month' in ds.coords:
            ds = ds.rename({"month": "time"})
    if 'year' in ds.coords:
        ds = ds.rename({"year": "time"})  
        
    nemi_labels = ds.bvb_regime.copy() # Cluster labels
    nc = max(np.unique(nemi_labels)) + 1
    
    if single_instance:
        lat, lon = nemi_labels.lat, nemi_labels.lon  # Shape : (lat,lon)

        nlat = lat.shape[0]
        nlon = lon.shape[0]
        
        ent = np.load(path_entropy)
        _ = np.reshape(ent, [nlat, nlon])
        da_ent = xr.DataArray(_, coords={'lat': lat, 'lon': lon}) 
        
    else:
        time, lat, lon = nemi_labels.time, nemi_labels.lat, nemi_labels.lon # Shape : (time, lat,lon)
        
        ntim = time.shape[0]
        nlat = lat.shape[0]
        nlon = lon.shape[0]
        
        ent = np.load(path_entropy)
        _ = np.reshape(ent, [ntim, nlat, nlon])
        da_ent = xr.DataArray(_, coords={'time': time,'lat': lat, 'lon': lon})       

    da_ent = da_ent/np.log2(nc)
    da_ent = da_ent * 100    

    return nemi_labels, da_ent
    

# Calculate mode 
def _mode(*args, **kwargs):
    vals = scipy.stats.mode(*args, **kwargs)
    # only return the mode (discard the count)
    return vals[0].squeeze()
    

def mode(obj, dim=None):
    # note: apply always moves core dimensions to the end
    # usually axis is simply -1 but scipy's mode function doesn't seem to like that
    # this means that this version will only work for DataArray's (not Datasets)
    assert isinstance(obj, xr.DataArray)
    axis = obj.ndim - 1
    
    mode_ds = xr.apply_ufunc(_mode, obj, input_core_dims=[[dim]], kwargs={'axis': axis, 'nan_policy': 'propagate'})
    return mode_ds


def return_H(D):
    valid_rows = []
    for i in range(2**D): 
        row = list(map(int, f"{i:0{D}b}"))
        
        # if sum(row) >= 2:
        valid_rows.append(row)
    

    return np.array(valid_rows)



"""
FROM: https://github.com/bekaiser/dominant_balance/blob/main/utils.py
"""

def get_optimal_balance_from_score(score, feature_mean, balance_combinations):
    maxabsval = np.amax(np.abs(feature_mean))
    loc_maxabsval = np.argwhere(np.abs(feature_mean)==maxabsval)
    
    score_max = np.nanmax(score)
    loc_matches = np.argwhere(score==np.amax(score))
    Nmatch = np.shape(np.argwhere(score==np.amax(score)))[0]
    
    if Nmatch > 1:
        # If there is more than one maximum score, find the balance that
        # includes the element with the maximum score.
        for i in range(Nmatch):
            balance = (balance_combinations[loc_matches[i],:])[0]
            if balance[loc_maxabsval] == 1.:
                balance_max = balance_combinations[loc_matches[i],:]
                
    elif Nmatch == 1:
        loc_max = (np.argwhere(score==np.amax(score))[:,0])[0]
        balance_max = balance_combinations[loc_max,:]
        
    elif Nmatch == 0:
        print('\nERROR: no maximum score found')
        print(' ')
        nf = np.shape(balance_combinations)[1]
        balance_max = np.ones([nf])
        score_max = nf / (2.*(nf-1.))
        
    return balance_max, score_max


def get_m_score(balance, features, bias = 'unbiased', penalise=True):
    # features need to be raw, i.e. not standardized.
    # print(np.shape(np.shape(features))[0])
    # if penalise:
    #     print('Penalising for large differences in magnitudes..')
    #     print('................................................')

    if bias == 'unbiased':
        if np.shape(np.shape(features))[0] == 1: # vector (single data sample)
    
            if (np.sum(balance) == features.shape[0]) or (np.sum(balance) == 0): # All ones or all zeros
                score = 0.
                
            else:
                if np.amin(np.abs(features)) == 0.:
                    nonzero_features = np.abs(features[np.nonzero(features)])
                    np.abs(features)/np.amin(np.abs(nonzero_features))
                    #features_normalized = np.abs(features)/np.array([1e-20])
                    
                else:
                    features_normalized = np.abs(features)/np.amin(np.abs(features))
                    
                select, remain = get_sets( balance, features_normalized  )
                select_max, select_min, select_imax, select_imin = find_maxmin( select )
                remain_max, remain_min, remain_imax, remain_imin = find_maxmin( remain )

                if select_min <= remain_max:
                    G = 0.
                    
                else:
                    G = ( np.log(select_min - remain_max) ) / ( np.log(select_min+remain_max) )

                G = max(G, 0.) # if G < 0, set to 0
                
                if select_min == select_max:
                    P = 0.
                else:
                    P =  np.log10(select_max) - np.log10(select_min)
                    
                if penalise:
                    # print('Penalising for large differences in magnitudes..')
                    score = G / (1. + P )
                else:
                    score = G 

        return score        

    else:
            
        if sum(balance) == 0.:
            balance = np.ones(np.shape(balance))
    
        # 1) Get the differences for the full set of features
        diffs_full, id_full = get_diff_vec(np.ones(np.shape(balance)) , features)
    
        # 2) Get the differences for the reduced set of features
        diffs_red, id_red = get_diff_vec(balance, features)
    
        # If any difference is exactly zero:
        if np.any(diffs_red==0.) == True:
            loc0 = np.argwhere(diffs_red==0.)[:, 0]
            diffs_red[loc0] = (1e-100) * np.ones(np.shape(loc0)[0])
    
        if np.any(diffs_full==0.) == True:
            loc0 = np.argwhere(diffs_full==0.)[:,0]
            diffs_full[loc0] = (1e-100)*np.ones(np.shape(loc0)[0])
    
        sum_score = np.sum(np.log10(diffs_red)) / np.sum(np.log10(diffs_full))
    
        # 5) Bias
        Mactive = np.shape(np.nonzero(balance)[0])[0] # Number of active features in subset
        Qactive = np.shape(diffs_red)[0] # Number of active differences in subset
        if Qactive == 0:
            print('\n\n ERROR: Nb = 0 \n\n')
        factor = (Qactive+1.) / (2.*Qactive) # (Nb+1)/(2*Nb)
    
        score = sum_score * factor

        return score
    

def get_diff_vec(grid_balance, grid_features):
    locs = np.nonzero(grid_balance)[0] # Grid balance is 0's and 1's for on/off terms
    if np.shape(locs)[0] == 0: # Quiescent balance: trivial solution
        locs = np.nonzero(np.ones(np.shape(grid_balance)))[0] # 4 zeros = 4 ones in terms of differences
        
    abs_balance = np.abs(grid_features[locs]) # Absolute values of 'on'  terms
    N_balance = np.shape(abs_balance)[0] # Number of 'on' terms
    max_abs_term = np.amax(np.abs(grid_features)) # Maximum term, may or may not be in grid_balance!
    max_loc = np.argwhere(abs_balance==max_abs_term)

    if np.shape(max_loc)[0] == 0:
        max_loc = np.nan
    else:
        max_loc = max_loc[0]
        
    diffs = np.zeros([N_balance])
    for ib in range(N_balance):
        if ib == max_loc:
            diffs[ib] = np.nan
        else:
            diffs[ib] = abs( max_abs_term - abs_balance[ib] ) /  abs( max_abs_term + abs_balance[ib] )

    id = locs[np.argwhere(np.isnan(diffs)-1)[:,0]] # Difference indices
    diffs = diffs[np.argwhere(np.isnan(diffs)-1)[:,0]] # Differences
    
    if np.shape(diffs)[0] == 0: # If no differences are found
        return np.array([1.]), 0
    else:
        return diffs, id


def find_maxmin(vector):
    # Find the max/min absolute values
    max = np.amax(np.abs(vector))
    min = np.amin(np.abs(vector))
    imax = np.argwhere(np.abs(vector)==max)[0,0]
    imin = np.argwhere(np.abs(vector)==min)[0,0]
    
    return max, min, imax, imin


def get_sets(hypothesis, vector):
    # Find the absolute values of the selected and remainder subsets
    iselect = np.flatnonzero(hypothesis)
    iremain = np.flatnonzero(hypothesis-1.)
    select = np.abs(vector[iselect])
    remain = np.abs(vector[iremain])
    
    return select, remain


def generate_balances(nf, bias_flag):
    # nf = number of features
    nc = int(np.power(2, nf)-1)
    grid_balance = np.zeros([nc, nf])
    i = 0
    for vertex in generate_vertices(nf):
        if sum(vertex) == 0.:
            continue
        else:
            grid_balance[i,:] = vertex
            i = i+1
    # Remove balances with only one term:
    grid_balance = grid_balance[nf:None,:]
    if bias_flag == 'unbiased':
        # Remove full set:
        grid_balance = grid_balance[0:np.shape(grid_balance)[0]-1,:]
        
    return grid_balance


def generate_vertices(n):
    # n is the number of dimensions of the cube (3 for a 3d cube)
    for number_of_ones in range(n + 1):
        for location_of_ones in combinations(range(n), number_of_ones):
            result = [0] * n
            for location in location_of_ones:
                result[location] = 1
            yield result    

"""
END: https://github.com/bekaiser/dominant_balance/blob/main/utils.py
"""            


def relabel_B_one_to_one(A: xr.DataArray,
                         B: xr.DataArray,
                         fill_value: int = -1):
    """
    Give every label in B a *unique* partner in A that maximises
    total overlap; extra labels are left at `fill_value`.

    Returns
    -------
    newB     : DataArray, same shape as B
    mapping  : dict {old_B_label: new_label}
    """
    # Distinct labels
    lab_A = np.unique(A)
    lab_B = np.unique(B)

    # Contingency / overlap matrix
    inv_A = {lab: i for i, lab in enumerate(lab_A)}
    inv_B = {lab: j for j, lab in enumerate(lab_B)}

    a_idx = np.vectorize(inv_A.get)(A.data.ravel())
    b_idx = np.vectorize(inv_B.get)(B.data.ravel())

    M = np.zeros((lab_A.size, lab_B.size), dtype=int)
    np.add.at(M, (a_idx, b_idx), 1)          # counts of co-occurrence

    # Hungarian ⇒ best one-to-one assignment (maximise overlap)
    row, col = linear_sum_assignment(-M)      # minus ⇒ maximise

    mapping = {lab_B[j]: lab_A[i] for i, j in zip(row, col)}

    # Any B-labels that never got a partner keep their own code,
    # or choose a filler (e.g. -1, or the next unused integer):
    for lab in lab_B:
        mapping.setdefault(lab, fill_value)

    # Apply *label* mapping (no further merging of pixels)
    newB = xr.apply_ufunc(lambda x: np.vectorize(mapping.get)(x),
                          B,  dask="allowed",  output_dtypes=[B.dtype]).rename(B.name + "_one_to_one")

    return newB, mapping



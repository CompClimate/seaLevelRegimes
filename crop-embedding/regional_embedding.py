import numpy as np
import xarray as xr
import xesmf as xe

bvb_terms = ["beta_V", "BPT", "Curl_Adv", "Curl_taus", "Curl_taub", "Curl_diff", "Mass_flux", "eta_dt", "Curl_dudt"]


class AuxFunctions():
    def __init__(self):
        """"""
        pass
    
    def log_info(self, message):
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


def out_regridder(res_deg="p25"):
    """
    Build target regular grid for the XESMF-based regridder.
    out_res_deg: float
        Target latitude/longitude resolution in degrees (default 0.125 for 1/8°).
    """
    af = AuxFunctions()
    
    # Build target regular grid 
    af.log_info(f"Build output dataset new regular lat-lon grid of CM4X-{res_deg} ...")
    # # Build 1D grid
    lat, lon = None, None
    if (res_deg == "p125") or (res_deg == 0.125):
        lat = np.linspace(-90 + 180/(2*2240), 90 - 180/(2*2240), 2240)
        lon = np.linspace(-180 + 360/(2*2880), 180 - 360/(2*2880), 2880)
    if (res_deg == "p25") or (res_deg == 0.25):
        lat = np.linspace(-90 + 180/(2*1080), 90 - 180/(2*1080), 1080)
        lon = np.linspace(-180 + 360/(2*1440), 180 - 360/(2*1440), 1440)
        
    # Build 2D grid
    lon2d, lat2d = np.meshgrid(lon, lat)

    # Prepare output grid 
    ds_out = xr.Dataset({"lat": (("y", "x"), lat2d),
                         "lon": (("y", "x"), lon2d),},
                         coords={"y": lat, "x": lon,},).set_coords(["lat", "lon"])
    return ds_out


def normalize_lon(lon_arr):
    """
    Normalize longitude coordinates to -180:180 to avoid wrapping issues
    """
    lon = np.array(lon_arr)
    lon = np.where(lon > 180, lon - 360, lon)
    lon = np.where(lon <= -180, lon + 360, lon)
    
    return lon


def rename_cm4x_dims(cm4x_ds):
    # Get dimensions of the CM4X dataset
    ds = cm4x_ds.copy()
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name=ds.name or "variable")
    elif not isinstance(ds, xr.Dataset):
        raise TypeError("Input must be an xarray.DataArray or xarray.Dataset.")
    dims = set(ds.dims)
    
    # Rename T-grid dims and coords if present
    if {"yh", "xh"}.issubset(dims):
        grid = "T-grid/center (yh, xh)"
        ds = ds.rename({"yh": "y", "xh": "x", "geolat": "lat", "geolon": "lon"})

    # Rename Q-grid dims and coords if present
    elif {"yq", "xq"}.issubset(dims):
        grid = "Q-grid/corner (yq, xq)"
        ds = ds.rename({"yq": "y", "xq": "x", "geolat_c": "lat", "geolon_c": "lon"})

    # Rename U-grid dims and coords if present
    elif {"yh", "xq"}.issubset(dims):
        grid = "U-grid (yh, xq)"
        ds = ds.rename({"yh": "y", "xq": "x", "geolat_u": "lat", "geolon_u": "lon"})

    # Rename V-grid dims and coords if present
    elif {"yq", "xh"}.issubset(dims):
        grid = "V-grid (yq, xh)"
        ds = ds.rename({"yq": "y", "xh": "x", "geolat_v": "lat", "geolon_v": "lon"})
    
    elif len(dims) <= 2:
        if {"yq"}.issubset(dims):
            ds = ds.rename({"yq": "y"})
            grid = "Q-grid (yq)"
        elif {"yh"}.issubset(dims):
            ds = ds.rename({"yh": "y"})
            grid = "T-grid (yh)"
        elif {"xq"}.issubset(dims):
            ds = ds.rename({"xq": "x"})
            grid = "Q-grid (xq)"
        elif {"xh"}.issubset(dims):
            ds = ds.rename({"xh": "x"})
            grid = "T-grid (xh)"
    else:
        raise ValueError("Unknown grid! The grid must be one of: "
                         "[T-grid/center (yh,xh), Q-grid/corner (yq,xq), "
                         "U-grid (yh,xq), V-grid (yq,xh)], (yh), (xh), (yq), (xq)")
    
    # Rename and Normalize longitudes to -180:180 to avoid wrapping issues
    try:
        ds = ds.assign_coords(x=normalize_lon(ds["x"])).sortby("x")
    except KeyError:
        pass

    return ds, grid
    

def apply_regridder(cm4x_ds, ds_out=None, res_deg="p25", method='nearest_s2d'):
    """
    Regrid the CM4X dataset to a regular lat-lon grid at resolution `out_res_deg`.
    
    Parameters
    ----------
    cm4x_ds: xarray.Dataset
        Input CM4X dataset (opened with xarray; may be dask-backed).
    voi_name: string 
        Name of variable to regrid.
    res_deg: float
        Target latitude/longitude resolution in degrees (default 0.125 for 1/8°).
    method: str
        Regridding method - 'nearest_s2d' by default. Alternatives: 'conservative', 'nearest_s2d', etc.
    
    Returns
    -------
    xarray.Dataset
        Regridded dataset on regular lat-lon grid with dims (time, depth, lat, lon) or (time, lat, lon), and the grid type.
    """
    af = AuxFunctions()
    if ds_out is None:
        ds_out = out_regridder(res_deg)
    if res_deg is None and ds_out is None:
        raise ValueError("At least one of these two input arguments, 'ds_out' and 'res_deg', is required. Both cannot be empty!")

    
    af.log_info("Rename input dims and Normalize longitudes to -180:180 ...")
    # Rename dims
    ds, grid = rename_cm4x_dims(cm4x_ds=cm4x_ds)
    attrs = ds.attrs.copy()
    print(f"Regridding from –– {grid}")
    
    af.log_info("Prepare input dataset for new regular lat-lon grid ...")
    # Prepare input grid 
    lon2d, lat2d = ds["lon"].values, ds["lat"].values
    ds_in = xr.Dataset({"lat": (("y", "x"), lat2d),
                        "lon": (("y", "x"), lon2d),}).set_coords(["lat", "lon"])
    
    af.log_info("Build the xesmf-based regridder ...")
    # Build regridder 
    ds_in.attrs = attrs
    ds_out.attrs = attrs
    regridder = xe.Regridder(ds_in, ds_out, method, periodic=True, reuse_weights=False)

    af.log_info("Apply to the new regular lat-lon regridder to dataset ...")
    # Apply to dataset 
    ds_reg = regridder(ds)
    ds_reg.attrs = attrs

    return ds_reg


def extract_regional_embedding(emb_path, bvb_path, 
                               bvb_terms=bvb_terms,
                               reg_bounds={"lon": slice(-95, 10.5), "lat": slice(9.5, 70.5)}, 
                               data_res="p25", geo_path=None,
                               base_dir="/group/maikesgrp/laique/PPAN/CM4X",):
    """
    Back-project UMAP embeddings, regrid geographically, and extract valid regional samples.
    """

    # Load UMAP embeddings and original BVB data
    embeddings = np.load(emb_path)
    bvb = xr.open_zarr(bvb_path, chunks=None)

    # Identify grid cells with complete, finite predictor data
    baseline = bvb[bvb_terms[0]]
    valid = xr.ones_like(baseline, dtype=bool)
    for var in bvb_terms:
        valid &= np.isfinite(bvb[var])
    valid = valid.values
    baseline = baseline.load()

    # Load geographic coordinates from the original grid
    if {"geolat_c", "geolon_c"}.issubset(bvb.variables):
        geo = bvb[["geolat_c", "geolon_c"]].load()
    else:
        geo_path = geo_path or f"{base_dir}/BVB/CM4X-{data_res}_BVB_and_tracers_geocoords_2005_2014_time_mean.zarr"
        with xr.open_zarr(geo_path, chunks=None) as geo_ds:
            geo = geo_ds[["geolat_c", "geolon_c"]].load()
    bvb.close()

    # Validate embedding samples against the original complete-case mask
    if embeddings.ndim != 2 or embeddings.shape[0] != valid.sum():
        raise ValueError(
            f"Expected embeddings with shape (n_valid, n_components), "
            f"where n_valid={valid.sum():,}; received {embeddings.shape}. "
            "Check that the embedding uses the same predictors, grid, "
            "mask, and flattening order as the original data.")

    # Back-project all components simultaneously onto the native grid
    full = np.full((*baseline.shape, embeddings.shape[1]), np.nan)
    full[valid, :] = embeddings

    emb_da = xr.DataArray(full, dims=(*baseline.dims, "UMAP_comp"),
                          coords={**baseline.coords, "UMAP_comp": np.arange(1, embeddings.shape[1] + 1),},
                          name="embedding", attrs={"long_name": "UMAP embedding components"},)
    emb_ds = emb_da.to_dataset().assign_coords(geolat_c=geo["geolat_c"], geolon_c=geo["geolon_c"],)

    # Regrid onto the target geographic grid
    emb_ds = apply_regridder(cm4x_ds=emb_ds, res_deg=data_res,).drop_vars(["lat", "lon"], errors="ignore")
    emb_ds = emb_ds.rename({"y": "lat", "x": "lon"})
    emb_ds = emb_ds.transpose("month", "lat", "lon", "UMAP_comp")
    emb_ds.attrs.update({
        "description": (
            f"{embeddings.shape[1]}D UMAP embedding back-projected "
            "onto the regridded BVB geographic grid."),
        "embedding_path": str(emb_path),
        "bvb_path": str(bvb_path),
        "data_resolution": data_res,
        "regridding_resolution_degrees": data_res,})

    # Select the requested region
    if reg_bounds is not None:
        emb_ds = emb_ds.sel(reg_bounds)

    # Stack grid cells and retain samples valid across all components
    sample_dims = [d for d in ("month", "lat", "lon") if d in emb_ds.dims]
    X = emb_ds["embedding"].stack(n_grid=sample_dims).transpose("n_grid", "UMAP_comp")
    sub_emb = X.where(np.isfinite(X).all("UMAP_comp"), drop=True)

    return sub_emb, emb_ds



# Predictability of Sea Level Regimes

Determining and predicting regimes of sea level variability as outlined with the vorticity budgets.

---

## Quick summary

- Step 1 — embeddings: generate low-dimensional embeddings from input fields (UMAP).
- Step 2 — build clusterings from embeddings.
- Step 3 — entropy & baseline labels: compute entropy-like diagnostics per cluster.
- Step 4 — regimes identification: identify and export regimes using entropy + clustering.


---

## Prerequisites

- Linux (tested on HPC clusters)
- Python 3.10+ (uses conda in provided scripts)
- Conda environment (example name `proc_env`) with required packages:
  - numpy, scipy, pandas, xarray, scikit-learn, umap-learn, matplotlib, dask (optional), tensorflow (optional), joblib
- SLURM if running on cluster

Example minimal steps:

```bash
# Create environment (example)
conda create -n proc_env python=3.10 -y
conda activate proc_env
pip install -r requirements.txt   # at the repository root
```


---

## Extract a regional embedding

`crop-embedding/regional_embedding.py` maps a step 1 UMAP embedding back onto the native CM4X grid, regrids it to a regular lat-lon grid (xESMF, `nearest_s2d`), and optionally crops it to a region of interest.

Extra requirements: `xesmf` (install from conda-forge: `conda install -c conda-forge xesmf`) and `zarr`.

Import:

```python
# From inside crop-embedding/
from regional_embedding import extract_regional_embedding

# From anywhere else, add the folder to the path first
import sys
sys.path.append("/path/to/seaLevelRegimes/crop-embedding")
from regional_embedding import extract_regional_embedding
```

Run:

```python
sub_emb, emb_ds = extract_regional_embedding(
    emb_path="/path/to/embedding.npy",           # UMAP output, shape (n_valid, n_components)
    bvb_path="/path/to/CM4X-p25_BVB.zarr",       # BVB data the embedding was built from
    bvb_terms=["beta_V", "BPT", "Curl_Adv"],     # same predictors, same order, as in step 1
    reg_bounds={"lon": slice(-95, 10.5), "lat": slice(9.5, 70.5)},  # defaults to the North Atlantic ocean
    data_res="p25",                              # "p25" (1/4°) or "p125" (1/8°)
)
```

Arguments:

- `emb_path` — `.npy` file with one row per valid grid cell of `bvb_path`.
- `bvb_path` — Zarr store with the BVB variables and a `month` dimension.
- `bvb_terms` — variables used to build the embedding; a cell is valid only if all of them are finite. Defaults to the nine BVB terms: `beta_V`, `BPT`, `Curl_Adv`, `Curl_taus`, `Curl_taub`, `Curl_diff`, `Mass_flux`, `eta_dt`, `Curl_dudt`.
- `reg_bounds` — dict passed to `xarray.Dataset.sel`. Longitudes are in -180 to 180 and both axes are ascending. Pass `None` to keep the global field.
- `data_res` — target grid resolution, `"p25"` or `"p125"`.
- `geo_path` — Zarr store with `geolat_c`/`geolon_c`, used when `bvb_path` does not contain them. Defaults to `{base_dir}/BVB/CM4X-{data_res}_BVB_and_tracers_geocoords_2005_2014_time_mean.zarr`.
- `base_dir` — root used to build the default `geo_path` (default `/group/maikesgrp/laique/PPAN/CM4X`).

Returns:

- `sub_emb` — `xarray.DataArray` of shape `(n_grid, UMAP_comp)`, the valid samples in the region, ready for clustering.
- `emb_ds` — `xarray.Dataset` whose `embedding` variable has dims `(month, lat, lon, UMAP_comp)`.

A `ValueError` means the number of embedding rows does not match the number of valid cells: check that `bvb_terms`, the grid, and the mask match the run that produced the embedding.

---

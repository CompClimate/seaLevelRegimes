# Export embedding

import io
import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xarray as xr

# Ensure UTF-8 encoding for stdout
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%SZ',
)
log = logging.getLogger(__name__)




# dimension aliases accepted on the input grid, in sample order
GRID_DIM_ALIASES = (('time',), ('lat', 'latitude', 'y'), ('lon', 'longitude', 'x'))
BVB_TERMS = ["beta_V", "BPT", "Mass_flux", "eta_dt", "Curl_dudt", "Curl_taus", 
             "Curl_taub", "Curl_Adv", "Curl_diff"]

class GeoGrid:
    """ The mapping between the input grid and the NEMI sample axis.

    Samples are the finite grid cells taken in C order, so a plain reshape
    inverts the flattening. ``valid`` records which cells survived the finite
    mask (land and missing data are dropped before embedding) and ``coords``
    restores the axes, which is what puts the vote map and the entropy back on
    (lat, lon) or (time, lat, lon) at the end of the run.
    """

    def __init__(self, dims, coords, valid, features):
        self.dims = tuple(dims)
        self.coords = {key: np.asarray(value) for key, value in coords.items()}
        self.valid = np.asarray(valid)
        self.features = list(features)

    @property
    def shape(self):
        return tuple(len(self.coords[dim]) for dim in self.dims)

    def unstack(self, values):
        """ Scatter a per-sample vector back onto the full grid (masked cells NaN). """
        flat = np.full(self.valid.size, np.nan, dtype=float)
        flat[self.valid] = values
        return flat.reshape(self.shape)

    def to_dataset(self, fields):
        """ Back-project ``{name: per-sample vector}`` into a geo-referenced Dataset. """
        return xr.Dataset(
            {name: (self.dims, self.unstack(values)) for name, values in fields.items()},
            coords={dim: self.coords[dim] for dim in self.dims})

    def save(self, path):
        """ Persist the grid so '--mode cluster' can back-project without the input. """
        np.savez(path, dims=np.array(self.dims), valid=self.valid,
                 features=np.array(self.features),
                 **{f"coord_{dim}": self.coords[dim] for dim in self.dims})

    @classmethod
    def load(cls, path):
        saved = np.load(path, allow_pickle=True)
        dims = [str(dim) for dim in saved['dims']]
        return cls(dims, {dim: saved[f"coord_{dim}"] for dim in dims},
                   saved['valid'], [str(name) for name in saved['features']])


def grid_dims(ds):
    """ The (time,) lat, lon dimensions of a dataset, in sample order. """
    dims = []
    for aliases in GRID_DIM_ALIASES:
        found = [alias for alias in aliases if alias in ds.dims]
        if found:
            dims.append(found[0])
        elif aliases[0] != 'time':          # time is optional, lat/lon are not
            raise KeyError(f"no '{aliases[0]}' dimension on the input; "
                           f"found {tuple(ds.dims)}")
    return tuple(dims)


def load_grid(path, features=BVB_TERMS):
    """ Flatten a gridded ``.nc``/``.zarr`` dataset into the NEMI feature matrix.

    Every data variable is a feature and every grid cell -- (lat, lon) or
    (time, lat, lon) -- is a sample. Cells that are not finite in all features
    (land, missing data) are dropped, so ``X`` is dense.

    Args:
        path (str): the ``.nc`` or ``.zarr`` input.
        features (list, optional): variables to use, in order. Defaults to
            every BV term data variable in the file.

    Returns:
        (X, grid): the (``n_samples``, ``n_features``) matrix and the
        :class:`GeoGrid` that maps it back onto the geo coordinates.
    """
    ds = xr.open_zarr(path, chunks=None) if str(path).endswith('.zarr') \
        else xr.open_dataset(path, chunks=None)

    names = list(features) if features else list(ds.data_vars)
    missing = [name for name in names if name not in ds.data_vars]
    if missing:
        raise KeyError(f"variables missing from {path}: {missing}; "
                       f"available: {sorted(ds.data_vars)}")
    ds = ds[names]
    dims = grid_dims(ds)

    n_cells = int(np.prod([ds.sizes[dim] for dim in dims]))
    X = np.empty((n_cells, len(names)), dtype=np.float32)
    for column, name in enumerate(names):
        if set(ds[name].dims) != set(dims):
            raise ValueError(f"variable '{name}' has dims {ds[name].dims}, expected {dims}")
        X[:, column] = ds[name].transpose(*dims).values.reshape(-1)

    valid = np.isfinite(X).all(axis=1)
    grid = GeoGrid(dims, {dim: ds[dim].values for dim in dims}, valid, names)
    print(f"Loaded {path}")
    print(f"Grid: {dict(zip(dims, grid.shape))} | features: {names}")
    print(f"Samples: {valid.sum():,} finite cells of {valid.size:,} "
          f"({100 * valid.mean():.1f}%)")
    return X[valid], grid


def load_input(path, features=BVB_TERMS):
    """ Feature matrix from a gridded ``.nc``/``.zarr`` file, or a plain ``.npy``.

    The ``.npy`` route takes an (``n_samples``, ``n_features``) matrix as-is and
    carries no geo coordinates, so its results cannot be back-projected.
    """
    if str(path).endswith('.npy'):
        return np.load(path), None
    return load_grid(path, features)


# ---------------------------------------------------------------------------
# umap-learn UMAP initialisation (CPU, scikit-learn-compatible estimator).
# ---------------------------------------------------------------------------
def _init_umap():
    """Import umap-learn's UMAP. Exits on failure."""
    try:
        from umap import UMAP as skUMAP
        log.info("umap-learn detected – using CPU UMAP.")
        return skUMAP
    except ImportError as exc:
        log.error("umap-learn import failed (dependency or installation issue): %s", exc)
        sys.exit(1)
    except Exception as exc:
        log.error("Failed to initialise umap-learn (%s: %s).", type(exc).__name__, exc)
        sys.exit(1)


UMAP = _init_umap()

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def process_embedding(data: np.ndarray, umap_kwargs: dict,
                      output_file: str) -> np.ndarray | None:
    """
    Compute a UMAP embedding with umap-learn (CPU) and save it to disk.

    Args:
        data (np.ndarray): Input data, shape (n_samples, n_features).
        umap_kwargs (dict): UMAP keyword arguments:
            - umap_md (float): min_dist, default 0.1
            - umap_nn (int): n_neighbors, default 200
            - n_epochs (int): training epochs, default None (auto)
            - learning_rate (float): default 1.0
            - init (str): initialisation method, default 'random'
            - umap_rs (int): random_state, default 42
        output_file (str): Path to save the output .npy embedding.
    Returns:
        np.ndarray if the embedding was computed, None if it already existed.
    """
    umap_md       = umap_kwargs.get("umap_md", 0.1)
    umap_nn       = umap_kwargs.get("umap_nn", 200)
    umap_rs     = umap_kwargs.get("umap_rs", 42)
    n_epochs      = umap_kwargs.get("n_epochs", None)
    init          = umap_kwargs.get("init", "random")
    learning_rate = umap_kwargs.get("learning_rate", 1.0)

    if os.path.exists(output_file):
        log.info(f'Embedding already exists – MD={umap_md}, NN={umap_nn}.')
        print(f'It can be loaded at:\n<{output_file}>\n')
        return None

    log.info(f'Computing embedding: MD{umap_md}, NN: {umap_nn}')

    umap_params = dict(n_neighbors  = umap_nn,
                       n_components = 3,
                       min_dist     = umap_md,
                       learning_rate= learning_rate,
                       init         = init,
                       # umap-learn uses None for auto (unlike cuML's 0).
                       n_epochs     = n_epochs,
                       random_state = umap_rs)

    embedding = UMAP(**umap_params).fit_transform(data.astype(np.float32))
    if not isinstance(embedding, np.ndarray):
        embedding = np.asarray(embedding)

    log.info('Saving embedding ...')
    np.save(output_file, embedding)

    return embedding


def run_embedding(input_file: str, umap_kwargs: dict,
                  output_file: str) -> np.ndarray | None:
    """
    Load input data, run umap-learn UMAP embedding (CPU), and save the result.

    Args:
        input_file (str): Path to the input Parquet data file.
        umap_kwargs (dict): UMAP keyword arguments (see process_embedding).
        output_file (str): Path to save the output .npy embedding.
    Returns:
        np.ndarray if the embedding was computed, None if it already existed.
    """
    print("\n-------------- STARTING NEMI RUN ----------------\n")

    log.info("Loading data ...")
    data, grid = load_input(input_file, features=BVB_TERMS)
    print('Scaling the features')
    data = StandardScaler().fit_transform(data)

    log.info('Started the Manifold Representation Learning.')
    embedding = process_embedding(data, umap_kwargs, output_file)
    log.info('Completed the Manifold Representation Learning.')

    return embedding


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="umap-learn CPU UMAP embedding")
    parser.add_argument("input_file",  type=str)
    parser.add_argument("min_dist",    type=float)
    parser.add_argument("n_neighbors", type=int)
    parser.add_argument("random_state", type=str, default="42")
    parser.add_argument("output_file", type=str)
    args = parser.parse_args()

    t0 = time.time()

    umap_kwargs = {"umap_md": args.min_dist,
                   "umap_nn": args.n_neighbors,
                   "umap_rs": int(args.random_state),
                   "init":    "random",  # Critical for HPC robustness
                   }

    _ = run_embedding(input_file=args.input_file,
                      umap_kwargs=umap_kwargs,
                      output_file=args.output_file)

    print(f"\nTotal execution time: {time.time() - t0:.2f} s\n")

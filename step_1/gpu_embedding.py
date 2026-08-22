# Export embedding

import io
import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd

# Ensure UTF-8 encoding for stdout
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%SZ',
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# cuML UMAP initialisation — fails fast if GPU / cuML is unavailable.
# ---------------------------------------------------------------------------
def _init_cuml_umap():
    """Import cuML UMAP and bind to GPU device 0. Exits on failure."""
    try:
        import warnings
        import cupy
        if cupy.cuda.runtime.getDeviceCount() == 0:
            log.error("No CUDA-capable GPU found. This script requires a GPU with cuML.")
            sys.exit(1)
        # Explicitly bind to device 0 so cuML doesn't bypass CUDA_VISIBLE_DEVICES.
        cupy.cuda.Device(0).use()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            import cuml
            from cuml.manifold import UMAP as cuUMAP
        cuml.set_global_output_type("numpy")  # always return np.ndarray
        log.info("RAPIDS cuML detected – using GPU-accelerated UMAP.")
        return cuUMAP
    except ImportError as exc:
        log.error("cuML import failed (dependency or installation issue): %s", exc)
        sys.exit(1)
    except Exception as exc:
        log.error("Failed to initialise cuML (%s: %s).", type(exc).__name__, exc)
        sys.exit(1)


UMAP = _init_cuml_umap()


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def process_embedding(data: np.ndarray, umap_kwargs: dict,
                      output_file: str) -> np.ndarray | None:
    """
    Compute a UMAP embedding with cuML and save it to disk.

    Args:
        data (np.ndarray): Input data, shape (n_samples, n_features).
        umap_kwargs (dict): UMAP keyword arguments:
            - umap_md (float): min_dist, default 0.1
            - umap_nn (int): n_neighbors, default 200
            - n_epochs (int): training epochs, default None (auto → 0 for cuML)
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

    print(f"{'='*7} Key Parameters Used (with defaults for others) {'='*7}")
    print(f"{' '*5}• Backend          = cuML (GPU)")
    print(f"{' '*5}• UMAP min_dist    = {umap_md}")
    print(f"{' '*5}• UMAP n_neighbors = {umap_nn}")
    print(f"{' '*5}• UMAP random_state = {umap_rs}")
    print(f"{' '*5}• UMAP learning_rate = {learning_rate}")
    print(f"{' '*5}• UMAP n_epochs    = {n_epochs}")
    print(f"{' '*5}• UMAP init        = {init}")

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
                       # cuML uses 0 for auto; None is not accepted.
                       n_epochs     = 0 if n_epochs is None else n_epochs,
                       random_state = umap_rs)
    # nn_descent (cuML default) requires n_neighbors < internal graph_degree (~64).
    # Switch to brute-force KNN when n_neighbors is large enough to hit that limit.
    if umap_nn >= 64:
        umap_params["build_algo"] = "brute_force_knn"

    embedding = UMAP(**umap_params).fit_transform(data.astype(np.float32))
    if not isinstance(embedding, np.ndarray):
        embedding = np.asarray(embedding)

    log.info('Saving embedding ...')
    np.save(output_file, embedding)

    return embedding


def run_embedding(input_file: str, umap_kwargs: dict,
                  output_file: str) -> np.ndarray | None:
    """
    Load input data, run cuML UMAP embedding, and save the result.

    Args:
        input_file (str): Path to the input Parquet data file.
        umap_kwargs (dict): UMAP keyword arguments (see process_embedding).
        output_file (str): Path to save the output .npy embedding.
    Returns:
        np.ndarray if the embedding was computed, None if it already existed.
    """
    print("\n-------------- STARTING NEMI RUN ----------------\n")

    log.info("Loading data ...")
    data = pd.read_parquet(input_file).values

    log.info('Started the Manifold Representation Learning.')
    embedding = process_embedding(data, umap_kwargs, output_file)
    log.info('Completed the Manifold Representation Learning.')

    return embedding


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="cuML GPU UMAP embedding")
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

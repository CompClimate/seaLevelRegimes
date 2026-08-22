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
    data = pd.read_parquet(input_file).values

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

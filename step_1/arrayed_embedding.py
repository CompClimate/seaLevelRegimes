# Export embedding (HPC-safe, restartable)

import io
import os
import sys
import time
import argparse
import pickle
import numpy as np
import pandas as pd

# Ensure UTF-8 encoding for stdout
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# -----------------------------------------------------------------------------
# Path setup (unchanged logic, but safer)
# -----------------------------------------------------------------------------
SOURCE = os.path.abspath("/home/Laique.Djeutchouang/DEVs/BV-Regimes/NEMI/seaLevelRegimes")

if SOURCE not in sys.path:
    sys.path.insert(1, SOURCE)

from src import nemi_func as nf
from src import aux_func as af

# -----------------------------------------------------------------------------
# Base directory (static data location)
# -----------------------------------------------------------------------------
BASE_DIR = "/work/lnd/CM4X/NEMI"


# -----------------------------------------------------------------------------
# Checkpoint helpers
# -----------------------------------------------------------------------------
def save_checkpoint(path, state):
    if path is None:
        return
    with open(path, "wb") as f:
        pickle.dump(state, f)


def load_checkpoint(path):
    if path is None or not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------------------------
# Embedding core
# -----------------------------------------------------------------------------
def process_embedding(data, umap_kwargs, output_file, checkpoint=None,):
    """
    Compute (or resume) UMAP embedding and save result.
    Args:
        data (np.ndarray): Input data to embed.
        umap_kwargs (dict): UMAP parameters.
        output_file (str): Output file path for the embedding.
        checkpoint (str, optional): Checkpoint file path. Defaults to None.
    Returns:
        np.ndarray: The computed UMAP embedding.
    """

    umap_md = umap_kwargs.get("umap_md", 0.5)
    umap_nn = umap_kwargs.get("umap_nn", 200)
    n_epochs = umap_kwargs.get("n_epochs", None)
    init = umap_kwargs.get("init", "random")
    n_jobs = umap_kwargs.get("n_jobs", -1)
    learning_rate = umap_kwargs.get("learning_rate", 1.0)

    # ------------------------------------------------------------------
    # Output guard (idempotent)
    # ------------------------------------------------------------------
    if os.path.exists(output_file):
        af.log_info(f"Embedding already exists: <{output_file.split('/')[-1]}>")
        return np.load(output_file)

    # ------------------------------------------------------------------
    # Load checkpoint if available
    # ------------------------------------------------------------------
    state = load_checkpoint(checkpoint) or {}

    # ------------------------------------------------------------------
    # UMAP embedding (checkpointed)
    # ------------------------------------------------------------------
    if "embedding" not in state:
        af.log_info(f"Running UMAP: (md={umap_md}, nn={umap_nn}) ...")
        embedding = nf.apply_umap(dfn=data,
                                  min_dist=umap_md,
                                  umap_neighbors=umap_nn,
                                  learning_rate=learning_rate,
                                  n_epochs=n_epochs,
                                  init=init,
                                  n_jobs=n_jobs)

        state["embedding"] = embedding
        save_checkpoint(checkpoint, state)
    else:
        af.log_info("Resuming embedding from checkpoint")
        embedding = state["embedding"]

    af.log_info("Saving embedding to disk ...")
    np.save(output_file, embedding)

    return embedding


# -----------------------------------------------------------------------------
# Top-level runner (Slurm-friendly)
# -----------------------------------------------------------------------------
def run_embedding(input_file, umap_kwargs, output_file, checkpoint=None,):
    af.log_info("Loading data ...")
    data = pd.read_parquet(input_file).values

    embedding = process_embedding(data=data, umap_kwargs=umap_kwargs, output_file=output_file, checkpoint=checkpoint,)

    return embedding


# -----------------------------------------------------------------------------
# CLI entry point (job-array aligned)
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="UMAP embedding for BV budget (HPC-safe)")

    parser.add_argument("input_file", type=str)
    parser.add_argument("min_dist", type=float)
    parser.add_argument("n_neighbors", type=int)
    parser.add_argument("output_file", type=str)
    parser.add_argument("member", type=int)

    parser.add_argument("--checkpoint", type=str, default=None)

    args = parser.parse_args()

    start = time.time()
    
    umap_kwargs = {"umap_md": args.min_dist,
                   "umap_nn": args.n_neighbors,
                   "init": "random",  # Critical for HPC robustness
                   "n_jobs": int(os.environ.get("SLURM_CPUS_PER_TASK", 1)), # Use allocated CPUs or number of threads
                   }

    _ = run_embedding(input_file=args.input_file,
                      umap_kwargs=umap_kwargs,
                      output_file=args.output_file,
                      checkpoint=args.checkpoint,)

    elapsed = time.time() - start
    print(f"\n✔ Total execution time: {elapsed:.2f} s\n")

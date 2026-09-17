"""
dbs_batched.py
=======================
Memory-safe single (eps, min_samples) DBSCAN run on a large embedding.

Meant to be invoked once per (eps, min_samples) combination — e.g. from a
Slurm job array — as the parallel counterpart to sweep_batched.py, which
sweeps the full (eps, ms) grid serially on one node.

Why this is memory-safe:
  * The neighbour graph is built once, in CHUNKS, to cap peak memory.
  * If the graph would exceed the node's RAM budget the run aborts before
    materializing it, instead of OOMing (see MAX_EDGES in build_graph).

Usage:
    python dbs_batched.py <embedding.npy> --eps 0.1 --min-samples 50 \\
        --output-dir <dir> --node-ram-gb 128
"""

import argparse
import logging
import sys
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors, sort_graph_by_row_values
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

CHUNK          = 500_000  # points per radius-query batch (lower → less peak RAM)
BYTES_PER_EDGE = 12        # float32 distance (4B) + int64 index (8B) once nnz > 2^31


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(nn: NearestNeighbors, X: np.ndarray, eps: float,
                node_ram_gb: int) -> sp.csr_matrix | None:
    """
    Chunked radius-neighbours graph at `eps`. None if too dense for node_ram_gb.
    """
    t0 = time.time()
    max_edges = (node_ram_gb * 1e9) / (2 * BYTES_PER_EDGE)  # ÷2: sp.vstack briefly holds
                                                              # both blocks and merged graph
    n = len(X)
    blocks, edges = [], 0
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        g = nn.radius_neighbors_graph(X[start:stop], radius=eps,
                                      mode="distance", sort_results=False)
        g.data = g.data.astype(np.float32, copy=False)  # BallTree returns float64 regardless of X's dtype
        edges += g.nnz
        if edges > max_edges:
            log.warning(f"eps={eps}: >{max_edges:,.0f} edges — too dense for "
                       f"{node_ram_gb} GB node, aborting.")
            return None
        blocks.append(g)

    G = sp.vstack(blocks).tocsr()
    G.data[G.data == 0] = 1e-12                              # keep true zero-distance edges
    sort_graph_by_row_values(G, warn_when_not_sorted=False)   # DBSCAN prefers sorted rows
    log.info(f"eps={eps}: graph {G.nnz:,} edges, {G.data.nbytes/1e9:.1f} GB ({time.time()-t0:.0f}s)")
    return G


# ---------------------------------------------------------------------------
# DBSCAN fit
# ---------------------------------------------------------------------------

def run_dbscan(X: np.ndarray, eps: float, min_samples: int, node_ram_gb: int,
              algorithm: str = "ball_tree") -> dict | None:
    """
    Run one (eps, min_samples) DBSCAN fit. None if the graph was too dense.
    Args:
        X (np.ndarray): Input embedding, shape (n_samples, n_features).
        eps (float): Neighbourhood radius.
        min_samples (int): Minimum samples to form a core point.
        node_ram_gb (int): Node RAM budget in GB.
        algorithm (str): DBSCAN search algorithm. 
    Returns:
        dict: Cluster-quality metrics and labels, or None if the graph was too dense.
    """
    t0 = time.time()
    nn = NearestNeighbors(algorithm=algorithm, leaf_size=10, n_jobs=-1).fit(X)
    log.info(f"BallTree built in {time.time()-t0:.0f}s")
    G = build_graph(nn, X, eps, node_ram_gb)
    if G is None:
        return None

    labels     = DBSCAN(eps=eps, min_samples=min_samples,
                        metric="precomputed", n_jobs=-1).fit_predict(G)
    n_clusters = int(labels.max()) + 1
    noise_frac = float((labels == -1).mean())
    coverage   = 1 - noise_frac

    # Balance-penalized effective coverage: coverage × (1 − HHI), where HHI is the
    # Herfindahl-Hirschman Index of cluster-size shares (non-noise points). Rewards
    # broad coverage while penalizing a single dominant "blob" cluster.
    if n_clusters > 0:
        counts = np.bincount(labels[labels >= 0])
        shares = counts / counts.sum()
        hhi    = float((shares ** 2).sum())
    else:
        hhi = 0.0
    eff_coverage_hhi = coverage * (1 - hhi)

    log.info(f"eps={eps} ms={min_samples} → {n_clusters} clusters, {coverage:.1%} coverage, "
             f"{noise_frac:.1%} noise, eff_cov={eff_coverage_hhi:.1%}")

    return dict(cluster_labels=labels.astype(np.int32), n_clusters=n_clusters,
               coverage=coverage, noise_frac=noise_frac,
               eff_coverage_hhi=eff_coverage_hhi, eps=eps, ms_values=min_samples)


def save_results(results: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **results)
    log.info(f"Results saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("embedding", type=Path, help="Path to .npy embedding (shape n × d).")
    p.add_argument("--output-dir", type=Path, default=Path("dbscan_results"), metavar="DIR")
    p.add_argument("--node-ram-gb", type=int, default=256, metavar="GB",
                   help="Node RAM budget (match the Slurm --mem for this job).")
    p.add_argument("--umap-md", type=float, default=0.1, metavar="F",
                   help="UMAP min_dist value (for output labelling only).")
    p.add_argument("--umap-nn", type=int, default=200, metavar="N",
                   help="UMAP n_neighbors value (for output labelling only).")
    p.add_argument("--ens-member", type=int, default=1, metavar="N",
                   help="Ensemble member index of the UMAP embedding.")

    g = p.add_argument_group("DBSCAN parameters")
    g.add_argument("--eps", type=float, default=0.5, metavar="F",
                   help="Neighbourhood radius (eps).")
    g.add_argument("--min-samples", type=int, default=5, metavar="N",
                   help="Minimum samples to form a core point.")
    g.add_argument("--algorithm", default="ball_tree",
                   choices=["ball_tree", "kd_tree", "brute", "auto"],
                   help="DBSCAN search algorithm (default: ball_tree).")

    p.add_argument("--no-normalize", action="store_true",
                   help="Skip MinMaxScaler normalization of the embedding.")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # run_tag  = f"eps{args.eps}_ms{args.min_samples}"
    run_dir  = args.output_dir # / run_tag
    fname    = f"emb_{args.ens_member:02d}th_ensemble_md{args.umap_md}_nn{args.umap_nn}.npz"
    out_path = run_dir / fname
    if out_path.exists():
        log.info(f"Output already exists – skipping DBSCAN.  ({out_path})")
        return 0

    log.info(f"Loading embedding: {args.embedding}")
    embedding = np.load(args.embedding).astype(np.float32)
    if embedding.ndim != 2:
        log.error(f"Expected a 2-D array; got shape {embedding.shape}.")
        return 1
    log.info(f"Embedding shape: {embedding.shape}")

    if not args.no_normalize:
        embedding = MinMaxScaler().fit_transform(embedding)
        log.info("Embedding normalized (MinMaxScaler).")

    log.info(f"DBSCAN: eps={args.eps}  min_samples={args.min_samples}  algorithm={args.algorithm}")

    results = run_dbscan(embedding, eps=args.eps, min_samples=args.min_samples, 
                         node_ram_gb=args.node_ram_gb, algorithm=args.algorithm)
    if results is None:
        return 1

    save_results(results, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

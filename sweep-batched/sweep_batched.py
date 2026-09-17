"""
sweep_batched.py
=======================
Memory-safe DBSCAN parameter sweep on a large 3D embedding (~41M points).

Why this is faster and lighter than naive sweeping:
  * The expensive part of DBSCAN is the neighbour search. For a fixed eps it
    is IDENTICAL for every min_samples value — so we compute it once per eps
    (not once per (eps, ms) pair) as a sparse distance graph, built in CHUNKS
    to cap peak memory, then run DBSCAN(metric="precomputed") for each ms.
  * Labels are EXACTLY what plain DBSCAN(eps, ms) would return.
  * Sweeping eps in ascending order keeps the largest graph for last, and
    each graph is freed before the next eps starts.

If a single eps still OOMs, its neighbour graph is intrinsically too dense
(each point has too many neighbours at that radius) — that eps would OOM in
plain DBSCAN too. The script prints edge counts so you can spot this early
and drop that eps from the sweep.

Usage:
    python sweep_batched.py <embedding.npy> <output_dir>
"""

import sys
import time
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from sklearn.neighbors import NearestNeighbors, sort_graph_by_row_values
from sklearn.cluster import DBSCAN

# ── 1. Config ─────────────────────────────────────────────────────────────────
EPS_VALUES  = list(np.arange(0.035, 0.045, 0.001)) # [0.05, 0.052, 0.053, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08] # list(np.linspace(0.00001, 0.0001, 10))  # [0.01, 0.05, 0.10, 0.20, 0.50]  # ascending: cheapest first
MS_VALUES   = list(np.arange(80, 101, 2))      # [155, 160, 165, 170, 175, 180, 185, 190, 195] # [20, 30, 50, 60, 70, 80, 90, 100, 150] # [10, 50, 100, 200, 500]
CHUNK       = 500_000                         # points per query batch (lower → less peak RAM)
NODE_RAM_GB = 256                             # match the Slurm --mem for this job
BYTES_PER_EDGE = 12                           # float32 distance (4B) + int64 index (8B) —
                                              # scipy forces int64 indices once nnz > 2^31.
MAX_EDGES = (NODE_RAM_GB * 1e9) / (2 * BYTES_PER_EDGE)
                                              # abort an eps if its graph exceeds this.
                                              # ÷2 leaves headroom for sp.vstack(blocks),
                                              # which holds blocks and the merged graph
                                              # at once — a transient ~2× memory peak.

if len(sys.argv) != 3:
    sys.exit(f"Usage: python {Path(__file__).name} <embedding.npy> <output_dir>")

emb_path   = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
output_dir.mkdir(parents=True, exist_ok=True)

X = np.load(emb_path).astype(np.float32)
n = len(X)
print(f"Embedding: {emb_path.name}  shape={X.shape}", flush=True)

# ── 2. Build the BallTree ONCE (reused for every eps) ─────────────────────────
t0 = time.time()
nn = NearestNeighbors(algorithm="ball_tree", leaf_size=30, n_jobs=-1).fit(X)
print(f"BallTree built in {time.time()-t0:.0f}s", flush=True)

# ── 3. Sweep ──────────────────────────────────────────────────────────────────
n_clusters       = np.zeros((len(EPS_VALUES), len(MS_VALUES)), dtype=int)
coverage         = np.full((len(EPS_VALUES), len(MS_VALUES)), np.nan, dtype=float)
eff_coverage_hhi = np.full((len(EPS_VALUES), len(MS_VALUES)), np.nan, dtype=float)

for i, eps in enumerate(EPS_VALUES):

    # 3a. Chunked radius query at THIS eps → sparse distance graph.
    #     Peak memory during construction is bounded by CHUNK.
    t0, blocks, edges, aborted = time.time(), [], 0, False
    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        g = nn.radius_neighbors_graph(X[start:stop], radius=eps,
                                      mode="distance", sort_results=False)
        g.data = g.data.astype(np.float32, copy=False)  # BallTree returns float64 regardless of X's dtype
        edges += g.nnz
        if edges > MAX_EDGES:                      # graph too dense → skip eps
            print(f"eps={eps}: >{MAX_EDGES:,} edges — too dense, skipping",
                  flush=True)
            aborted = True
            break
        blocks.append(g)
    if aborted:
        del blocks
        continue

    G = sp.vstack(blocks).tocsr()
    del blocks
    G.data[G.data == 0] = 1e-12                    # keep true zero-distance edges
    sort_graph_by_row_values(G, warn_when_not_sorted=False)  # DBSCAN prefers sorted rows
    print(f"eps={eps}: graph {G.nnz:,} edges, {G.data.nbytes/1e9:.1f} GB "
          f"({time.time()-t0:.0f}s)", flush=True)

    # 3b. All min_samples values reuse the SAME graph — nearly free.
    for j, ms in enumerate(MS_VALUES):
        t1     = time.time()
        labels = DBSCAN(eps=eps, min_samples=ms,
                        metric="precomputed", n_jobs=-1).fit_predict(G)
        k, noise = int(labels.max()) + 1, float((labels == -1).mean())
        n_clusters[i, j] = k
        coverage[i, j]   = 1 - noise

        # Balance-penalized effective coverage: coverage × (1 − HHI), where HHI is
        # the Herfindahl-Hirschman Index of cluster-size shares (non-noise points).
        # Rewards broad coverage while penalizing a single dominant "blob" cluster.
        if k > 0:
            counts = np.bincount(labels[labels >= 0])
            shares = counts / counts.sum()
            hhi    = float((shares ** 2).sum())
        else:
            hhi = 0.0
        eff_coverage_hhi[i, j] = coverage[i, j] * (1 - hhi)

        path = output_dir / f"sweep_eps{eps}_ms{ms}.npz"
        np.savez_compressed(path, labels=labels.astype(np.int32),
                            n_clusters=k, coverage=coverage[i, j], noise_frac=noise,
                            eff_coverage_hhi=eff_coverage_hhi[i, j])
        print(f"  eps={eps:<5} ms={ms:<4} → {k:>4} clusters, {coverage[i, j]:6.1%} coverage, "
              f"{noise:5.1%} noise, eff_cov={eff_coverage_hhi[i, j]:6.1%}  "
              f"({time.time()-t1:.0f}s)", flush=True)

    del G                                          # free before next eps

# ── 4. Summary ────────────────────────────────────────────────────────────────
print("\n(eps, ms)        clusters   noise    eff_cov")
for i, eps in enumerate(EPS_VALUES):
    for j, ms in enumerate(MS_VALUES):
        if np.isnan(coverage[i, j]):
            continue
        print(f"({eps:<5}, {ms:<4})   {n_clusters[i, j]:>6}   {1 - coverage[i, j]:6.1%}   "
              f"{eff_coverage_hhi[i, j]:6.1%}")

summary_path = output_dir / f"{emb_path.stem}_sweep_results.npz"
np.savez_compressed(summary_path, eps_values=EPS_VALUES, ms_values=MS_VALUES,
                    n_clusters=n_clusters, coverage=coverage,
                    eff_coverage_hhi=eff_coverage_hhi)

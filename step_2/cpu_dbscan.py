"""Run DBSCAN on a UMAP embedding using scikit-learn (CPU)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_BACKEND = f"sklearn {sklearn.__version__} (CPU)"


# ---------------------------------------------------------------------------
# Core fit
# ---------------------------------------------------------------------------

def run_dbscan(embedding: np.ndarray, eps: float, min_samples: int,
               algorithm: str) -> dict:
    """Run sklearn DBSCAN and return cluster-quality metrics."""
    labels = DBSCAN(eps=float(eps), min_samples=int(min_samples),
                    algorithm=algorithm).fit_predict(embedding)

    assigned = labels != -1
    n_points = len(labels)
    unique_clusters = np.unique(labels[assigned])
    n_clusters = len(unique_clusters)
    coverage = assigned.sum() / n_points

    if n_clusters > 0:
        counts = np.array([np.sum(labels == k) for k in unique_clusters])
        dominant_frac = counts.max() / n_points
        mean_median_ratio = counts.mean() / np.median(counts)
        shares = counts / counts.sum()
        eff_coverage_hhi = coverage * (1.0 - (shares ** 2).sum())
        cov_excl_largest = (assigned.sum() - counts.max()) / n_points
    else:
        dominant_frac = 0.0
        mean_median_ratio = np.nan
        eff_coverage_hhi = 0.0
        cov_excl_largest = 0.0

    log.info(
        f"  → {n_clusters} clusters  coverage={coverage*100:.1f}%  "
        f"dominant={dominant_frac*100:.1f}%  eff_cov_hhi={eff_coverage_hhi*100:.1f}%"
    )
    return dict(cluster_labels=labels, n_clusters=n_clusters, coverage=coverage,
                dominant_frac=dominant_frac, mean_median_ratio=mean_median_ratio,
                eff_coverage_hhi=eff_coverage_hhi, cov_excl_largest=cov_excl_largest)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = ("cluster_labels", "n_clusters", "coverage", "dominant_frac",
                  "mean_median_ratio", "eff_coverage_hhi", "cov_excl_largest")


def save_results(results: dict, path: Path,
                 eps: float | None = None, ms_value: int | None = None) -> None:
    missing = [k for k in _REQUIRED_KEYS if k not in results]
    if missing:
        raise KeyError(f"Results dict missing required keys: {missing}")

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: results[k] for k in _REQUIRED_KEYS}
    payload["backend"] = np.array(_BACKEND)
    if eps is not None:
        payload["eps"] = float(eps)
    if ms_value is not None:
        payload["ms_value"] = int(ms_value)

    np.savez_compressed(path, **payload)
    log.info(f"DBSCAN results saved → {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("embedding", type=Path,
                   help="Path to .npy UMAP embedding (shape n × d).")
    p.add_argument("--output-dir", type=Path, default=Path("dbscan_results"), metavar="DIR")
    p.add_argument("--umap-md", type=float, default=0.1, metavar="F",
                   help="UMAP min_dist value (for output labelling only).")
    p.add_argument("--umap-nn", type=int, default=200, metavar="N",
                   help="UMAP n_neighbors value (for output labelling only).")
    p.add_argument("--ens-member", type=int, default=1, metavar="N",
                   help="Ensemble member index of UMAP embedding.")

    g = p.add_argument_group("DBSCAN parameters")
    g.add_argument("--eps", type=float, default=0.5, metavar="F",
                   help="Neighbourhood radius (eps).")
    g.add_argument("--min-samples", type=int, default=5, metavar="N",
                   help="Minimum samples to form a core point.")
    g.add_argument("--algorithm", default="ball_tree",
                   choices=["ball_tree", "kd_tree", "brute", "auto"],
                   help="DBSCAN search algorithm (default: ball_tree).")

    p.add_argument("--no-normalize", action="store_true",
                   help="Skip StandardScaler / MinMaxScaler normalization.")
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run_dir  = args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fname    = f"emb_{args.ens_member:02d}th_ensemble_md{args.umap_md}_nn{args.umap_nn}.npz"
    out_path = run_dir / fname
    if out_path.exists():
        log.info(f"Output already exists – skipping DBSCAN.  ({out_path})")
        return 0

    log.info(f"Loading embedding: {args.embedding}")
    embedding = np.load(args.embedding)
    if embedding.ndim != 2:
        log.error(f"Expected a 2-D array; got shape {embedding.shape}.")
        return 1
    log.info(f"Embedding shape: {embedding.shape}")

    if not args.no_normalize:
        # embedding = StandardScaler().fit_transform(embedding)
        embedding = MinMaxScaler().fit_transform(embedding)
        log.info("Embedding normalized (StandardScaler or MinMaxScaler).")

    log.info(f"Backend: {_BACKEND}")
    log.info(f"DBSCAN: eps={args.eps}  min_samples={args.min_samples}  algorithm={args.algorithm}")

    results = run_dbscan(embedding.astype(np.float32),
                         eps=args.eps, min_samples=args.min_samples,
                         algorithm=args.algorithm)

    save_results(results, out_path, eps=args.eps, ms_value=args.min_samples)
    return 0


if __name__ == "__main__":
    sys.exit(main())

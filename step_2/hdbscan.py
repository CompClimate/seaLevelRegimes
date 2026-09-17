"""Run a single HDBSCAN fit on a UMAP embedding and save the result."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------

def _cuml_memory_ok(n_points: int, min_samples: int) -> bool:
    """Preflight check: estimate cuML HDBSCAN peak memory vs free GPU memory.

    cuML HDBSCAN aborts with a C++ std::terminate (uncatchable from Python)
    when it runs out of GPU memory. This check is intentionally conservative
    so we fall back to a CPU backend before that can happen.
    """
    try:
        import cupy as cp
        free_bytes, total_bytes = cp.cuda.Device().mem_info
        # Dominant cost: k-NN graph storage (n × k floats + indices) plus
        # intermediate buffers, tree structures, and cuML internals.
        # Empirical safety factor of 20× gives a conservative upper bound.
        k = max(min_samples, 16)
        estimated_peak = n_points * k * 4 * 20
        log.info(
            f"GPU memory preflight: ~{estimated_peak / 1024**3:.1f} GB estimated  "
            f"(free={free_bytes / 1024**3:.1f} GB, total={total_bytes / 1024**3:.1f} GB)"
        )
        if estimated_peak > free_bytes:
            log.warning(
                f"Estimated GPU peak ({estimated_peak / 1024**3:.1f} GB) exceeds "
                f"free GPU memory ({free_bytes / 1024**3:.1f} GB) – falling back to CPU."
            )
            return False
        return True
    except Exception as e:
        log.warning(f"GPU memory preflight failed ({e}); proceeding optimistically.")
        return True  # cannot check — proceed and let cuML decide


def _has_gpu() -> bool:
    """Return True if at least one NVIDIA GPU is accessible via nvidia-smi."""
    try:
        subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError,
            subprocess.TimeoutExpired):
        return False


def get_hdbscan_class(force_cpu: bool = False) -> tuple:
    """
    Return (HDBSCANClass, backend_label, has_relative_validity).

    Priority order:
      1. RAPIDS cuML  (GPU, fast – no relative_validity_)
      2. hdbscan package  (relative_validity_ available)
      3. scikit-learn >= 1.3  (relative_validity_ available)
    """
    if not force_cpu and _has_gpu():
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning,
                                        message=r".*cuda\.(cudart|cuda).*")
                from cuml.cluster import HDBSCAN as cuHDBSCAN
            log.info("RAPIDS cuML detected – using GPU-accelerated HDBSCAN.")
            return cuHDBSCAN, "cuML (GPU)", False
        except ImportError:
            log.warning("cuML not installed – falling back to CPU HDBSCAN.")
        except Exception as e:
            log.warning(f"cuML import failed ({type(e).__name__}: {e}) – "
                         "falling back to CPU HDBSCAN.")

    try:
        from hdbscan import HDBSCAN as hdbHDBSCAN
        log.info("Using hdbscan package (relative_validity_ available).")
        return hdbHDBSCAN, "hdbscan pkg (CPU)", True
    except ImportError:
        pass

    try:
        import sklearn
        sk_ver = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
        if sk_ver >= (1, 3):
            from sklearn.cluster import HDBSCAN as skHDBSCAN
            log.info(f"Using scikit-learn {sklearn.__version__} HDBSCAN "
                     f"(relative_validity_ available).")
            return skHDBSCAN, f"sklearn {sklearn.__version__} (CPU)", True
        log.warning(f"scikit-learn {sklearn.__version__} < 1.3 – "
                    "no HDBSCAN support; trying hdbscan package.")
    except ImportError:
        pass

    log.error("No HDBSCAN backend found.  "
              "Install scikit-learn >= 1.3 or pip install hdbscan.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

def run_hdbscan(embedding: np.ndarray, 
                mcs_value: int,
                ms_value: int,
                HDBSCANClass,
                has_relative_validity: bool,
                cluster_selection_method: str = "eom",
                cluster_selection_epsilon: float = 0.0) -> dict:
    """
    Run HDBSCAN for (min_cluster_size, min_samples) combination.

    Parameters
    ----------
    embedding                : (n, d) float32 array – the UMAP embedding.
    mcs_value               : min_cluster_size value.
    ms_value               : min_samples value.
    HDBSCANClass             : backend HDBSCAN class.
    has_relative_validity    : whether the backend exposes relative_validity_.
    cluster_selection_method : 'eom' or 'leaf'.
    cluster_selection_epsilon: post-clustering merge threshold (0 = off).

    Returns
    -------
    dict with keys:
        n_clusters          : (n_mcs, n_ms) int
        coverage            : (n_mcs, n_ms) float – fraction of non-noise pts.
        dominant_frac       : (n_mcs, n_ms) float – fraction in largest cluster.
        mean_median_ratio   : (n_mcs, n_ms) float – mean/median cluster size.
        eff_coverage_hhi    : (n_mcs, n_ms) float – coverage × (1 − HHI).
        cov_excl_largest    : (n_mcs, n_ms) float – coverage outside dominant cluster.
        relative_validity   : (n_mcs, n_ms) float – DBCV approx (NaN if unavailable).
        composite_score     : (n_mcs, n_ms) float – eff_coverage_hhi × max(0, rel_val);
                              falls back to eff_coverage_hhi when rel_val is NaN.
        labels_grid         : (n_mcs, n_ms) object array – per-point label arrays.
    """
    model = HDBSCANClass(
        min_cluster_size          = int(mcs_value),
        min_samples               = int(ms_value),
        cluster_selection_method  = cluster_selection_method,
        cluster_selection_epsilon = float(cluster_selection_epsilon),
    )
    raw    = model.fit_predict(embedding)
    labels = (raw.get() if hasattr(raw, "get") else np.asarray(raw)).astype(int)

    assigned        = labels != -1
    unique_clusters = np.unique(labels[assigned])
    n_clusters              = len(unique_clusters)

    n_points = embedding.shape[0]
    coverage    = assigned.sum() / n_points

    if n_clusters > 0:
        counts = np.array([np.sum(labels == k) for k in unique_clusters])
        dominant_frac     = counts.max() / n_points
        mean_median_ratio = counts.mean() / np.median(counts)
        shares            = counts / counts.sum()
        hhi               = (shares ** 2).sum()
        eff_coverage_hhi  = coverage * (1.0 - hhi)
        cov_excl_largest  = (assigned.sum() - counts.max()) / n_points
    else:
        dominant_frac     = 0.0
        mean_median_ratio = np.nan
        eff_coverage_hhi  = 0.0
        cov_excl_largest  = 0.0

    # DBCV approximation — free from sklearn / hdbscan backends
    rv = np.nan
    if has_relative_validity:
        try:
            rv = float(model.relative_validity_)
        except AttributeError:
            pass
    relative_validity = rv

    # Composite: scale eff_coverage_hhi by cluster validity when available
    if not np.isnan(rv):
        composite_score = eff_coverage_hhi * max(0.0, rv)
    else:
        composite_score = eff_coverage_hhi

    rv_str = f"{rv:.3f}" if not np.isnan(rv) else "n/a"
    log.info(f"           → {n_clusters} clusters  "
             f"coverage={coverage*100:.1f}%  "
             f"dominant={dominant_frac*100:.1f}%  "
             f"eff_cov={eff_coverage_hhi*100:.1f}%  "
             f"rel_val={rv_str}  "
             f"composite={composite_score:.4f}")

    hdbscan_outputs = dict(cluster_labels    = labels,
                           n_clusters        = n_clusters,
                           coverage          = coverage,
                           dominant_frac     = dominant_frac,
                           mean_median_ratio = mean_median_ratio,
                           eff_coverage_hhi  = eff_coverage_hhi,
                           cov_excl_largest  = cov_excl_largest,
                           relative_validity = relative_validity,
                           composite_score   = composite_score)
    return hdbscan_outputs


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_results(results: dict, backend_label: str, path: Path,
                 mcs_value: int | None = None,
                 ms_value:  int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    required = ("cluster_labels", "n_clusters", "coverage", "dominant_frac", "mean_median_ratio",
                "eff_coverage_hhi", "cov_excl_largest", "relative_validity", "composite_score")
    missing = [k for k in required if k not in results]
    if missing:
        raise KeyError(f"Results dict is missing required keys: {missing}")

    payload = {k: results[k] for k in required}
    payload["backend_label"] = np.array(backend_label)
    if mcs_value is not None:
        payload["mcs_value"] = mcs_value
    if ms_value is not None:
        payload["ms_value"]  = ms_value

    np.savez_compressed(path, **payload)
    log.info(f"HDBSCAN clustering results saved → {path}")



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input / output ───────────────────────────────────────────────────
    p.add_argument("embedding", type=Path,
                   help="Path to .npy UMAP embedding (shape n × d).")
    p.add_argument("--output-dir", type=Path, default=Path("hdbscan_sweep"), metavar="DIR",
                   help="Output directory for all figures and result files.")
    p.add_argument("--umap-md", type=float, default=0.1, metavar="F",
                   help="UMAP min_dist value (for labelling outputs; does not affect the sweep).")
    p.add_argument("--umap-nn", type=int, default=200, metavar="N",
                   help="UMAP n_neighbors value (for labelling outputs; does not affect the sweep).")
    p.add_argument("--ens-member",  type=int, default=1,  metavar="N",
                   help="Ensemble member index of UMAP embedding.")

    # ── Grid parameters ────────────────────────────────────────────────────
    g = p.add_argument_group("HDBSCAN grid")
    g.add_argument("--min-cluster",  type=int, default=20,  metavar="N",
                   help="Value of the min_cluster_size parameter.")

    g.add_argument("--min-samples",  type=int, default=5,  metavar="N",
                   help="Value of the min_samples parameter.")

    g.add_argument("--cluster-selection-method", choices=["eom", "leaf"], default="eom",
                   help="'eom' (default, larger clusters) or 'leaf' (more, smaller clusters).")
    g.add_argument("--cluster-selection-epsilon", type=float, default=0.0, metavar="F",
                   help="Post-clustering merge distance threshold (0.0 = disabled).")

    # ── Preprocessing ──────────────────────────────────────────────────────
    p.add_argument("--no-normalize", action="store_true",
                   help="Skip StandardScaler normalization (on by default).")

    # ── Backend ────────────────────────────────────────────────────────────
    p.add_argument("--cpu", action="store_true",
                   help="Force CPU backend even when cuML is available.")

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # ── Model-specific output subdirectory ────────────────────────────────
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Outputs → {output_dir}")

    # ── Load embedding ─────────────────────────────────────────────────────
    log.info(f"Loading embedding: {args.embedding}")
    embedding = np.load(args.embedding)
    if embedding.ndim != 2:
        log.error(f"Expected a 2-D array; got shape {embedding.shape}.")
        return 1
    
    log.info(f"Embedding shape: {embedding.shape}")

    # ── Normalize ─────────────────────────────────────────────────────────
    if not args.no_normalize:
        embedding = StandardScaler().fit_transform(embedding)
        log.info("Embedding normalized with StandardScaler (use --no-normalize to skip).")
    else:
        log.info("Normalization skipped (--no-normalize).")

    # ── Build parameter grids ──────────────────────────────────────────────
    mcs_value = int(args.min_cluster)
    ms_value = int(args.min_samples)

    log.info(f"The min_cluster_size value: {mcs_value}")
    log.info(f"The min_samples value:  {ms_value}")
    log.info(f"Total HDBSCAN fits: 1")

    run_tag = (f"mcs{args.min_cluster}_ms{args.min_samples}"
               f"_{args.cluster_selection_method}")
    output_dir = Path(output_dir / run_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Run tag: {run_tag}")

    # ── Run HDBSCAN clustering ───────────────────────────────────────
    HDBSCANClass, backend_label, has_rv = get_hdbscan_class(force_cpu=args.cpu)

    if "cuML" in backend_label:
        if not _cuml_memory_ok(embedding.shape[0], ms_value):
            log.info("Switching to CPU backend due to GPU memory constraints.")
            HDBSCANClass, backend_label, has_rv = get_hdbscan_class(force_cpu=True)
        else:
            # cuML requires float32; transfer to GPU once to avoid per-fit CPU→GPU copies.
            embedding = embedding.astype(np.float32)
            try:
                import cupy as cp
                embedding = cp.asarray(embedding)
                log.info(f"Embedding transferred to GPU memory "
                         f"({embedding.nbytes / 1024**2:.1f} MB).")
            except Exception as e:
                log.warning(f"Could not move embedding to GPU ({type(e).__name__}: {e}); using CPU array.")

    log.info(f"Clustering the embedding with {backend_label} – "
             f"min_cluster_size={int(mcs_value)}  min_samples={int(ms_value)}")

    results = run_hdbscan(embedding, mcs_value, ms_value, HDBSCANClass,
                          has_relative_validity        = has_rv,
                          cluster_selection_method     = args.cluster_selection_method,
                          cluster_selection_epsilon    = args.cluster_selection_epsilon,)

    # ── Save HDBSCAN clustering results ───────────────────────────────────────
    run_tag_file = f"emb_{args.ens_member:02d}th_ensemble_md{args.umap_md:.1f}_nn{args.umap_nn}.npz"
    save_results(results, backend_label, output_dir/f"{run_tag_file}", mcs_value=mcs_value, ms_value=ms_value)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())



r"""
Majority vote over a DBSCAN ensemble, with a spatial mode filter applied to
each member *before* Hungarian alignment and voting.

    load members:
    ->  [cap to top-K]  ->  align + vote           (BEFORE)
                        ->  smooth -> align + vote  (AFTER)

Both vote maps are written to the same .npz so the effect of smoothing can be
compared directly.  Noise (-1) is excluded from the spatial filter and from the
ensemble vote in both branches.

Labels are 1-D over the ocean points that survive the complete-case mask of the
BVB predictors; load_complete_mask()/grid_labels() back-project them onto the
source grid, which is what the spatial filter needs.
"""

from __future__ import annotations

import sys
import argparse
import logging
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# BVB predictors defining the complete-case mask (same set used to build the
# clustering input).
BVB_TERMS = ["beta_V", "BPT", "Mass_flux", "eta_dt",
             "Curl_dudt", "Curl_taus", "Curl_taub", "Curl_Adv", "Curl_diff"]

# Off-grid sentinel used while filtering: cells outside the complete-case mask.
# Distinct from noise (-1) so the filter can tell "no data" from "clustered as
# noise".  Both are negative, and only non-negative labels ever vote.
LAND = -2


#  Utilities    

def noise_fraction(labels: np.ndarray) -> float:
    """Fraction of points classified as noise (-1)."""
    return float((labels == -1).sum() / len(labels)) if len(labels) > 0 else 1.0


def _build_cost_matrix(
    ref_labels: np.ndarray,    # (n_ocean_pts,) — reference member
    mem_labels: np.ndarray,    # (n_ocean_pts,) — current member
    ref_clusters: np.ndarray,  # sorted unique non-noise labels in reference
    mem_clusters: np.ndarray,  # sorted unique non-noise labels in member
) -> np.ndarray:
    """
    Build a (n_ref_clusters × n_mem_clusters) overlap matrix.

    Entry [i, j] = number of grid cells where reference assigns ref_clusters[i]
    AND member assigns mem_clusters[j].  Cells where either side is noise (-1)
    are excluded.  scipy.optimize.linear_sum_assignment handles rectangular
    matrices natively.
    """
    both_clustered = (ref_labels != -1) & (mem_labels != -1)
    ref_c = ref_labels[both_clustered]
    mem_c = mem_labels[both_clustered]

    n_ref, n_mem = len(ref_clusters), len(mem_clusters)
    if ref_c.size == 0:
        return np.zeros((n_ref, n_mem), dtype=np.int64)

    # ref_clusters/mem_clusters are sorted unique arrays (from np.unique) and
    # ref_c/mem_c are guaranteed subsets, so searchsorted gives exact indices.
    ref_idx = np.searchsorted(ref_clusters, ref_c)
    mem_idx = np.searchsorted(mem_clusters, mem_c)

    flat = ref_idx * n_mem + mem_idx
    cost = np.bincount(flat, minlength=n_ref * n_mem).reshape(n_ref, n_mem)
    return cost.astype(np.int64)


def _remap_labels(labels: np.ndarray, label_map: dict[int, int]) -> np.ndarray:
    """
    Vectorized equivalent of ``np.vectorize(label_map.__getitem__)(labels)``.

    np.vectorize is a thin wrapper around a Python-level loop, so it does not
    scale to large point counts. This builds a small lookup table (indexed by
    the label's offset from the minimum key) and remaps via fancy indexing,
    which runs entirely in C.
    """
    keys = np.fromiter(label_map.keys(), dtype=np.int64)
    vals = np.fromiter(label_map.values(), dtype=np.int64)
    lo   = int(keys.min())

    lut = np.empty(int(keys.max()) - lo + 1, dtype=np.int32)
    lut[keys - lo] = vals
    return lut[labels - lo]


#  Geographic back-projection 

def load_complete_mask(grid_zarr: Path):
    """
    Load the complete-case mask (no NaN in any predictor) from the BVB grid.

    This is the mask that selected the ocean points in the first place, so a
    label vector is exactly its flattening, in order.

    Args:
        grid_zarr (Path): Path to the BVB Dataset with the original grid,
            dims (lat, lon) or (time, lat, lon).

    Returns:
        xarray.DataArray: boolean mask carrying the grid's dims and coords.
    """
    import xarray as xr  # imported lazily: only needed for gridded work

    bvb_ds = xr.open_zarr(grid_zarr, chunks=None)

    ref_da        = bvb_ds[BVB_TERMS[0]]
    complete_mask = xr.full_like(ref_da, True, dtype=bool)
    for var in BVB_TERMS:
        complete_mask = complete_mask & bvb_ds[var].notnull()

    return complete_mask


def grid_labels(labels: np.ndarray, complete_mask):
    """
    Back-project labels onto the grid, preserving the original NaN pattern.

    Args:
        labels (numpy.ndarray): 1-D array of embedded data or labels.
        complete_mask (xarray.DataArray): mask from load_complete_mask().

    Returns:
        xarray.DataArray: gridded labels, NaN off the mask.
    """
    import xarray as xr

    reconstructed = np.full(complete_mask.shape, np.nan)
    reconstructed[complete_mask.values] = labels

    return xr.DataArray(reconstructed,
                        dims=complete_mask.dims,
                        coords=complete_mask.coords,
                        name="geo_projected")


def load_and_grid_labels(grid_zarr: Path, labels: np.ndarray):
    """Back-project labels straight from a grid file (convenience wrapper)."""
    return grid_labels(labels, load_complete_mask(grid_zarr))


#  Spatial mode filter 

def _sorted_runs(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Sort along axis 0 and return the run length ending at each position.

    Counting how often each label occurs along an axis is the one operation
    both the spatial filter and the ensemble vote need, and doing it by
    materialising one plane per label costs O(n_labels) memory — untenable when
    a member carries thousands of clusters.  Sorting instead groups equal
    labels into contiguous runs, so every label's count is a run length and the
    cost depends only on the length of axis 0.

    Negative entries (noise and off-grid) never vote: a sentinel pushes them
    above every real label and their runs are zeroed.

    Returns
    -------
    srt : sorted copy of ``stack``, invalid entries replaced by the sentinel
    run : run length of equal labels ending at each position; 0 at sentinels.
          Within a run this increases monotonically, so a run's total count
          sits at its last position.
    """
    sentinel = np.iinfo(np.int32).max

    srt = np.where(stack < 0, sentinel, stack).astype(np.int32, copy=False)
    srt.sort(axis=0)

    run = np.ones(srt.shape, dtype=np.int32)
    for i in range(1, srt.shape[0]):
        np.add(run[i - 1], 1, out=run[i], where=(srt[i] == srt[i - 1]))
    run[srt == sentinel] = 0

    return srt, run


def _mode_from_runs(srt: np.ndarray, run: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Modal label and its count, from the output of _sorted_runs().

    argmax takes the first maximum, and within a run the count peaks at the
    run's end, so ties resolve to the smallest label — deterministic and
    independent of the input ordering.
    """
    best       = run.argmax(axis=0)
    best_count = np.take_along_axis(run, best[None], axis=0)[0]
    best_label = np.take_along_axis(srt, best[None], axis=0)[0]
    return best_label, best_count


def _mode_filter_slice(sl: np.ndarray, radius: int, fill_noise: bool) -> np.ndarray:
    """
    One pass of the mode filter over a single (ny, nx) label slice.

    Rather than convolving a one-hot plane per cluster — one pass over the grid
    per label, and these runs carry thousands of labels — the neighbourhood is
    gathered by shifting the slice once per kernel offset.  That is the same
    neighbourhood a convolution sums over, but its cost depends on the kernel
    size (k*k shifts) instead of on the number of clusters.

    Ties go to the cell's own label, which keeps the filter conservative.
    """
    ny, nx = sl.shape
    k      = 2 * radius + 1

    # Pad with LAND in y, and wrap in x since longitude is periodic.  Rows past
    # the poles are simply treated as no-data.
    padded = np.full((ny + 2 * radius, nx + 2 * radius), LAND, dtype=np.int32)
    padded[radius:radius + ny, radius:radius + nx] = sl
    padded[radius:radius + ny, :radius]           = sl[:, nx - radius:]
    padded[radius:radius + ny, radius + nx:]      = sl[:, :radius]

    # stack[i] is the slice shifted by one kernel offset.
    stack = np.empty((k * k, ny, nx), dtype=np.int32)
    for i, (dy, dx) in enumerate((dy, dx) for dy in range(k) for dx in range(k)):
        stack[i] = padded[dy:dy + ny, dx:dx + nx]

    own_count = (stack == sl[None, :, :]).sum(axis=0).astype(np.int32)
    own_count[sl < 0] = 0

    best_label, best_count = _mode_from_runs(*_sorted_runs(stack))

    out = sl.copy()
    # A clustered cell adopts the neighbourhood mode unless its own label
    # already ties for it.
    replace = (sl >= 0) & (best_count > 0) & (own_count < best_count)
    out[replace] = best_label[replace]

    if fill_noise:
        # Optional: let a noise cell surrounded by a clear cluster join it.
        fill = (sl == -1) & (best_count > 0)
        out[fill] = best_label[fill]

    return out


def smooth_member(
    labels: np.ndarray,
    complete_mask,
    kernel_size: int = 3,
    fill_noise: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Smooth one member by replacing each cell with its neighbourhood mode.

    The labels are back-projected onto the grid, filtered, and gathered back
    into the same 1-D ocean-point ordering.  When the grid has a leading time
    axis each 2-D (lat, lon) slice is filtered independently: the neighbourhood
    is spatial, not temporal.

    Noise is excluded — it casts no vote and keeps its own label — so it is
    carried through the filter unchanged and re-enters the ensemble vote
    intact.  Set fill_noise to let noise cells take the neighbourhood mode too.

    Returns
    -------
    smoothed : (n_ocean_pts,) int32
    changed  : fraction of ocean points whose label changed
    """
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd and >= 3, got {kernel_size}")

    mask = complete_mask.values
    grid = np.full(mask.shape, LAND, dtype=np.int32)
    grid[mask] = labels

    # Treat (lat, lon) and (time, lat, lon) alike: filter every 2-D slice.
    slices = grid.reshape((-1,) + grid.shape[-2:])
    for t in range(slices.shape[0]):
        slices[t] = _mode_filter_slice(slices[t], kernel_size // 2, fill_noise)

    smoothed = grid[mask].astype(np.int32)
    changed  = float((smoothed != labels).mean()) if labels.size else 0.0
    return smoothed, changed


#  Cluster capping (memory-safety post-processing) 

def cap_to_top_k_clusters(labels: np.ndarray, k: int) -> np.ndarray:
    """
    Retain only the ``k`` largest clusters, relabelled 0..k-1 by descending
    size, and merge every other non-noise cluster into a single overflow
    cluster labelled ``k``. Noise (-1) is left untouched.

    The Hungarian cost matrix in _build_cost_matrix is
    (n_ref_clusters x n_mem_clusters); with very large ensembles this can
    blow up memory. Capping the per-member cluster count before alignment
    bounds that matrix regardless of how many raw clusters a member has.
    """
    non_noise = labels != -1
    if not non_noise.any():
        return labels.copy()

    uniq, counts = np.unique(labels[non_noise], return_counts=True)
    if uniq.size <= k:
        return labels.copy()

    top_order  = np.argsort(-counts)[:k]  # indices of k largest clusters
    top_labels = uniq[top_order]

    label_map = {-1: -1}
    for new_lbl, orig_lbl in enumerate(top_labels):
        label_map[int(orig_lbl)] = new_lbl
    for orig_lbl in uniq:
        if int(orig_lbl) not in label_map:
            label_map[int(orig_lbl)] = k

    return _remap_labels(labels, label_map)


#  Core alignment 

def align_ensemble(
    labels_ensemble: list[np.ndarray],
    reference: int | str = "min_noise",
) -> tuple[np.ndarray, int, dict[int, dict]]:
    """
    Align DBSCAN labels across ensemble members with the Hungarian algorithm.

    All arrays in labels_ensemble must have the same length (n_ocean_pts).
    The number of unique cluster labels may differ freely across members.

    Parameters
    ----------
    labels_ensemble : list of (n_ocean_pts,) int np.ndarray
        One DBSCAN label array per ensemble member.  Convention: -1 = noise,
        0, 1, 2, ... = clusters.
    reference : int or "min_noise"
        Index of the reference member, or "min_noise" to auto-select the
        member with the fewest noise points.

    Returns
    -------
    aligned : (n_members, n_ocean_pts) int32 array
    ref_idx : int
    alignment_maps : dict[member_idx -> dict[orig_label -> aligned_label]]
    """
    n_members = len(labels_ensemble)
    n_ocean   = labels_ensemble[0].shape[0]

    assert all(lbl.shape == (n_ocean,) for lbl in labels_ensemble), \
        "All label arrays must have the same length (n_ocean_pts)."

    if reference == "min_noise":
        noise_fracs = [noise_fraction(lbl) for lbl in labels_ensemble]
        ref_idx     = int(np.argmin(noise_fracs))
    elif isinstance(reference, int):
        ref_idx = reference
    else:
        raise ValueError(f"reference must be an int or 'min_noise', got {reference!r}")

    ref_labels   = labels_ensemble[ref_idx]
    ref_clusters = np.unique(ref_labels[ref_labels != -1])
    offset_base  = int(ref_clusters.max()) + 1 if len(ref_clusters) > 0 else 0

    log.info(
        "Reference: member %d  |  noise frac = %.3f  |  n_clusters = %d",
        ref_idx + 1, noise_fraction(ref_labels), len(ref_clusters),
    )

    aligned        = np.empty((n_members, n_ocean), dtype=np.int32)
    alignment_maps = {}

    for m, mem_labels in enumerate(labels_ensemble):
        mem_clusters = np.unique(mem_labels[mem_labels != -1])
        label_map    = {-1: -1}

        if m == ref_idx:
            label_map.update({int(c): int(c) for c in ref_clusters})

        elif len(mem_clusters) == 0:
            pass  # fully noisy member — identity map ({-1: -1} only)

        else:
            # Rectangular cost matrix: (n_ref_clusters × n_mem_clusters).
            # linear_sum_assignment maximises overlap; unmatched member
            # clusters (Case B: more member clusters than reference) receive
            # offset labels above the reference label space.
            cost             = _build_cost_matrix(ref_labels, mem_labels,
                                                  ref_clusters, mem_clusters)
            row_ind, col_ind = linear_sum_assignment(-cost)

            for row, col in zip(row_ind, col_ind):
                label_map[int(mem_clusters[col])] = int(ref_clusters[row])

            next_offset = offset_base
            for mc in mem_clusters:
                if int(mc) not in label_map:
                    label_map[int(mc)] = next_offset
                    next_offset       += 1

            n_ref_k     = len(ref_clusters)
            n_mem_k     = len(mem_clusters)
            n_matched   = len(row_ind)
            n_unmatched = n_mem_k - n_matched
            log.info(
                "  Member %3d  |  ref clusters = %3d  |  mem clusters = %3d  |"
                "  matched = %3d  |  unmatched = %2d  |  noise frac = %.3f",
                m + 1, n_ref_k, n_mem_k, n_matched, n_unmatched,
                noise_fraction(mem_labels),
            )

        alignment_maps[m] = label_map
        aligned[m] = _remap_labels(mem_labels, label_map)

    return aligned, ref_idx, alignment_maps


# ── Post-alignment: majority vote + entropy ────────────────────────────────────

def majority_vote_and_entropy(
    aligned: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Majority-vote label map and per-cell Shannon entropy.

    Noise (-1) is excluded from both the vote and the entropy:

    * Vote: for each cell, the plurality cluster among non-noise member
      assignments wins.  A cell is labeled noise only when every member
      assigned noise to it.

    * Entropy: Shannon entropy is computed over cluster-only votes,
      normalized by the number of non-noise votes for that cell.  Cells
      that are noise in all members have entropy = 0.

    Parameters
    ----------
    aligned : (n_members, n_ocean_pts) int32 — output of align_ensemble()

    Returns
    -------
    vote_map    : (n_ocean_pts,) int32   — majority-vote label per cell
    entropy_map : (n_ocean_pts,) float32 — Shannon entropy H per cell
    noise_frac  : float  — fraction of cells whose majority vote is noise
    member_noise_frac : (n_ocean_pts,) float32 — per-cell fraction of the
        n_members ensemble members that classified that cell as noise
    """
    n_members, n_cells = aligned.shape

    # Counting votes label-by-label would need one (n_cells,) plane per cluster;
    # with thousands of clusters and tens of millions of cells that array alone
    # runs to hundreds of GB.  Sorting along the (short) member axis instead
    # gives every label's count as a run length, so memory scales with the
    # number of members, not the number of clusters.
    srt, run = _sorted_runs(aligned)

    plurality_cluster, best_count = _mode_from_runs(srt, run)

    # A cell is noise only when no member offered a cluster for it.
    vote_map = np.where(best_count > 0, plurality_cluster, -1).astype(np.int32)

    # Each label contributes once, read at the end of its run, where the run
    # length equals that label's total count.
    n_non_noise = (n_members - (aligned == -1).sum(axis=0)).astype(np.float64)

    noise_count    = (aligned == -1).sum(axis=0)       # (n_cells,)
    member_noise_frac = (noise_count / n_members).astype(np.float32)

    run_end       = np.ones(srt.shape, dtype=bool)
    run_end[:-1]  = srt[:-1] != srt[1:]
    run_end      &= run > 0          # sentinels never end a counted run

    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.where(run_end, run / np.maximum(n_non_noise, 1)[None, :], 0.0)
        plogp = np.where(probs > 0, probs * np.log2(probs), 0.0)
    entropy_map = (-plogp.sum(axis=0)).astype(np.float32)

    noise_frac = float((vote_map == -1).sum() / n_cells)

    return vote_map, entropy_map, noise_frac, member_noise_frac


def align_and_vote(ensemble: list[np.ndarray], tag: str) -> dict:
    """Run alignment + majority vote over one ensemble and log the summary."""
    log.info("=========== Hungarian Alignment (%s) ===========", tag)
    aligned, ref_idx, _ = align_ensemble(ensemble, reference="min_noise")
    log.info("Aligned shape : %s  (n_members × n_ocean_pts)", aligned.shape)

    log.info("=========== Majority Vote + Entropy (%s) ===========", tag)
    vote_map, entropy_map, noise_frac, member_noise_frac = majority_vote_and_entropy(aligned)

    log.info("Unique majority-vote labels : %d", np.unique(vote_map).size)
    log.info("Entropy range               : [%.3f, %.3f]",
             entropy_map.min(), entropy_map.max())
    log.info("Noise fraction (vote)       : %.3f", noise_frac)

    out_dict = {"vote_map": vote_map, "entropy_map": entropy_map,
                "noise_frac": noise_frac, "ref_idx": ref_idx,
                "member_noise_frac": member_noise_frac}
    
    return out_dict


# ── I/O helpers ────────────────────────────────────────────────────────────────

def db_load_results(path: Path) -> dict:
    results    = dict(np.load(path, allow_pickle=True))
    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data-dir", type=Path, default=Path("dbscan_results"), metavar="DIR")
    p.add_argument("--umap-md",  type=float, default=0.1, metavar="F",
                   help="UMAP min_dist value (for file naming only).")
    p.add_argument("--umap-nn",  type=int,   default=200, metavar="N",
                   help="UMAP n_neighbors value (for file naming only).")
    p.add_argument("--member-size", type=int, default=20, metavar="N",
                   help="Number of ensemble members for majority voting.")
    p.add_argument("--top-k-clusters", type=int, default=None, metavar="K",
                   help="Cap each ensemble member to its K largest clusters "
                        "before Hungarian alignment, merging the rest into a "
                        "single overflow cluster labelled K (noise -1 is "
                        "unaffected). Bounds the Hungarian cost-matrix size "
                        "for runs with very large cluster counts. Default: "
                        "no capping.")
    p.add_argument("--output-file", type=Path, help="Path to output file.")

    g = p.add_argument_group("DBSCAN parameters")
    g.add_argument("--eps",         type=float, default=0.5, metavar="F",
                   help="Neighbourhood radius (eps).")
    g.add_argument("--min-samples", type=int,   default=5,   metavar="N",
                   help="Minimum samples to form a core point.")

    s = p.add_argument_group("Spatial smoothing")
    s.add_argument("--grid-zarr", type=Path, required=True, metavar="PATH",
                   help="BVB zarr holding the original grid; its complete-case "
                        "mask back-projects the labels for filtering.")
    s.add_argument("--smooth-kernel", type=int, default=3, metavar="N",
                   help="Side of the square neighbourhood for the mode filter "
                        "(odd, >= 3). Default: 3.")
    s.add_argument("--smooth-fill-noise", action="store_true",
                   help="Let noise cells also take the neighbourhood mode. "
                        "Default: noise is excluded and passed through "
                        "unchanged.")
    return p


def main(argv=None) -> int:
    args     = build_parser().parse_args(argv)
    data_dir = args.data_dir
    md, nn   = args.umap_md, args.umap_nn
    eps, ms  = args.eps, args.min_samples

    log.info("=========== DBSCAN ensemble (before alignment) ===========")
    ensemble = []
    for m in range(1, args.member_size + 1):
        data_path = data_dir / f"emb_{m:02d}th_ensemble_md{md}_nn{nn}.npz"
        results   = db_load_results(data_path)
        labels    = results["cluster_labels"]
        ensemble.append(labels.astype(np.int32))
        log.info(
            "  Member %3d : n_ocean_points = %d  |  noise frac = %.3f",
            m, labels.size, noise_fraction(labels),
        )

    if args.top_k_clusters is not None:
        log.info("=========== Capping members to top %d clusters ===========", args.top_k_clusters)
        capped = []
        for m, labels in enumerate(ensemble, start=1):
            n_before = len(np.unique(labels[labels != -1]))
            labels_k = cap_to_top_k_clusters(labels, args.top_k_clusters)
            n_after  = len(np.unique(labels_k[labels_k != -1]))
            log.info(
                "  Member %3d : n_clusters = %d -> %d  |  noise frac = %.3f",
                m, n_before, n_after, noise_fraction(labels_k),
            )
            capped.append(labels_k)
        ensemble = capped

    # ── Branch 1: vote on the raw (unsmoothed) members ─────────────────────────
    before = align_and_vote(ensemble, "before smoothing")

    # ── Branch 2: spatial mode filter per member, then vote ────────────────────
    log.info("=========== Spatial mode filter (%d×%d) ===========",
             args.smooth_kernel, args.smooth_kernel)
    complete_mask = load_complete_mask(args.grid_zarr)
    log.info("Grid: %s %s  |  n_ocean_points = %d  |  fill_noise = %s",
             complete_mask.dims, complete_mask.shape,
             int(complete_mask.values.sum()), args.smooth_fill_noise)

    if complete_mask.values.sum() != ensemble[0].size:
        raise SystemExit(
            f"Grid mask has {int(complete_mask.values.sum())} valid cells but "
            f"the members carry {ensemble[0].size} labels — --grid-zarr does "
            f"not match this run."
        )

    smoothed_ensemble = []
    changed_fracs     = []
    for m, labels in enumerate(ensemble, start=1):
        labels_s, changed = smooth_member(
            labels, complete_mask,
            kernel_size = args.smooth_kernel,
            fill_noise  = args.smooth_fill_noise,
        )
        smoothed_ensemble.append(labels_s)
        changed_fracs.append(changed)
        log.info(
            "  Member %3d : n_clusters = %d -> %d  |  noise frac = %.3f -> %.3f"
            "  |  changed frac = %.4f",
            m,
            len(np.unique(labels[labels != -1])),
            len(np.unique(labels_s[labels_s != -1])),
            noise_fraction(labels), noise_fraction(labels_s), changed,
        )

    after = align_and_vote(smoothed_ensemble, "after smoothing")

    log.info("=========== Summary ===========")
    log.info("Vote noise fraction : %.3f (before)  ->  %.3f (after)",
             before["noise_frac"], after["noise_frac"])
    log.info("Mean vote entropy   : %.4f (before)  ->  %.4f (after)",
             float(before["entropy_map"].mean()), float(after["entropy_map"].mean()))

    out_dict = {
        # before smoothing — same key names as majority_vote.py
        "vote_map"             : before["vote_map"],
        "entropy_map"          : before["entropy_map"],
        "noise_frac"           : before["noise_frac"],
        "member_noise_frac"    : before["member_noise_frac"],
        "ref_idx"              : before["ref_idx"],
        # after smoothing
        "vote_map_smoothed"    : after["vote_map"],
        "entropy_map_smoothed" : after["entropy_map"],
        "noise_frac_smoothed"  : after["noise_frac"],
        "member_noise_frac_smoothed" : after["member_noise_frac"],
        "ref_idx_smoothed"     : after["ref_idx"],
        # smoothing provenance
        "smooth_kernel"        : args.smooth_kernel,
        "smooth_fill_noise"    : args.smooth_fill_noise,
        "smooth_changed_frac"  : np.asarray(changed_fracs, dtype=np.float64),
        # run parameters
        "eps"                  : eps,
        "ms"                   : ms,
        "md"                   : md,
        "nn"                   : nn,
    }

    np.savez_compressed(args.output_file, **out_dict)
    log.info("Saved results to → %s", args.output_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())

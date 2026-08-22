"""
hungarian_alignment.py
======================
Align HDBSCAN cluster labels across a UMAP ensemble using the Hungarian
(linear sum assignment) algorithm.

Context
-------
20 UMAP embeddings of the same ocean dataset, obtained with different random
seeds (same MD, NN). Each embedding has the same shape (n_ocean_pts, 2).
HDBSCAN is run independently on each — it always produces a label array of
length n_ocean_pts (one label per point), but decides the NUMBER of clusters
on its own. This number varies across members, making the cost matrix
rectangular and the Hungarian matching non-trivial for the larger side.

The algorithm finds the optimal permutation of each member's cluster labels
relative to a fixed reference member, maximising spatial overlap (number of
shared grid cells assigned to the same regime). Unmatched clusters (member
has more clusters than reference) receive offset labels that do not collide
with the reference label space.

Dependencies: numpy, scipy
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


# ── Utilities ─────────────────────────────────────────────────────────────────

def noise_fraction(labels: np.ndarray) -> float:
    """Fraction of points classified as noise (-1)."""
    return float((labels == -1).sum() / len(labels)) if len(labels) > 0 else 1.0


def _build_cost_matrix(
    ref_labels:   np.ndarray,   # (n_ocean_pts,) — reference member
    mem_labels:   np.ndarray,   # (n_ocean_pts,) — current member
    ref_clusters: np.ndarray,   # sorted unique non-noise labels in reference
    mem_clusters: np.ndarray,   # sorted unique non-noise labels in member
) -> np.ndarray:
    """
    Build a (n_ref_clusters × n_mem_clusters) overlap matrix.

    Entry [i, j] = number of grid cells where:
        reference assigns ref_clusters[i]  AND
        member    assigns mem_clusters[j]

    Cells where either side is noise (-1) are excluded — noise is not a
    regime and must never "consume" a matching slot.

    The matrix is rectangular when the two members find different numbers
    of clusters, which is the typical case with HDBSCAN.
    scipy.optimize.linear_sum_assignment handles rectangular matrices
    natively: given an (m × n) matrix with m < n it finds the m best
    column assignments, leaving n-m columns unmatched.
    """
    # Only count cells where both members assigned a real cluster
    both_clustered = (ref_labels != -1) & (mem_labels != -1)
    ref_c = ref_labels[both_clustered]
    mem_c = mem_labels[both_clustered]

    n_ref, n_mem = len(ref_clusters), len(mem_clusters)
    cost = np.zeros((n_ref, n_mem), dtype=np.int64)

    for i, rc in enumerate(ref_clusters):
        ref_mask = (ref_c == rc)
        for j, mc in enumerate(mem_clusters):
            cost[i, j] = int(np.count_nonzero(ref_mask & (mem_c == mc)))

    return cost


# ── Core alignment ────────────────────────────────────────────────────────────

def align_ensemble(
    labels_ensemble: list[np.ndarray],
    reference: int | str = "min_noise",
) -> tuple[np.ndarray, int, dict[int, dict]]:
    """
    Align HDBSCAN labels across ensemble members with the Hungarian algorithm.

    All arrays in labels_ensemble must have the same length (n_ocean_pts).
    The NUMBER of unique cluster labels may differ freely across members.

    Parameters
    ----------
    labels_ensemble : list of (n_ocean_pts,) int np.ndarray
        One HDBSCAN label array per ensemble member.
        Convention: -1 = noise, 0, 1, 2, ... = clusters.

    reference : int or "min_noise"
        Index of the reference member, or "min_noise" (default) to
        auto-select the member with the fewest noise points (most
        informative cluster structure — the safest anchor).

    Returns
    -------
    aligned : (n_members, n_ocean_pts) int32 array
        Aligned label arrays stacked into a single array, ready for
        majority voting and entropy computation.

    ref_idx : int
        Index of the chosen reference member.

    alignment_maps : dict[member_idx -> dict[orig_label -> aligned_label]]
        Per-member remapping dictionaries for inspection / provenance.
    """
    n_members  = len(labels_ensemble)
    n_ocean    = labels_ensemble[0].shape[0]

    assert all(lbl.shape == (n_ocean,) for lbl in labels_ensemble), \
        "All label arrays must have the same length (n_ocean_pts)."

    # ── Choose reference ───────────────────────────────────────────────────────
    if reference == "min_noise":
        noise_fracs = [noise_fraction(lbl) for lbl in labels_ensemble]
        ref_idx     = int(np.argmin(noise_fracs))
    elif isinstance(reference, int):
        ref_idx = reference
    else:
        raise ValueError(f"reference must be an int or 'min_noise', got {reference!r}")

    ref_labels   = labels_ensemble[ref_idx]
    ref_clusters = np.unique(ref_labels[ref_labels != -1])   # sorted, non-noise

    # Offset base: unmatched member clusters will receive labels starting here,
    # guaranteeing no collision with any reference label
    offset_base  = int(ref_clusters.max()) + 1 if len(ref_clusters) > 0 else 0

    print(f"Reference : member {ref_idx}  |  "
          f"noise frac = {noise_fraction(ref_labels):.3f}  |  "
          f"n_clusters = {len(ref_clusters)}")
    print()

    aligned        = np.empty((n_members, n_ocean), dtype=np.int32)
    alignment_maps = {}

    # ── Align every member ─────────────────────────────────────────────────────
    for m, mem_labels in enumerate(labels_ensemble):

        mem_clusters = np.unique(mem_labels[mem_labels != -1])
        label_map    = {-1: -1}   # noise is always identity

        if m == ref_idx:
            label_map.update({int(c): int(c) for c in ref_clusters})

        elif len(mem_clusters) == 0:
            pass   # fully noisy member — identity map ({-1: -1} only)

        else:
            # ── Rectangular cost matrix ────────────────────────────────────────
            # Shape: (n_ref_clusters × n_mem_clusters) — may be non-square.
            # linear_sum_assignment always matches the shorter axis optimally.
            #
            # Case A — member has FEWER clusters than reference (n_mem < n_ref):
            #   matrix is taller than wide; scipy matches all member clusters
            #   to the best n_mem reference clusters. The remaining (n_ref - n_mem)
            #   reference clusters are simply unrepresented in this member.
            #
            # Case B — member has MORE clusters than reference (n_mem > n_ref):
            #   matrix is wider than tall; scipy matches all reference clusters
            #   to their best partner in the member. The remaining
            #   (n_mem - n_ref) member clusters are unmatched and receive
            #   offset labels above the reference label space.
            cost             = _build_cost_matrix(ref_labels, mem_labels,
                                                  ref_clusters, mem_clusters)
            row_ind, col_ind = linear_sum_assignment(-cost)   # maximise overlap

            for row, col in zip(row_ind, col_ind):
                label_map[int(mem_clusters[col])] = int(ref_clusters[row])

            # Unmatched member clusters (Case B only)
            next_offset = offset_base
            for mc in mem_clusters:
                if int(mc) not in label_map:
                    label_map[int(mc)] = next_offset
                    next_offset       += 1

            n_ref_k     = len(ref_clusters)
            n_mem_k     = len(mem_clusters)
            n_matched   = len(row_ind)
            n_unmatched = n_mem_k - n_matched
            print(f"  Member {m:>3d}  |  "
                  f"ref clusters = {n_ref_k:>3d}  |  "
                  f"mem clusters = {n_mem_k:>3d}  |  "
                  f"matched = {n_matched:>3d}  |  "
                  f"unmatched = {n_unmatched:>2d}  |  "
                  f"noise frac = {noise_fraction(mem_labels):.3f}")

        alignment_maps[m] = label_map

        # Vectorised remap — np.vectorize with a dict is clean and fast enough
        # for the label arrays (n_ocean_pts values, small dict)
        aligned[m] = np.vectorize(label_map.__getitem__)(mem_labels)

    return aligned, ref_idx, alignment_maps


# ── Post-alignment: majority vote + entropy ───────────────────────────────────

def majority_vote_and_entropy(
    aligned: np.ndarray,         # (n_members, n_ocean_pts) int32
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Majority-vote label map and per-cell Shannon entropy from the aligned
    ensemble.  Noise (-1) is treated as a valid vote class, not discarded.

    Parameters
    ----------
    aligned : (n_members, n_ocean_pts) int32 — output of align_ensemble()

    Returns
    -------
    vote_map    : (n_ocean_pts,) int32   — majority-vote label per cell
    entropy_map : (n_ocean_pts,) float32 — Shannon entropy H per cell
    noise_frac  : float  — fraction of cells whose majority vote is noise
    """
    n_members, n_cells = aligned.shape

    # Shift so noise (-1) → 0 for bincount; clusters shift up by 1
    shifted    = aligned + 1                              # (n_members, n_cells)
    n_labels   = int(shifted.max()) + 1

    vote_map    = np.empty(n_cells, dtype=np.int32)
    entropy_map = np.empty(n_cells, dtype=np.float32)

    for c in range(n_cells):
        col  = shifted[:, c].astype(np.intp)
        bc   = np.bincount(col, minlength=n_labels)      # vote counts per label

        vote_map[c] = int(np.argmax(bc)) - 1             # shift back: 0 → -1

        probs           = bc / n_members
        nonzero         = probs[probs > 0]
        entropy_map[c]  = float(-np.sum(nonzero * np.log2(nonzero)))

    noise_frac = float((vote_map == -1).sum() / n_cells)
    return vote_map, entropy_map, noise_frac


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    rng          = np.random.default_rng(42)
    n_members    = 20
    n_ocean_pts  = 5000   # all members — same length

    # Simulate HDBSCAN results where each member finds a different number of
    # clusters (between 3 and 8) and uses an arbitrary label permutation
    ensemble = []
    print("=== Simulated ensemble (before alignment) ===")
    for m in range(n_members):
        n_k    = rng.integers(3, 9)        # HDBSCAN finds 3–8 clusters
        labels = rng.integers(0, n_k, size=n_ocean_pts)
        perm   = rng.permutation(n_k)
        labels = perm[labels]              # scramble label order
        noise  = rng.choice(n_ocean_pts, size=int(0.12 * n_ocean_pts), replace=False)
        labels[noise] = -1
        ensemble.append(labels.astype(np.int32))
        print(f"  Member {m:>3d} : n_clusters = {n_k}  |  "
              f"noise frac = {noise_fraction(labels):.3f}")

    print()
    print("=== Hungarian Alignment ===")
    aligned, ref_idx, maps = align_ensemble(ensemble, reference="min_noise")

    print()
    print(f"aligned shape : {aligned.shape}  (n_members × n_ocean_pts)")
    print(f"Label map for member 1 : {maps[1]}")

    print()
    print("=== Majority Vote + Entropy ===")
    vote_map, entropy_map, noise_frac = majority_vote_and_entropy(aligned)

    print(f"Unique majority-vote labels : {np.unique(vote_map)}")
    print(f"Entropy range               : [{entropy_map.min():.3f}, "
          f"{entropy_map.max():.3f}]")
    print(f"Noise fraction (vote)       : {noise_frac:.3f}")
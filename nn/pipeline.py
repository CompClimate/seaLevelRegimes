#!/usr/bin/env python
"""
Neural-network pipeline for sea-level (BV/BVB) regime classification.

The pipeline trains a multilayer perceptron (MLP) that maps the nine barotropic
vorticity budget (BVB) terms of a grid point/month onto a *probability
distribution* over the BV regimes, and then performs probabilistic inference:

  * Probabilistic maps -> which regimes are likely at each (time, lat, lon);
  * Entropy maps       -> how reliable that assignment is
                          (low entropy  -> one regime dominates -> confident,
                           high entropy -> ambiguous / transitional dynamics).

Design notes
------------
* Time-aware split: the data are a geophysical time series, so train/validation
  are split by time, never randomly.
* The scaler is fitted on the training months only and reused for validation and
  inference (no leakage).
* Class imbalance is handled with inverse-frequency regime weights, optionally
  ramped in with a curriculum (uniform -> balanced over `curriculum_warmup`
  epochs).
* Learning-rate scheduling uses a dual criterion (validation loss + entropy),
  and training is guarded by an early-stopping controller that restores the
  best weights.
* Stacking is done with plain NumPy reshapes (no xarray MultiIndex), and batches
  are cut as tensor slices instead of per-sample `Dataset.__getitem__` calls;
  both are considerably faster and lighter than the naive versions.
* Inference is streamed in blocks of months and appended to a Zarr store, so the
  memory cost never scales with the length of the record.

Usage
-----
    python pipeline.py --input <features.zarr> --outdir <dir> [options]
    python pipeline.py --help

Typically driven by `pipeline.sh` (Slurm) via `submit.sh`.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from sklearn.preprocessing import StandardScaler
from torch import nn

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Root of the static/derived CM4X data (mirrors src.aux_func.BASE_DIR).
BASE_DIR = os.environ.get("SLVP_BASE_DIR", "/work/lnd/CM4X")

# Barotropic vorticity budget terms used as NN features.
# Source of truth: src/aux_func.py::bvb_terms
BVB_TERMS = ["beta_V", "BPT", "Mass_flux", "eta_dt",
             "Curl_dudt", "Curl_taus", "Curl_taub", "Curl_Adv", "Curl_diff"]

LABEL_VAR = "bvb_regime"
GRID_DIMS = ("time", "lat", "lon")
EPS = 1e-12

LOGGER = logging.getLogger("bvb.pipeline")

PROB_DESCRIPTION = ("Predicted probability distribution over all BV regimes for each grid "
                    "point and time.")
ENT_DESCRIPTION = ("Normalised Shannon entropy in [0, 1] (or [0, 100] percent), measuring "
                   "probabilistic calibration and regime ambiguity: (1) low entropy -> one "
                   "regime dominates -> high confidence, stable classification; (2) high "
                   "entropy -> probabilities are spread across regimes -> ambiguous or "
                   "transitional dynamics.")
DS_DESCRIPTION = ("In this BV regime classification problem, a probabilistic inference has been "
                  "used. The model does not assign a single regime outright: for each grid point "
                  "and time it outputs a probability distribution over all BV regimes. From these "
                  "probabilities we compute the entropy, which summarises how confident the model "
                  "is. Together, (1) probabilistic maps tell which regimes are likely, and "
                  "(2) entropy maps tell how reliable that assignment is. This combination allows "
                  "us to distinguish well-defined BV regimes from regions or times of dynamical "
                  "transition, making the classification both informative and physically "
                  "interpretable.")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class ModelConfig:
    """Everything needed to rebuild the network from a checkpoint."""
    features: list[str] = field(default_factory=lambda: list(BVB_TERMS))
    label_var: str = LABEL_VAR
    n_regimes: int = 15
    hidden: tuple[int, ...] = (256, 128, 64, 32, 16)
    rare_regimes: bool = True
    dropout: float = 0.0

    @property
    def n_features(self) -> int:
        return len(self.features)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        d = dict(d)
        d["hidden"] = tuple(d["hidden"])
        d["features"] = list(d["features"])
        return cls(**d)


@dataclass
class TrainConfig:
    """Optimisation / scheduling hyper-parameters."""
    epochs: int = 100
    batch_size: int = 8192
    lr: float = 1e-3
    weight_decay: float = 1e-5
    train_frac: float = 0.7
    # Early stopping (on the dual criterion)
    patience: int = 16
    min_delta: float = 1e-4
    # Dual-criterion LR scheduler
    lambda_entropy: float = 0.25
    sched_factor: float = 0.5
    sched_patience: int = 8
    sched_threshold: float = 1e-3
    sched_cooldown: int = 5
    min_lr: float = 1e-6
    # Class imbalance
    class_weights: str = "balanced"      # balanced | curriculum | none
    curriculum_warmup: int = 10
    # Runtime
    data_on_device: str = "auto"         # auto | gpu | cpu
    seed: int = 42


# --------------------------------------------------------------------------- #
# Runtime helpers
# --------------------------------------------------------------------------- #

def setup_logging(verbose: bool = False) -> None:
    """Configure stdout logging (UTC timestamps), matching the project style."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s - %(asctime)s - %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
        stream=sys.stdout,
        force=True,
    )
    logging.Formatter.converter = time.gmtime


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve the compute device, honouring 'auto'."""
    if spec in (None, "", "auto"):
        spec = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("CUDA requested but unavailable - falling back to CPU.")
        device = torch.device("cpu")
    return device


def configure_runtime(seed: int, device: torch.device) -> None:
    """Seed the RNGs and size the thread pool from the Slurm allocation."""
    n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 0)) or (os.cpu_count() or 1)
    torch.set_num_threads(max(1, n_threads))

    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        # TF32 matmuls: free speed-up on Ampere/Ada, irrelevant for accuracy here.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    LOGGER.info("Device: %s | torch threads: %d | seed: %d",
                device, torch.get_num_threads(), seed)


def make_dirs(path: str | Path) -> Path:
    """Create a directory (and parents) if it does not exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #

def open_dataset(path: str | Path,
                 features: list[str],
                 label_var: str | None = None,
                 years: tuple[str, str] | None = None,
                 time_stride: int = 1) -> xr.Dataset:
    """
    Open the feature/label store and subset it.

    Args:
        path: Zarr store (or NetCDF file) with the BVB terms and regime labels.
        features: Feature variable names that must be present.
        label_var: Label variable name, or None for inference-only datasets.
        years: Optional inclusive (start, end) year selection, e.g. ("2005", "2011").
        time_stride: Keep every n-th time step (1 = keep all).

    Returns:
        xr.Dataset restricted to the requested variables and time range.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input dataset not found: {path}")

    if path.suffix == ".zarr" or path.is_dir():
        ds = xr.open_zarr(path, chunks=None)
    else:
        ds = xr.open_dataset(path, chunks=None)

    wanted = list(features) + ([label_var] if label_var else [])
    missing = [v for v in wanted if v not in ds.variables]
    if missing:
        raise KeyError(f"Variables missing from {path}: {missing}. "
                       f"Available: {sorted(ds.data_vars)}")

    missing_dims = [d for d in GRID_DIMS if d not in ds.dims]
    if missing_dims:
        raise KeyError(f"Dataset {path} is missing the dimensions {missing_dims}; "
                       f"expected {GRID_DIMS}.")

    ds = ds[wanted]
    if years is not None:
        ds = ds.sel(time=slice(years[0], years[1]))
    if time_stride > 1:
        ds = ds.isel(time=slice(None, None, time_stride))
    if ds.sizes["time"] == 0:
        raise ValueError(f"No time steps left after subsetting {path} with "
                         f"years={years}, time_stride={time_stride}.")

    LOGGER.info("Loaded %s | time: %s -> %s (%d steps) | grid: %d x %d",
                path.name,
                np.datetime_as_string(ds["time"].values[0], unit="D"),
                np.datetime_as_string(ds["time"].values[-1], unit="D"),
                ds.sizes["time"], ds.sizes["lat"], ds.sizes["lon"])
    return ds


@dataclass
class SampleSet:
    """Flattened, finite-only samples plus what is needed to map them back."""
    X: np.ndarray                 # (n_valid, n_features) float32
    y: np.ndarray | None          # (n_valid,) int64, or None
    valid: np.ndarray             # (n_total,) bool  -- ocean/finite mask
    shape: tuple[int, int, int]   # (ntime, nlat, nlon)

    @property
    def n_valid(self) -> int:
        return self.X.shape[0]


def prepare_ml_data(ds: xr.Dataset,
                    features: list[str],
                    label_var: str | None = None,
                    scaler: StandardScaler | None = None,
                    fit_scaler: bool = False) -> SampleSet:
    """
    Stack (time, lat, lon) onto a sample axis, drop non-finite samples and scale.

    This replaces `xr.Dataset.stack`, which builds an expensive MultiIndex: a
    plain C-order reshape carries the same information because the sample order
    is exactly `(time, lat, lon)` row-major, and `SampleSet.shape` is enough to
    invert it.

    Args:
        ds: Dataset holding the feature variables (and optionally the labels).
        features: Feature variable names, in the order the model expects them.
        label_var: Label variable name, or None when labels are not needed.
        scaler: StandardScaler to apply (fitted on the training months only).
        fit_scaler: Fit the scaler on these samples before transforming.

    Returns:
        SampleSet with finite features (and labels), the validity mask and the
        original grid shape.
    """
    shape = tuple(int(ds.sizes[d]) for d in GRID_DIMS)
    n_total = int(np.prod(shape))

    X = np.empty((n_total, len(features)), dtype=np.float32)
    for j, var in enumerate(features):
        X[:, j] = ds[var].transpose(*GRID_DIMS).values.reshape(-1)

    valid = np.isfinite(X).all(axis=1)

    y = None
    if label_var is not None:
        y_flat = ds[label_var].transpose(*GRID_DIMS).values.reshape(-1)
        valid &= np.isfinite(y_flat)
        y = y_flat[valid].astype(np.int64)

    Xv = X[valid]
    del X

    if scaler is not None:
        Xv = (scaler.fit_transform(Xv) if fit_scaler else scaler.transform(Xv))
        Xv = Xv.astype(np.float32, copy=False)

    LOGGER.info("Prepared %s samples out of %s grid cells (%.1f%% valid), %.2f GiB",
                f"{Xv.shape[0]:,}", f"{n_total:,}", 100.0 * valid.mean(),
                Xv.nbytes / 2**30)
    return SampleSet(X=Xv, y=y, valid=valid, shape=shape)


def time_based_split(ds: xr.Dataset, train_frac: float = 0.7) -> tuple[xr.Dataset, xr.Dataset]:
    """
    Split a (time, lat, lon) dataset chronologically.

    Because this is a geophysical time series, the split is by time (month) and
    never random: the validation months always follow the training months.

    Args:
        ds: Input dataset.
        train_frac: Fraction of the time steps used for training.

    Returns:
        (train_ds, val_ds)
    """
    if not 0.0 < train_frac < 1.0:
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}.")

    ntime = ds.sizes["time"]
    ntrain = max(1, min(ntime - 1, int(round(train_frac * ntime))))

    train_ds = ds.isel(time=slice(0, ntrain))
    val_ds = ds.isel(time=slice(ntrain, None))

    LOGGER.info("Time-based split: %d training months / %d validation months",
                train_ds.sizes["time"], val_ds.sizes["time"])
    return train_ds, val_ds


def compute_regime_weights(y: np.ndarray, n_regimes: int, alpha: float = 1.0) -> torch.Tensor:
    """
    Inverse-frequency regime weights, to counteract class imbalance.

    Regimes that are absent from `y` get a zero weight (they can never be a
    target, and the naive `count.sum() / (count + eps)` would otherwise blow
    them up to ~1e8 and destroy the loss normalisation).

    Args:
        y: Integer label vector.
        n_regimes: Total number of regimes (classes).
        alpha: Curriculum blend in [0, 1]; 0 -> uniform weights, 1 -> fully
            balanced. Values in between ramp the balancing in gradually.

    Returns:
        torch.Tensor of shape (n_regimes,), mean 1 over the present regimes.
    """
    counts = np.bincount(y, minlength=n_regimes).astype(np.float64)
    if counts.size > n_regimes:
        raise ValueError(f"Labels contain {counts.size} classes but n_regimes={n_regimes}.")

    present = counts > 0
    weights = np.zeros(n_regimes, dtype=np.float64)
    weights[present] = counts[present].sum() / counts[present]
    weights[present] /= weights[present].mean()

    if alpha < 1.0:  # blend towards uniform weights over the present regimes
        weights = (1.0 - alpha) * present.astype(np.float64) + alpha * weights

    return torch.tensor(weights, dtype=torch.float32)


def curriculum_alpha(epoch: int, warmup: int) -> float:
    """Curriculum ramp: 0 at epoch 0, 1 once `warmup` epochs have elapsed."""
    return 1.0 if warmup <= 0 else min(1.0, epoch / float(warmup))


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #

class TensorBatcher:
    """
    Minimal-overhead batch iterator over two resident tensors.

    A `DataLoader` over a per-sample `Dataset` pays a Python call and a collate
    per *sample*; with tens of millions of tabular rows that dominates the run
    time. Here the features and labels live in one contiguous tensor and each
    batch is a single gather, optionally staged straight on the GPU.

    Args:
        X, y: NumPy arrays of features and labels.
        batch_size: Samples per batch.
        shuffle: Reshuffle the sample order every epoch (training only).
        device: Device the batches are consumed on.
        storage: 'auto' | 'gpu' | 'cpu' - where the full tensors are held.
            'auto' keeps them on the GPU when they comfortably fit.
        min_batch: Drop a trailing batch smaller than this (BatchNorm needs >= 2).
    """

    #: Fraction of free GPU memory the resident tensors may occupy under 'auto'.
    GPU_BUDGET = 0.35

    def __init__(self, X, y, batch_size, *, shuffle=False, device=None,
                 storage="auto", min_batch=1, generator=None):
        device = device or torch.device("cpu")
        self.device = device
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.min_batch = min_batch
        self.generator = generator

        Xt = torch.from_numpy(np.ascontiguousarray(X))
        yt = torch.from_numpy(np.ascontiguousarray(y))

        self.storage = self._resolve_storage(storage, device, Xt.nbytes + yt.nbytes)
        if self.storage.type == "cuda":
            self.X, self.y = Xt.to(self.storage), yt.to(self.storage)
        else:
            # Pinned host memory makes the H2D copy of each batch async.
            pin = device.type == "cuda"
            self.X = Xt.pin_memory() if pin else Xt
            self.y = yt.pin_memory() if pin else yt

        self.n = self.X.shape[0]
        if self.n == 0:
            raise ValueError("TensorBatcher received an empty sample set.")

    @classmethod
    def _resolve_storage(cls, storage, device, nbytes) -> torch.device:
        if device.type != "cuda" or storage == "cpu":
            return torch.device("cpu")
        if storage == "gpu":
            return device
        free, _ = torch.cuda.mem_get_info(device)
        if nbytes < cls.GPU_BUDGET * free:
            LOGGER.info("Staging %.2f GiB of samples on %s", nbytes / 2**30, device)
            return device
        LOGGER.info("Samples (%.2f GiB) exceed the GPU budget - streaming from host memory.",
                    nbytes / 2**30)
        return torch.device("cpu")

    def __len__(self) -> int:
        full, rest = divmod(self.n, self.batch_size)
        return full + (1 if rest >= self.min_batch else 0)

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(self.n, device=self.storage, generator=self.generator)
        else:
            order = None

        non_blocking = self.storage.type == "cpu" and self.device.type == "cuda"
        for start in range(0, self.n, self.batch_size):
            stop = min(start + self.batch_size, self.n)
            if stop - start < self.min_batch:
                break
            idx = slice(start, stop) if order is None else order[start:stop]
            xb = self.X[idx].to(self.device, non_blocking=non_blocking)
            yb = self.y[idx].to(self.device, non_blocking=non_blocking)
            yield xb, yb


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class BVBMLP(nn.Module):
    """
    Fully connected classifier over the BVB terms.

    Each hidden block is Linear -> BatchNorm -> activation (-> Dropout), which
    keeps the pre-activations well conditioned. The activation follows the
    original heuristic: deep stacks use a smooth, non-saturating unit (SiLU when
    rare regimes must be resolved, GELU otherwise), while shallow stacks use Tanh.
    The head emits logits; the softmax is handled by the loss.
    """

    def __init__(self, n_features: int = 9, n_regimes: int = 15,
                 hidden: tuple[int, ...] = (64, 32, 16),
                 rare_regimes: bool = True, dropout: float = 0.0):
        super().__init__()

        if len(hidden) >= 3:
            activation = nn.SiLU if rare_regimes else nn.GELU
        else:
            activation = nn.Tanh

        layers: list[nn.Module] = []
        in_dim = n_features
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), activation()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, n_regimes))

        self.net = nn.Sequential(*layers)
        self.n_features = n_features
        self.n_regimes = n_regimes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # logits; softmax handled by the loss


class DualCriterionScheduler:
    """
    Learning-rate scheduler driven by validation loss *and* entropy.

    A model can lower its loss while remaining diffuse over regimes; adding the
    entropy to the plateau metric makes the LR react to confidence as well as
    to accuracy.
    """

    def __init__(self, scheduler, lambda_entropy: float = 0.3):
        """
        Args:
            scheduler: Wrapped scheduler, e.g. ReduceLROnPlateau.
            lambda_entropy: Weight of the entropy penalty in the combined metric.
        """
        self.scheduler = scheduler
        self.lambda_entropy = lambda_entropy
        self.last_metric = float("inf")

    def step(self, val_loss: float, val_entropy: float) -> float:
        self.last_metric = val_loss + self.lambda_entropy * val_entropy
        self.scheduler.step(self.last_metric)
        return self.last_metric


class TrainingController:
    """
    Early stopping with best-weight restoration.

    Tracks the combined validation metric, keeps a CPU copy of the best weights
    and signals when `patience` epochs have passed without improvement.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-4, restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best

        self.best_score = np.inf
        self.best_epoch = -1
        self.best_state: dict | None = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float, model: nn.Module, epoch: int = -1) -> bool:
        """Returns True when this epoch improved on the best score so far."""
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best:
                self.best_state = {k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()}
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False

    def restore(self, model: nn.Module) -> None:
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)
            LOGGER.info("Restored best weights from epoch %d (score %.6f)",
                        self.best_epoch, self.best_score)


# --------------------------------------------------------------------------- #
# Training / evaluation
# --------------------------------------------------------------------------- #

@dataclass
class EpochStats:
    """Per-epoch aggregates. Entropy is normalised to [0, 1]."""
    loss: float
    entropy: float
    accuracy: float
    balanced_accuracy: float = float("nan")


def _normalised_entropy(log_probs: torch.Tensor) -> torch.Tensor:
    """Sum of per-sample Shannon entropies, normalised by ln(K) so it lies in [0, 1].

    Computed from log-softmax rather than from `log(softmax + eps)`: same result,
    but numerically stable and one fewer exp/log round trip.
    """
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=1).sum() / math.log(log_probs.shape[1])


def train_one_epoch(model, batches, optimizer, criterion, device) -> EpochStats:
    """
    Run one training epoch.

    Metrics are accumulated in device-side tensors and synchronised once at the
    end of the epoch, instead of calling `.item()` on every batch.

    Args:
        model: Network being trained.
        batches: Iterable of (features, labels) batches already on `device`.
        optimizer: Optimizer.
        criterion: Loss function (expects logits).
        device: Compute device.

    Returns:
        EpochStats with the mean loss, normalised entropy and accuracy.
    """
    model.train()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    ent_sum = torch.zeros((), device=device, dtype=torch.float64)
    correct = torch.zeros((), device=device, dtype=torch.float64)
    n_seen = 0

    for X, y in batches:
        optimizer.zero_grad(set_to_none=True)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            log_probs = torch.log_softmax(logits, dim=1)
            bs = y.numel()
            loss_sum += loss.detach().double() * bs
            ent_sum += _normalised_entropy(log_probs).double()
            correct += (log_probs.argmax(dim=1) == y).sum().double()
            n_seen += bs

    if n_seen == 0:
        raise RuntimeError("Training epoch saw no samples.")

    return EpochStats(loss=(loss_sum / n_seen).item(),
                      entropy=(ent_sum / n_seen).item(),
                      accuracy=(correct / n_seen).item())


@torch.no_grad()
def evaluate(model, batches, criterion, device, n_regimes: int) -> EpochStats:
    """
    Evaluate on the validation set.

    Also accumulates the confusion matrix (one `bincount` per batch) so that the
    balanced accuracy - the metric that actually reflects the rare regimes - is
    available for free.

    Args:
        model: Network being evaluated.
        batches: Iterable of (features, labels) batches already on `device`.
        criterion: Loss function (expects logits).
        device: Compute device.
        n_regimes: Number of classes.

    Returns:
        EpochStats with loss, normalised entropy, accuracy and balanced accuracy.
    """
    model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    ent_sum = torch.zeros((), device=device, dtype=torch.float64)
    confusion = torch.zeros(n_regimes * n_regimes, device=device, dtype=torch.int64)
    n_seen = 0

    for X, y in batches:
        logits = model(X)
        loss = criterion(logits, y)

        log_probs = torch.log_softmax(logits, dim=1)
        pred = log_probs.argmax(dim=1)

        bs = y.numel()
        loss_sum += loss.detach().double() * bs
        ent_sum += _normalised_entropy(log_probs).double()
        confusion += torch.bincount(y * n_regimes + pred, minlength=n_regimes * n_regimes)
        n_seen += bs

    if n_seen == 0:
        raise RuntimeError("Evaluation saw no samples.")

    confusion = confusion.reshape(n_regimes, n_regimes).double()
    support = confusion.sum(dim=1)
    per_class = torch.diagonal(confusion)[support > 0] / support[support > 0]

    return EpochStats(loss=(loss_sum / n_seen).item(),
                      entropy=(ent_sum / n_seen).item(),
                      accuracy=(torch.diagonal(confusion).sum() / n_seen).item(),
                      balanced_accuracy=per_class.mean().item())


def init_history() -> dict[str, list]:
    """Initialise the training-history record."""
    return {"epoch": [], "lr": [],
            "train_loss": [], "val_loss": [],
            "train_entropy": [], "val_entropy": [],
            "train_accuracy": [], "val_accuracy": [], "val_balanced_accuracy": [],
            "metric": []}


def save_history(history: dict[str, list], path: str | Path) -> None:
    """Write the training history to CSV (one row per epoch)."""
    path = Path(path)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(history.keys())
        writer.writerows(zip(*history.values()))
    LOGGER.info("Training history written to %s", path)


def train_model(model, train_batches, val_batches, optimizer, criterion,
                scheduler, controller, device, history, n_epochs,
                n_regimes, class_weight_fn=None) -> dict[str, list]:
    """
    Full training loop: epochs, dual-criterion scheduling, early stopping.

    Args:
        model: Network to train (already on `device`).
        train_batches / val_batches: Batch iterables.
        optimizer: Optimizer.
        criterion: Loss function; its `weight` buffer is refreshed each epoch
            when `class_weight_fn` is given (curriculum weighting).
        scheduler: DualCriterionScheduler.
        controller: TrainingController (early stopping + best-weight restore).
        device: Compute device.
        history: Dict from `init_history()`, updated in place.
        n_epochs: Maximum number of epochs.
        n_regimes: Number of classes.
        class_weight_fn: Optional `epoch -> weight tensor` callable.

    Returns:
        The updated history dict.
    """
    LOGGER.info("Training for up to %d epochs (%d train batches, %d val batches)",
                n_epochs, len(train_batches), len(val_batches))
    t0 = time.time()

    for epoch in range(1, n_epochs + 1):
        if class_weight_fn is not None:
            criterion.weight = class_weight_fn(epoch).to(device)

        train_stats = train_one_epoch(model, train_batches, optimizer, criterion, device)
        val_stats = evaluate(model, val_batches, criterion, device, n_regimes)

        metric = scheduler.step(val_stats.loss, val_stats.entropy)
        lr = optimizer.param_groups[0]["lr"]
        improved = controller.step(metric, model, epoch)

        history["epoch"].append(epoch)
        history["lr"].append(lr)
        history["train_loss"].append(train_stats.loss)
        history["val_loss"].append(val_stats.loss)
        history["train_entropy"].append(train_stats.entropy)
        history["val_entropy"].append(val_stats.entropy)
        history["train_accuracy"].append(train_stats.accuracy)
        history["val_accuracy"].append(val_stats.accuracy)
        history["val_balanced_accuracy"].append(val_stats.balanced_accuracy)
        history["metric"].append(metric)

        LOGGER.info("epoch %3d/%d | loss %.5f/%.5f | entropy %.4f/%.4f | "
                    "acc %.4f/%.4f (bal %.4f) | metric %.5f | lr %.2e%s",
                    epoch, n_epochs, train_stats.loss, val_stats.loss,
                    train_stats.entropy, val_stats.entropy,
                    train_stats.accuracy, val_stats.accuracy,
                    val_stats.balanced_accuracy, metric, lr,
                    "  *" if improved else "")

        if controller.should_stop:
            LOGGER.info("Early stopping at epoch %d (no improvement for %d epochs).",
                        epoch, controller.patience)
            break

    controller.restore(model)
    LOGGER.info("Training finished in %.1f s.", time.time() - t0)
    return history


# --------------------------------------------------------------------------- #
# Probabilistic inference
# --------------------------------------------------------------------------- #

@torch.no_grad()
def predict_probabilistic_maps(model, ds_new, scaler, device, features,
                               entropy_unit: str = "fraction",
                               batch_size: int = 131072) -> xr.Dataset:
    """
    Probabilistic and entropy maps for one block of the record.

    The forward pass is chunked so that the activation memory stays bounded, and
    the land/invalid mask is preserved as NaN in the output.

    Args:
        model: Trained network.
        ds_new: Dataset with the feature variables on (time, lat, lon).
        scaler: Scaler fitted on the training months.
        device: Compute device.
        features: Feature variable names, in training order.
        entropy_unit: "fraction" -> [0, 1] or "percent" -> [0, 100].
        batch_size: Samples per forward pass.

    Returns:
        xr.Dataset with `regime_prob` (time, lat, lon, regime), `regime_entropy`,
        `regime_pred` and `regime_confidence`.
    """
    if entropy_unit not in ("fraction", "percent"):
        raise ValueError(f"entropy_unit must be 'fraction' or 'percent', got {entropy_unit!r}.")

    samples = prepare_ml_data(ds_new, features, label_var=None,
                              scaler=scaler, fit_scaler=False)
    n_total = int(np.prod(samples.shape))
    K = model.n_regimes

    model.eval().to(device)

    probs = np.empty((samples.n_valid, K), dtype=np.float32)
    entropy = np.empty(samples.n_valid, dtype=np.float32)

    log_K = math.log(K)
    for start in range(0, samples.n_valid, batch_size):
        stop = min(start + batch_size, samples.n_valid)
        xb = torch.from_numpy(samples.X[start:stop]).to(device)
        log_p = torch.log_softmax(model(xb), dim=1)
        p = log_p.exp()
        probs[start:stop] = p.cpu().numpy()
        entropy[start:stop] = (-(p * log_p).sum(dim=1) / log_K).cpu().numpy()

    if entropy_unit == "percent":
        entropy *= 100.0

    # Scatter back onto the full grid, keeping the land mask as NaN.
    prob_map = np.full((n_total, K), np.nan, dtype=np.float32)
    ent_map = np.full(n_total, np.nan, dtype=np.float32)
    pred_map = np.full(n_total, np.nan, dtype=np.float32)
    conf_map = np.full(n_total, np.nan, dtype=np.float32)

    prob_map[samples.valid] = probs
    ent_map[samples.valid] = entropy
    pred_map[samples.valid] = probs.argmax(axis=1).astype(np.float32)
    conf_map[samples.valid] = probs.max(axis=1)

    nt, nlat, nlon = samples.shape
    coords = {"time": ds_new["time"], "lat": ds_new["lat"], "lon": ds_new["lon"]}

    prob_da = xr.DataArray(prob_map.reshape(nt, nlat, nlon, K),
                           dims=GRID_DIMS + ("regime",),
                           coords={**coords, "regime": np.arange(K)},
                           name="regime_prob",
                           attrs={"description": PROB_DESCRIPTION, "units": "1"})
    ent_da = xr.DataArray(ent_map.reshape(nt, nlat, nlon),
                          dims=GRID_DIMS, coords=coords, name="regime_entropy",
                          attrs={"description": ENT_DESCRIPTION,
                                 "units": "percent" if entropy_unit == "percent" else "1"})
    pred_da = xr.DataArray(pred_map.reshape(nt, nlat, nlon),
                           dims=GRID_DIMS, coords=coords, name="regime_pred",
                           attrs={"description": "Most likely BV regime (argmax of regime_prob)."})
    conf_da = xr.DataArray(conf_map.reshape(nt, nlat, nlon),
                           dims=GRID_DIMS, coords=coords, name="regime_confidence",
                           attrs={"description": "Probability of the most likely BV regime.",
                                  "units": "1"})

    pred_ds = xr.merge([prob_da, ent_da, pred_da, conf_da])
    pred_ds.attrs = {"description": DS_DESCRIPTION,
                     "features": ", ".join(features),
                     "n_regimes": K,
                     "entropy_unit": entropy_unit}
    return pred_ds


def predict_to_store(model, ds_new, scaler, device, features, out_path,
                     entropy_unit: str = "fraction",
                     batch_size: int = 131072,
                     time_chunk: int = 12) -> Path:
    """
    Stream probabilistic inference over the whole record into a store.

    The probability field is `n_time * n_lat * n_lon * n_regimes` floats, which
    for a global eddy-permitting grid is far larger than memory. Inference is
    therefore run in blocks of `time_chunk` months and appended along `time`
    (Zarr); NetCDF outputs are assembled in memory and are only appropriate for
    short records.

    Args:
        model, ds_new, scaler, device, features, entropy_unit, batch_size:
            As in `predict_probabilistic_maps`.
        out_path: Destination `.zarr` store or `.nc` file.
        time_chunk: Number of time steps per inference block.

    Returns:
        Path to the written store.
    """
    out_path = Path(out_path)
    ntime = ds_new.sizes["time"]
    blocks = range(0, ntime, time_chunk)
    as_zarr = out_path.suffix == ".zarr"

    LOGGER.info("Inference over %d time steps in %d block(s) -> %s",
                ntime, len(blocks), out_path)

    accumulated = []
    for i, start in enumerate(blocks):
        block = ds_new.isel(time=slice(start, min(start + time_chunk, ntime)))
        pred = predict_probabilistic_maps(model, block, scaler, device, features,
                                          entropy_unit=entropy_unit,
                                          batch_size=batch_size)
        if as_zarr:
            pred = pred.chunk({"time": min(time_chunk, pred.sizes["time"])})
            if i == 0:
                pred.to_zarr(out_path, mode="w", consolidated=True)
            else:
                pred.to_zarr(out_path, append_dim="time", consolidated=True)
        else:
            accumulated.append(pred)
        LOGGER.info("  block %d/%d written (%d time steps)",
                    i + 1, len(blocks), block.sizes["time"])

    if not as_zarr:
        merged = xr.concat(accumulated, dim="time")
        encoding = {v: {"zlib": True, "complevel": 4} for v in merged.data_vars}
        merged.to_netcdf(out_path, encoding=encoding)

    LOGGER.info("Predictions saved to %s", out_path)
    return out_path


# --------------------------------------------------------------------------- #
# End-to-end estimator
# --------------------------------------------------------------------------- #

class BVBRegimeMLP:
    """
    Train / predict / persist wrapper around `BVBMLP`.

    Owns the network, the feature scaler and the training history, so that a
    checkpoint is self-sufficient: it can be reloaded and used for inference
    without repeating the architecture on the command line.
    """

    def __init__(self, config: ModelConfig | None = None, device: str | torch.device | None = None):
        self.config = config or ModelConfig()
        self.device = device if isinstance(device, torch.device) else resolve_device(device or "auto")
        self.model = BVBMLP(n_features=self.config.n_features,
                            n_regimes=self.config.n_regimes,
                            hidden=self.config.hidden,
                            rare_regimes=self.config.rare_regimes,
                            dropout=self.config.dropout).to(self.device)
        self.scaler = StandardScaler()
        self.history = init_history()
        self.fitted = False

        n_params = sum(p.numel() for p in self.model.parameters())
        LOGGER.info("Model: %d features -> %s -> %d regimes (%s parameters) on %s",
                    self.config.n_features, list(self.config.hidden),
                    self.config.n_regimes, f"{n_params:,}", self.device)

    # -- training ---------------------------------------------------------- #

    def fit(self, ds: xr.Dataset, train_cfg: TrainConfig | None = None) -> dict[str, list]:
        """
        Fit the classifier on a labelled dataset.

        Args:
            ds: Dataset with the feature variables and the regime labels.
            train_cfg: Optimisation hyper-parameters.

        Returns:
            The training history.
        """
        cfg = train_cfg or TrainConfig()
        train_ds, val_ds = time_based_split(ds, cfg.train_frac)

        train = prepare_ml_data(train_ds, self.config.features, self.config.label_var,
                                scaler=self.scaler, fit_scaler=True)
        val = prepare_ml_data(val_ds, self.config.features, self.config.label_var,
                              scaler=self.scaler, fit_scaler=False)

        for name, s in (("training", train), ("validation", val)):
            hi = int(s.y.max())
            if hi >= self.config.n_regimes:
                raise ValueError(f"The {name} labels reach {hi}, but n_regimes="
                                 f"{self.config.n_regimes}. Pass --n-regimes {hi + 1}.")

        # `min_batch=2`: BatchNorm cannot normalise a trailing batch of one sample.
        train_batches = TensorBatcher(train.X, train.y, cfg.batch_size, shuffle=True,
                                      device=self.device, storage=cfg.data_on_device,
                                      min_batch=2)
        val_batches = TensorBatcher(val.X, val.y, cfg.batch_size, shuffle=False,
                                    device=self.device, storage=cfg.data_on_device)

        # Class-imbalance handling
        if cfg.class_weights == "none":
            weights, weight_fn = None, None
        elif cfg.class_weights == "balanced":
            weights = compute_regime_weights(train.y, self.config.n_regimes).to(self.device)
            weight_fn = None
        elif cfg.class_weights == "curriculum":
            y_train = train.y
            weight_fn = (lambda epoch: compute_regime_weights(
                y_train, self.config.n_regimes,
                alpha=curriculum_alpha(epoch, cfg.curriculum_warmup)))
            weights = weight_fn(1).to(self.device)
        else:
            raise ValueError(f"Unknown class_weights strategy: {cfg.class_weights!r}")

        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=cfg.lr, weight_decay=cfg.weight_decay)
        base_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",                        # we minimise the combined metric
            factor=cfg.sched_factor,           # lr <- lr * factor
            patience=cfg.sched_patience,       # epochs to wait before reducing
            threshold=cfg.sched_threshold,     # minimum improvement that counts
            cooldown=cfg.sched_cooldown,       # epochs to wait after a reduction
            min_lr=cfg.min_lr,
        )
        scheduler = DualCriterionScheduler(base_scheduler, lambda_entropy=cfg.lambda_entropy)
        controller = TrainingController(patience=cfg.patience, min_delta=cfg.min_delta,
                                        restore_best=True)

        train_model(model=self.model,
                    train_batches=train_batches, val_batches=val_batches,
                    optimizer=optimizer, criterion=criterion,
                    scheduler=scheduler, controller=controller,
                    device=self.device, history=self.history,
                    n_epochs=cfg.epochs, n_regimes=self.config.n_regimes,
                    class_weight_fn=weight_fn)

        self.fitted = True
        return self.history

    # -- inference --------------------------------------------------------- #

    def predict(self, ds_new: xr.Dataset, entropy_unit: str = "fraction",
                batch_size: int = 131072) -> xr.Dataset:
        """Probabilistic and entropy maps for `ds_new`, held in memory."""
        return predict_probabilistic_maps(self.model, ds_new, self.scaler, self.device,
                                          self.config.features, entropy_unit=entropy_unit,
                                          batch_size=batch_size)

    def predict_to_store(self, ds_new: xr.Dataset, out_path, entropy_unit: str = "fraction",
                         batch_size: int = 131072, time_chunk: int = 12) -> Path:
        """Stream probabilistic inference for `ds_new` into a Zarr/NetCDF store."""
        return predict_to_store(self.model, ds_new, self.scaler, self.device,
                                self.config.features, out_path,
                                entropy_unit=entropy_unit, batch_size=batch_size,
                                time_chunk=time_chunk)

    # -- persistence ------------------------------------------------------- #

    def save(self, path: str | Path) -> Path:
        """
        Write weights, scaler statistics, architecture and history to `path`.

        The scaler is stored as plain arrays rather than as a pickled estimator,
        so the checkpoint stays readable across scikit-learn versions.
        """
        path = Path(path)
        make_dirs(path.parent)
        torch.save({"model_state": self.model.state_dict(),
                    "config": self.config.to_dict(),
                    "scaler": {"mean": self.scaler.mean_,
                               "scale": self.scaler.scale_,
                               "var": self.scaler.var_,
                               "n_samples_seen": self.scaler.n_samples_seen_},
                    "history": self.history},
                   path)
        LOGGER.info("Checkpoint written to %s", path)
        return path

    def load(self, path: str | Path) -> "BVBRegimeMLP":
        """Restore weights, scaler and history from a checkpoint written by `save`."""
        chk = torch.load(path, map_location=self.device, weights_only=False)

        self.config = ModelConfig.from_dict(chk["config"])
        self.model = BVBMLP(n_features=self.config.n_features,
                            n_regimes=self.config.n_regimes,
                            hidden=self.config.hidden,
                            rare_regimes=self.config.rare_regimes,
                            dropout=self.config.dropout).to(self.device)
        self.model.load_state_dict(chk["model_state"])

        stats = chk["scaler"]
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.asarray(stats["mean"])
        self.scaler.scale_ = np.asarray(stats["scale"])
        self.scaler.var_ = np.asarray(stats["var"])
        self.scaler.n_samples_seen_ = stats["n_samples_seen"]
        self.scaler.n_features_in_ = self.scaler.mean_.shape[0]

        self.history = chk.get("history", init_history())
        self.fitted = True
        LOGGER.info("Checkpoint loaded from %s", path)
        return self

    @classmethod
    def from_checkpoint(cls, path: str | Path, device=None) -> "BVBRegimeMLP":
        """Rebuild a fitted estimator straight from a checkpoint."""
        obj = cls.__new__(cls)
        obj.device = device if isinstance(device, torch.device) else resolve_device(device or "auto")
        obj.config = ModelConfig()
        obj.scaler = StandardScaler()
        obj.history = init_history()
        obj.model = None
        obj.fitted = False
        return obj.load(path)


# --------------------------------------------------------------------------- #
# Command-line interface
# --------------------------------------------------------------------------- #

def _int_tuple(text: str) -> tuple[int, ...]:
    """Parse '256,128,64' into (256, 128, 64)."""
    try:
        values = tuple(int(t) for t in text.replace(" ", "").split(",") if t)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid layer sizes: {text!r}") from exc
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError(f"Layer sizes must be positive integers: {text!r}")
    return values


def _str_list(text: str) -> list[str]:
    """Parse 'a,b,c' into ['a', 'b', 'c']."""
    values = [t for t in (s.strip() for s in text.split(",")) if t]
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated, non-empty list.")
    return values


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for `main`."""
    p = argparse.ArgumentParser(
        prog="pipeline.py",
        description="Train and apply an MLP classifier of BV(B) sea-level regimes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = p.add_argument_group("data")
    data.add_argument("-i", "--input", required=True,
                      help="Zarr store (or NetCDF file) with the BVB features and regime labels.")
    data.add_argument("--predict-input", default=None,
                      help="Dataset to run inference on. Defaults to --input.")
    data.add_argument("--features", type=_str_list, default=list(BVB_TERMS),
                      metavar="V1,V2,...", help="Feature variables, in model order.")
    data.add_argument("--label-var", default=LABEL_VAR, help="Regime label variable.")
    data.add_argument("--train-years", nargs=2, metavar=("START", "END"), default=None,
                      help="Inclusive year range used for training/validation, e.g. 2005 2011.")
    data.add_argument("--predict-years", nargs=2, metavar=("START", "END"), default=None,
                      help="Inclusive year range for inference. Defaults to the whole record.")
    data.add_argument("--time-stride", type=int, default=1,
                      help="Use every n-th month of the training record.")

    model = p.add_argument_group("model")
    model.add_argument("--n-regimes", type=int, default=15, help="Number of BV regimes (classes).")
    model.add_argument("--hidden", type=_int_tuple, default=(256, 128, 64, 32, 16),
                       metavar="H1,H2,...", help="Hidden layer sizes.")
    model.add_argument("--rare-regimes", dest="rare_regimes", action="store_true", default=True,
                       help="Use SiLU activations, better at resolving rare regimes.")
    model.add_argument("--no-rare-regimes", dest="rare_regimes", action="store_false",
                       help="Use GELU activations instead.")
    model.add_argument("--dropout", type=float, default=0.0, help="Dropout in the hidden blocks.")

    opt = p.add_argument_group("optimisation")
    opt.add_argument("-e", "--epochs", type=int, default=100, help="Maximum number of epochs.")
    opt.add_argument("-b", "--batch-size", type=int, default=8192, help="Samples per batch.")
    opt.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    opt.add_argument("--weight-decay", type=float, default=1e-5, help="Adam weight decay.")
    opt.add_argument("--train-frac", type=float, default=0.7,
                     help="Fraction of months used for training (the rest validates).")
    opt.add_argument("--patience", type=int, default=16, help="Early-stopping patience (epochs).")
    opt.add_argument("--min-delta", type=float, default=1e-4,
                     help="Minimum improvement that resets the early-stopping counter.")
    opt.add_argument("--lambda-entropy", type=float, default=0.25,
                     help="Weight of the entropy term in the scheduler/stopping metric.")
    opt.add_argument("--sched-factor", type=float, default=0.5, help="LR reduction factor.")
    opt.add_argument("--sched-patience", type=int, default=8, help="LR scheduler patience.")
    opt.add_argument("--sched-threshold", type=float, default=1e-3,
                     help="Minimum improvement counted by the LR scheduler.")
    opt.add_argument("--sched-cooldown", type=int, default=5,
                     help="Epochs to wait after an LR reduction.")
    opt.add_argument("--min-lr", type=float, default=1e-6, help="Learning-rate floor.")
    opt.add_argument("--class-weights", choices=("balanced", "curriculum", "none"),
                     default="balanced", help="Class-imbalance strategy.")
    opt.add_argument("--curriculum-warmup", type=int, default=10,
                     help="Epochs over which curriculum weights ramp to fully balanced.")

    inf = p.add_argument_group("inference")
    inf.add_argument("--entropy-unit", choices=("fraction", "percent"), default="fraction",
                     help="Entropy scaling in the output maps.")
    inf.add_argument("--predict-batch-size", type=int, default=131072,
                     help="Samples per inference forward pass.")
    inf.add_argument("--predict-time-chunk", type=int, default=12,
                     help="Months per inference block appended to the output store.")

    out = p.add_argument_group("output")
    out.add_argument("-o", "--outdir", default=f"{BASE_DIR}/outputs/nn",
                     help="Root directory for checkpoints, history and predictions.")
    out.add_argument("--tag", default=None,
                     help="Run name used for the output filenames. Derived from the "
                          "architecture and years when omitted.")
    out.add_argument("--pred-format", choices=("zarr", "nc"), default="zarr",
                     help="Format of the prediction store.")
    out.add_argument("--checkpoint", default=None,
                     help="Existing checkpoint to load (required for --mode predict).")
    out.add_argument("--overwrite", action="store_true",
                     help="Recompute and overwrite outputs that already exist.")

    run = p.add_argument_group("runtime")
    run.add_argument("--mode", choices=("train", "predict", "train-predict"),
                     default="train-predict", help="Which stages to run.")
    run.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda' or 'cuda:N'.")
    run.add_argument("--data-on-device", choices=("auto", "gpu", "cpu"), default="auto",
                     help="Where the resident sample tensors are staged.")
    run.add_argument("--seed", type=int, default=42, help="Random seed.")
    run.add_argument("-v", "--verbose", action="store_true", help="Debug-level logging.")

    return p


def default_tag(model_cfg: ModelConfig, years: tuple[str, str] | None) -> str:
    """Build a descriptive, filename-safe run name."""
    arch = "x".join(str(h) for h in model_cfg.hidden)
    span = f"_{years[0]}_{years[1]}" if years else ""
    return f"bvb_mlp_h{arch}_k{model_cfg.n_regimes}{span}"


def main(argv: list[str] | None = None) -> int:
    """
    Entry point: train the regime classifier and/or run probabilistic inference.

    Args:
        argv: Command-line arguments (defaults to `sys.argv[1:]`).

    Returns:
        Process exit status (0 on success).
    """
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    t_start = time.time()

    device = resolve_device(args.device)
    configure_runtime(args.seed, device)

    model_cfg = ModelConfig(features=args.features,
                            label_var=args.label_var,
                            n_regimes=args.n_regimes,
                            hidden=args.hidden,
                            rare_regimes=args.rare_regimes,
                            dropout=args.dropout)
    train_cfg = TrainConfig(epochs=args.epochs,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            train_frac=args.train_frac,
                            patience=args.patience,
                            min_delta=args.min_delta,
                            lambda_entropy=args.lambda_entropy,
                            sched_factor=args.sched_factor,
                            sched_patience=args.sched_patience,
                            sched_threshold=args.sched_threshold,
                            sched_cooldown=args.sched_cooldown,
                            min_lr=args.min_lr,
                            class_weights=args.class_weights,
                            curriculum_warmup=args.curriculum_warmup,
                            data_on_device=args.data_on_device,
                            seed=args.seed)

    train_years = tuple(args.train_years) if args.train_years else None
    predict_years = tuple(args.predict_years) if args.predict_years else None

    tag = args.tag or default_tag(model_cfg, train_years)
    outdir = make_dirs(args.outdir)
    model_dir = make_dirs(outdir / "models")
    pred_dir = make_dirs(outdir / "predictions")

    ckpt_path = Path(args.checkpoint) if args.checkpoint else model_dir / f"{tag}.pt"
    hist_path = model_dir / f"{tag}_history.csv"
    cfg_path = model_dir / f"{tag}_config.json"
    pred_path = pred_dir / f"{tag}_predictions.{args.pred_format}"

    LOGGER.info("Run tag: %s | mode: %s | outputs under %s", tag, args.mode, outdir)

    do_train = args.mode in ("train", "train-predict")
    do_predict = args.mode in ("predict", "train-predict")

    estimator: BVBRegimeMLP | None = None

    # -- training ---------------------------------------------------------- #
    if do_train:
        if ckpt_path.exists() and not args.overwrite:
            LOGGER.info("Checkpoint %s already exists - skipping training "
                        "(pass --overwrite to retrain).", ckpt_path)
        else:
            ds = open_dataset(args.input, model_cfg.features, model_cfg.label_var,
                              years=train_years, time_stride=args.time_stride)
            estimator = BVBRegimeMLP(model_cfg, device=device)
            estimator.fit(ds, train_cfg)
            estimator.save(ckpt_path)
            save_history(estimator.history, hist_path)
            cfg_path.write_text(json.dumps({"tag": tag,
                                            "input": str(args.input),
                                            "train_years": train_years,
                                            "time_stride": args.time_stride,
                                            "model": model_cfg.to_dict(),
                                            "training": asdict(train_cfg)},
                                           indent=2, default=str))
            LOGGER.info("Run configuration written to %s", cfg_path)
            ds.close()

    # -- inference --------------------------------------------------------- #
    if do_predict:
        if pred_path.exists() and not args.overwrite:
            LOGGER.info("Predictions %s already exist - skipping inference "
                        "(pass --overwrite to recompute).", pred_path)
        else:
            if estimator is None:
                if not ckpt_path.exists():
                    LOGGER.error("No checkpoint at %s; train first or pass --checkpoint.",
                                 ckpt_path)
                    return 2
                estimator = BVBRegimeMLP.from_checkpoint(ckpt_path, device=device)

            ds_new = open_dataset(args.predict_input or args.input,
                                  estimator.config.features, label_var=None,
                                  years=predict_years)
            estimator.predict_to_store(ds_new, pred_path,
                                       entropy_unit=args.entropy_unit,
                                       batch_size=args.predict_batch_size,
                                       time_chunk=args.predict_time_chunk)
            ds_new.close()

    LOGGER.info("Pipeline completed in %.1f s.", time.time() - t_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())

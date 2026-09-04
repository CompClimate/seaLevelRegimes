# Neural Network Inference for the Sea Level Regime Predictability


## Overview

Determining and predicting regimes of sea level variability using a Neural Network.

A multilayer perceptron (MLP) maps the nine barotropic vorticity budget (BVB)
terms of a grid point and month onto a **probability distribution over the BV
regimes**. The model never assigns a single regime outright, which is what makes
the results physically interpretable:

* **probabilistic maps** (`regime_prob`) tell *which* regimes are likely;
* **entropy maps** (`regime_entropy`) tell *how reliable* that assignment is —
  low entropy means one regime dominates (confident, stable classification),
  high entropy means the probabilities are spread across regimes (ambiguous or
  transitional dynamics).

Together they separate well-defined BV regimes from regions and times of
dynamical transition. `regime_pred` (the argmax) and `regime_confidence` (the
largest probability) are written alongside them for convenience.


## Layout

| File | Role |
| --- | --- |
| `pipeline.py` | The pipeline: data preparation, training, probabilistic inference. `python pipeline.py --help` is the complete option reference. |
| `pipeline.sh` | Slurm batch script. Sets up the environment and forwards **every** argument verbatim to `pipeline.py main()`. |
| `submit.sh` | Submitter. Consumes the job options, turns them into `sbatch` flags, and passes everything else through to the pipeline. |

The dependency chain is one-directional and each layer has one job:

```
submit.sh  ──sbatch──▶  pipeline.sh  ──python──▶  pipeline.py main()
(job resources)         (environment)             (the science)
```

Because `pipeline.sh` forwards its arguments untouched, `pipeline.py --help`
stays the single source of truth for what the pipeline accepts — there is no
second option list to keep in sync.


## Quick start

```bash
cd nn

# What would be submitted, without submitting it
./submit.sh --dry-run

# Train on the default record, then predict over the whole dataset
./submit.sh

# Watch the job
squeue -u $USER
tail -f dumps/nn/nn_<jobid>.out
```


## Running through Slurm

### Job options

Consumed by `submit.sh` itself; everything else is forwarded to `pipeline.py`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--account NAME` | `gfdl_o` | Slurm account |
| `--partition NAME` | `analysis`, or `gpu` when `--gpus > 0` | Slurm partition |
| `--time HH:MM:SS` | `12:00:00` | Wall-clock limit |
| `--mem SIZE` | `250G` | Memory per node |
| `--cpus N` | `8` | CPUs per task |
| `--gpus N` | `0` | GPUs; any value `> 0` switches to the GPU partition |
| `--gpu-type NAME` | `l40s` | GRES GPU type |
| `--job-name NAME` | derived from `--mode`/`--tag` | Slurm job name |
| `--logdir DIR` | `nn/dumps/nn` | Where the `.out`/`.err` logs go |
| `--constraint FEAT` | none | Slurm feature constraint |
| `--exclude NODES` | `an001,an002` | Nodes to avoid (see [Environment](#environment)) |
| `--env PATH` | `/work/lnd/ODRI/CONDA/conda_envs/nemi_env` | Conda environment to activate |
| `--base-dir DIR` | `/work/lnd/CM4X` | Data root the default input/output paths are built from |
| `--dry-run` | — | Print the `sbatch` command instead of submitting |
| `-h`, `--help` | — | Job options and their current defaults |

### Pipeline options

Forwarded to `pipeline.py`. The defaults below are the ones `submit.sh` sends —
edit the `OPT` block at the top of `submit.sh` to change them permanently. The
ones you will actually reach for:

| Option | Default | Meaning |
| --- | --- | --- |
| `--mode MODE` | `train-predict` | `train`, `predict`, or `train-predict` |
| `-i`, `--input PATH` | `<base-dir>/inputs/monthly_bvb_nn_features_num_labels.zarr` | Features + labels |
| `-o`, `--outdir DIR` | `<base-dir>/outputs/nn` | Root for all outputs |
| `--tag NAME` | derived from the architecture and years | Run name used in the filenames |
| `--train-years S E` | `2005 2011` | Inclusive year range for training/validation |
| `--predict-years S E` | whole record | Inclusive year range for inference |
| `--predict-input PATH` | same as `--input` | Predict on a *different* dataset |
| `--n-regimes N` | `15` | Number of BV regimes (classes) |
| `--hidden H1,H2,...` | `256,128,64,32,16` | Hidden layer sizes |
| `-e`, `--epochs N` | `100` | Maximum epochs (early stopping usually ends it sooner) |
| `-b`, `--batch-size N` | `8192` | Samples per batch |
| `--class-weights S` | `balanced` | `balanced`, `curriculum`, or `none` |
| `--entropy-unit U` | `fraction` | `fraction` → `[0, 1]`, `percent` → `[0, 100]` |
| `--checkpoint PATH` | derived from `--tag` | Checkpoint to load (needed for `--mode predict`) |
| `--overwrite` | — | Recompute outputs that already exist |
| `--device D` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N` |

Run `python pipeline.py --help` for the rest: learning rate and weight decay,
the train/validation fraction, early-stopping patience, the dual-criterion
scheduler settings, dropout, curriculum warm-up, inference chunk sizes, and the
random seed.

### Examples

```bash
# A larger network on the GPU partition
./submit.sh --gpus 1 --hidden 512,256,128,64,32 --batch-size 32768 --epochs 300

# Curriculum-weighted training over a specific period
./submit.sh --train-years 2005 2011 --class-weights curriculum --curriculum-warmup 15

# Inference only, from an existing checkpoint, on years the model never saw
./submit.sh --mode predict \
            --checkpoint /work/lnd/CM4X/outputs/nn/models/my_run.pt \
            --predict-years 2012 2014 --tag my_run_2012_2014

# Predict on a different dataset entirely
./submit.sh --mode predict --checkpoint <ckpt> \
            --predict-input /work/lnd/CM4X/BVB/other_features.zarr --tag other

# Bigger job, entropy in percent, retraining over an existing run
./submit.sh --cpus 16 --mem 400G --time 24:00:00 \
            --entropy-unit percent --tag my_run --overwrite

# Anything not listed above can be passed straight through after a bare --
./submit.sh --tag debug -- --verbose --seed 7
```

### Outputs

Everything lands under `--outdir`:

```
models/<tag>.pt                    weights + scaler statistics + architecture + history
models/<tag>_history.csv           per-epoch loss / entropy / accuracy / learning rate
models/<tag>_config.json           the full run configuration, for provenance
predictions/<tag>_predictions.zarr regime_prob, regime_entropy, regime_pred, regime_confidence
```

Runs are **idempotent**: an existing checkpoint or prediction store is skipped
with a log message rather than recomputed. Pass `--overwrite` to force the work.


## Running without Slurm

The pipeline is a plain script, so it runs anywhere the environment is available
— useful for short debugging runs on a login node or inside a notebook terminal:

```bash
conda activate /work/lnd/ODRI/CONDA/conda_envs/nemi_env
python pipeline.py --input <features.zarr> --outdir ./scratch \
                   --hidden 64,32 --epochs 5 --train-years 2005 2006 \
                   --tag debug --device cpu --verbose
```


## Using a trained model in a Jupyter notebook

This is the workflow for applying an already-trained model to a new dataset and
then analysing the result. Training stays on Slurm; the notebook only loads the
weights.

### 1. Point the notebook at the pipeline

Start the kernel from the same conda environment the job used, then:

```python
import sys

NN_DIR = "/home/Laique.Djeutchouang/DEVs/SLVP/seaLevelRegimes/nn"
if NN_DIR not in sys.path:
    sys.path.insert(0, NN_DIR)

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from pipeline import BVBRegimeMLP, open_dataset, BVB_TERMS, LABEL_VAR
```

### 2. Load the weights

`from_checkpoint` rebuilds the network, the feature scaler and the training
history from the `.pt` file alone — you do not need to restate the architecture.

```python
CKPT = "/work/lnd/CM4X/outputs/nn/models/my_run.pt"

model = BVBRegimeMLP.from_checkpoint(CKPT, device="cpu")   # or device="cuda"

print("features :", model.config.features)
print("hidden   :", model.config.hidden)
print("n_regimes:", model.config.n_regimes)
print("scaler mean:", np.round(model.scaler.mean_, 3))
```

`model.config.features` is authoritative: the new dataset must contain those
variables, and the model applies them in that order.

### 3. Inspect how training went

The history travels inside the checkpoint, so the curves are available without
opening the CSV.

```python
hist = pd.DataFrame(model.history).set_index("epoch")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
hist[["train_loss", "val_loss"]].plot(ax=axes[0], title="Loss")
hist[["train_entropy", "val_entropy"]].plot(ax=axes[1], title="Normalised entropy")
hist[["val_accuracy", "val_balanced_accuracy"]].plot(ax=axes[2], title="Validation accuracy")
for ax in axes:
    ax.set_xlabel("epoch")
fig.tight_layout()
```

Read `val_balanced_accuracy` rather than `val_accuracy` when the rare regimes
matter — it averages the per-regime skill instead of the per-cell skill.

### 4. Predict on a new dataset

`open_dataset` applies the same checks the job does: it verifies the variables
and dimensions, and optionally subsets by year.

```python
DATA = "/work/lnd/CM4X/BVB/other_features.zarr"

ds_new = open_dataset(DATA,
                      features=model.config.features,
                      label_var=None,             # labels are not needed for inference
                      years=("2012", "2014"))     # or None for the whole record

pred = model.predict(ds_new, entropy_unit="fraction")   # or "percent"
pred
```

`pred` is an `xr.Dataset` on the original grid, with land preserved as `NaN`:

| Variable | Dimensions | Meaning |
| --- | --- | --- |
| `regime_prob` | `(time, lat, lon, regime)` | Probability of each regime |
| `regime_entropy` | `(time, lat, lon)` | Normalised entropy, `0` = confident, `1` = maximally ambiguous |
| `regime_pred` | `(time, lat, lon)` | Most likely regime (argmax) |
| `regime_confidence` | `(time, lat, lon)` | Probability of that most likely regime |

> **Memory.** `model.predict` holds the whole result in memory, which is fine for
> a few years of a coarse grid. For a long record or a fine grid use
> `model.predict_to_store`, which streams the inference in blocks of months and
> appends them to a Zarr store:
>
> ```python
> model.predict_to_store(ds_new, "/work/lnd/CM4X/outputs/nn/predictions/other.zarr",
>                        entropy_unit="fraction", time_chunk=12)
> pred = xr.open_zarr("/work/lnd/CM4X/outputs/nn/predictions/other.zarr", chunks=None)
> ```
>
> This is the same routine the Slurm job uses. To reopen a store the job already
> wrote, skip straight to `xr.open_zarr` — no need to predict again.

### 5. Analyse the result

```python
# Time-mean probability of each regime, and the regime that dominates
prob_mean = pred["regime_prob"].mean("time")
ocean = prob_mean.notnull().any("regime")
dominant = prob_mean.fillna(-1.0).argmax("regime").where(ocean)

# Mean ambiguity, and where the classification is genuinely uncertain
mean_entropy = pred["regime_entropy"].mean("time")
ambiguous = pred["regime_entropy"] > 0.5
print(f"ambiguous cells: {float(ambiguous.mean()):.1%}")

# The regime field with the ambiguous cells masked out
confident_regime = pred["regime_pred"].where(pred["regime_entropy"] < 0.5)

# Expected regime occupancy through time (sums to 1; land is skipped by mean)
occupancy = pred["regime_prob"].mean(("lat", "lon"))

# The most dynamically ambiguous months in the record
ambiguity_ts = pred["regime_entropy"].mean(("lat", "lon"))
print(ambiguity_ts.sortby(ambiguity_ts, ascending=False).time.values[:5])
```

> **Land mask.** `argmax`/`idxmax` raise `ValueError: All-NaN slice encountered`
> on land cells, so mask first (`fillna(-1.0).argmax(...).where(ocean)` above).
> Reductions such as `mean` and `sum` skip `NaN` by default and need no such care.

If the new dataset happens to carry labels, scoring is direct:

```python
truth = xr.open_zarr(DATA, chunks=None)[LABEL_VAR].sel(time=pred.time)
valid = np.isfinite(truth.values) & np.isfinite(pred["regime_pred"].values)
print(f"accuracy: {(truth.values[valid] == pred['regime_pred'].values[valid]).mean():.4f}")
```

### 6. Visualise

```python
import cartopy.crs as ccrs

fig, axes = plt.subplots(1, 2, figsize=(14, 4.5),
                         subplot_kw={"projection": ccrs.Robinson(central_longitude=180)})

dominant.plot(ax=axes[0], transform=ccrs.PlateCarree(), cmap="tab20",
              cbar_kwargs={"label": "dominant BV regime"})
axes[0].set_title("Dominant regime (time mean)")

mean_entropy.plot(ax=axes[1], transform=ccrs.PlateCarree(), cmap="magma",
                  vmin=0, vmax=1, cbar_kwargs={"label": "normalised entropy"})
axes[1].set_title("Regime ambiguity (time-mean entropy)")

for ax in axes:
    ax.coastlines(linewidth=0.4)
fig.tight_layout()
```

The two panels are meant to be read together: the dominant-regime map is only
trustworthy where the entropy map is low, and the high-entropy regions are
themselves the interesting result — they mark transitional dynamics.

A single regime's probability field for one month:

```python
fig, ax = plt.subplots(figsize=(9, 4.5), subplot_kw={"projection": ccrs.PlateCarree()})
pred["regime_prob"].isel(time=0).sel(regime=2).plot(
    ax=ax, transform=ccrs.PlateCarree(), vmin=0, vmax=1, cmap="viridis",
    cbar_kwargs={"label": "P(regime 2)"})
ax.coastlines(linewidth=0.4)
```

Occupancy through time, and how the confidence is distributed:

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

occupancy.plot.line(x="time", ax=axes[0])
axes[0].set(title="Regime occupancy", ylabel="expected fraction of ocean cells")

entropy_values = pred["regime_entropy"].values
axes[1].hist(entropy_values[np.isfinite(entropy_values)], bins=40)
axes[1].set(title="Entropy distribution", xlabel="normalised entropy")
fig.tight_layout()
```

### 7. Save what you derived

```python
analysis = xr.Dataset({"dominant_regime": dominant, "mean_entropy": mean_entropy})
analysis.to_netcdf("/work/lnd/CM4X/outputs/nn/predictions/my_run_analysis.nc")
```


## Environment

`pipeline.sh` activates `$SLVP_CONDA_ENV` (default
`/work/lnd/ODRI/CONDA/conda_envs/nemi_env`), which must provide **pytorch**
alongside `xarray`, `zarr`, `scikit-learn` and `numpy`. Point it elsewhere with
`./submit.sh --env /path/to/env`.

Three environment variables are honoured, all exported automatically by
`submit.sh`:

| Variable | Purpose |
| --- | --- |
| `SLVP_NN_DIR` | Directory holding `pipeline.py` |
| `SLVP_CONDA_ENV` | Conda environment to activate |
| `SLVP_BASE_DIR` | Data root behind the default paths |

### Troubleshooting

* **`Illegal instruction (core dumped)`** — the job landed on one of the pre-AVX
  (2010-era) nodes in the heterogeneous `analysis` partition, where prebuilt
  PyTorch wheels abort. `submit.sh` excludes `an001,an002` by default; pass
  `--exclude ''` only if your build tolerates them.
* **Job fails with empty logs** — `--logdir` must be on a shared filesystem that
  the compute nodes can see. Node-local `/vftmp` will not work.
* **`Variables missing from ...`** — the input store does not carry the nine BVB
  terms under the expected names. Check `pipeline.BVB_TERMS` against
  `list(ds.data_vars)`, and override with `--features` if they differ.
* **Nothing happened and the job exited immediately** — the outputs already
  exist; the run was skipped. Add `--overwrite`, or use a fresh `--tag`.

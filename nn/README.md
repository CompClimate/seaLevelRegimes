# Neural Network Inference for the Sea Level Regime Predictability


## Overview

Determining and predicting regimes of sea level variability using a Neural Network.

A multilayer perceptron (MLP) maps the barotropic vorticity budget (BVB) terms
and the ocean-state variables of a grid point and month onto a **probability
distribution over the BV regimes**. The model never assigns a single regime
outright, which is what makes the results physically interpretable:

* **probabilistic maps** (`regime_prob`) tell *which* regimes are likely;
* **entropy maps** (`regime_entropy`) tell *how reliable* that assignment is —
  low entropy means one regime dominates (confident, stable classification),
  high entropy means the probabilities are spread across regimes (ambiguous or
  transitional dynamics).

Together they separate well-defined BV regimes from regions and times of
dynamical transition. `regime_pred` (the argmax) and `regime_confidence` (the
largest probability) are written alongside them for convenience.

The default feature set (`pipeline.BVB_TERMS`, 12 variables) is the nine BVB
terms plus three state variables:

```
beta_V  BPT  Mass_flux  eta_dt  Curl_dudt  Curl_taus  Curl_taub  Curl_Adv  Curl_diff
zos  tos  col_height
```

Regime IDs in the input store are **1-based** (`1..n_regimes`). The pipeline
shifts them down for `CrossEntropyLoss` and shifts them back on the way out, so
`regime_pred` keeps the same numbering as `bvb_regime` (`pipeline.LABEL_BASE`).


## Layout

| Path | Role |
| --- | --- |
| `pipeline.py` | The pipeline: data preparation, training, probabilistic inference. `python pipeline.py --help` is the complete option reference. |
| `pipeline.sh` | Slurm batch script. Activates the conda environment and forwards **every** argument verbatim to `pipeline.py main()`. |
| `submit.sh` | Submitter. Consumes the job options, turns them into `sbatch` flags, and passes everything else through to the pipeline. |
| `notebooks/analysis.ipynb` | Load a trained run, check it, and analyse the predictions. |
| `notebooks/run_pipeline.ipynb` | Drive the pipeline from a kernel — Python API, `main(argv)`, or `submit.sh`. |
| `dumps/nn/` | Slurm `.out`/`.err` logs, `nn_<jobid>.{out,err}`. |
| `archive/` | Superseded scripts and notebooks. |
| `environment.yml` | Conda/mamba specification for the runtime environment. |
| `requirements.txt` | The same dependencies as a pip fallback, for use inside an existing environment. |

The dependency chain is one-directional and each layer has one job:

```
submit.sh ──sbatch ──▶ pipeline.sh ──python ──▶ pipeline.py main()
(job resources)         (environment)             (the science)
```

Because `pipeline.sh` forwards its arguments untouched, `pipeline.py --help`
stays the single source of truth for what the pipeline accepts — there is no
second option list to keep in sync.


## Quick start

Create the environment first if you have not already — see
[Environment](#environment).

```bash
cd nn

# What would be submitted, without submitting it
./submit.sh --dry-run

# Train on the default record, then predict over the whole dataset
# (defaults: one H100, 32 cpus, 350G, 24h on the gpu-h100-h partition)
./submit.sh

# Watch the job
squeue -u $USER
tail -f dumps/nn/nn_<jobid>.out
```

A reference run — `--hidden 256,128,64,32,16 --epochs 300 --batch-size 16384
--class-weights curriculum --curriculum-warmup 25`, 12 features, 15 regimes,
58.1 M training samples staged on the GPU — trained to early stopping and wrote
its predictions in **~14 minutes** end to end on a single H100 NVL.


## Running through Slurm

### Job options

Consumed by `submit.sh` itself; everything else is forwarded to `pipeline.py`.

| Option | Default | Meaning |
| --- | --- | --- |
| `--account NAME` | `maikesgrp` | Slurm account |
| `--partition NAME` | `gpu-h100-h` | Slurm partition |
| `--time HH:MM:SS` | `24:00:00` | Wall-clock limit |
| `--mem SIZE` | `350G` | Memory per node |
| `--cpus N` | `32` | CPUs per task |
| `--gpus N` | `1` | GPUs; `0` runs CPU-only (pass `--partition` too) |
| `--gpu-type NAME` | `h100` | GRES GPU type → `--gres=gpu:<type>:<n>` |
| `--job-name NAME` | `BVB:NN:<mode>[:<tag>]` | Slurm job name |
| `--logdir DIR` | `nn/dumps/nn` | Where the `.out`/`.err` logs go |
| `--constraint FEAT` | none | Slurm feature constraint |
| `--exclude NODES` | none | Nodes to avoid |
| `--env PATH` | `/quobyte/maikesgrp/laique/CONDA/conda_envs/nemi_env` | Conda environment to activate |
| `--base-dir DIR` | `/group/maikesgrp/laique/PPAN/CM4X/NN4X` | Data root the default input/output paths are built from |
| `--dry-run` | — | Print the `sbatch` command instead of submitting |
| `-h`, `--help` | — | Job options and their current defaults |

`submit.sh` fails before queueing if `--input` does not exist, creates the log
and output directories, and exports `SLVP_NN_DIR`, `SLVP_CONDA_ENV` and
`SLVP_BASE_DIR` into the job.

### Pipeline options

Forwarded to `pipeline.py`. Two defaults differ between the two layers, because
`submit.sh` sends its own: edit the `OPT` block at the top of `submit.sh` to
change them permanently. The ones you will actually reach for:

| Option | `submit.sh` sends | `pipeline.py` default | Meaning |
| --- | --- | --- | --- |
| `--mode MODE` | `train-predict` | `train-predict` | `train`, `predict`, or `train-predict` |
| `-i`, `--input PATH` | `<base-dir>/inputs/global_NN4X_p25_monthly_features_nc15.zarr` | required | Features + labels |
| `-o`, `--outdir DIR` | `<base-dir>/outputs/nn` | `$SLVP_BASE_DIR/outputs/nn` | Root for all outputs |
| `--tag NAME` | derived | `bvb_mlp_h<arch>_k<K>_<y0>_<y1>` | Run name used in the filenames |
| `--train-years S E` | `2005 2011` | whole record | Inclusive year range for training/validation |
| `--predict-years S E` | whole record | whole record | Inclusive year range for inference |
| `--predict-input PATH` | same as `--input` | same as `--input` | Predict on a *different* dataset |
| `--features V1,V2,...` | `BVB_TERMS` | the 12 terms above | Feature variables, in model order |
| `--label-var NAME` | `bvb_regime` | `bvb_regime` | Regime label variable |
| `--time-stride N` | `1` | `1` | Use every n-th month of the training record |
| `--n-regimes N` | `15` | `15` | Number of BV regimes (classes) |
| `--hidden H1,H2,...` | `256,128,64,32,16` | `256,128,64,32,16` | Hidden layer sizes |
| `--rare-regimes` / `--no-rare-regimes` | on | on | SiLU activations (better on rare regimes) vs GELU |
| `-e`, `--epochs N` | `150` | `150` | Maximum epochs (early stopping usually ends it sooner) |
| `-b`, `--batch-size N` | `16384` | `16384` | Samples per batch |
| `--class-weights S` | `curriculum` | `balanced` | `balanced`, `curriculum`, or `none` |
| `--curriculum-warmup N` | `25` | `10` | Epochs over which curriculum weights ramp to balanced |
| `--entropy-unit U` | `fraction` | `fraction` | `fraction` → `[0, 1]`, `percent` → `[0, 100]` |
| `--pred-format F` | `zarr` | `zarr` | `zarr` or `nc` |
| `--checkpoint PATH` | derived from `--tag` | derived from `--tag` | Checkpoint to load (needed for `--mode predict`) |
| `--device D` | `auto` | `auto` | `auto`, `cpu`, `cuda`, `cuda:N` |
| `--data-on-device W` | `auto` | `auto` | Where the resident sample tensors live: `auto`, `gpu`, `cpu` |
| `--overwrite` | — | — | Recompute outputs that already exist |

Run `python pipeline.py --help` for the rest: learning rate and weight decay,
the train/validation fraction, early-stopping patience and `--min-delta`, the
entropy weight in the stopping metric (`--lambda-entropy`), the LR-scheduler
settings, dropout, the inference batch and time-chunk sizes, and the seed.

### Examples

```bash
# A larger network, longer schedule
./submit.sh --hidden 512,256,128,64,32,16 --batch-size 32768 --epochs 300

# GELU instead of SiLU, balanced class weights
./submit.sh --no-rare-regimes --class-weights balanced

# CPU-only run on the analysis partition
./submit.sh --gpus 0 --partition analysis --mem 250G --cpus 16

# Inference only, from an existing checkpoint, on years the model never saw
./submit.sh --mode predict \
            --checkpoint /group/maikesgrp/laique/PPAN/CM4X/NN4X/outputs/nn/models/bvb_mlp_h256x128x64x32x16_k15_2005_2011.pt \
            --predict-years 2012 2014 --tag my_run_2012_2014

# Predict on a different dataset entirely (e.g. the climatology store)
./submit.sh --mode predict --checkpoint <ckpt> \
            --predict-input /group/maikesgrp/laique/PPAN/CM4X/NN4X/inputs/global_NN4X_p25_clim_features_nc15.zarr \
            --tag clim

# Entropy in percent, retraining over an existing run
./submit.sh --entropy-unit percent --tag my_run --overwrite

# Anything not listed above can be passed straight through after a bare --
./submit.sh --tag debug -- --verbose --seed 7
```

### Outputs

Everything lands under `--outdir` (`<base-dir>/outputs/nn` by default):

```
models/<tag>.pt                    weights + scaler statistics + architecture + history
models/<tag>_history.csv           per-epoch loss / entropy / accuracy / lr / stopping metric
models/<tag>_config.json           the full run configuration, for provenance
predictions/<tag>_predictions.zarr regime_prob, regime_entropy, regime_pred, regime_confidence
```

with `<tag>` defaulting to `bvb_mlp_h<hidden>_k<n_regimes>_<start>_<end>`, e.g.
`bvb_mlp_h256x128x64x32x16_k15_2005_2011`. The runs currently on disk are:

| Tag | Features | Class weights | Notes |
| --- | --- | --- | --- |
| `bvb_mlp_h256x128x64x32_k15_2005_2011` | 9 BVB terms | curriculum | the run `analysis.ipynb` points at |
| `bvb_mlp_h512x256x128x64x32x16_k15_2005_2011` | 9 BVB terms | balanced | deeper, wider variant |
| `bvb_mlp_h256x128x64x32x16_k15_2005_2011` | 12 (BVB + `zos`/`tos`/`col_height`) | curriculum | latest, current default architecture |

`<tag>_history.csv` columns: `epoch, lr, train_loss, val_loss, train_entropy,
val_entropy, train_accuracy, val_accuracy, val_balanced_accuracy, metric` —
`metric` being the dual criterion (`val_loss + λ·val_entropy`) that drives both
the scheduler and early stopping.

Runs are **idempotent**: an existing checkpoint or prediction store is skipped
with a log message rather than recomputed. Pass `--overwrite` to force the work.


## Running without Slurm

The pipeline is a plain script, so it runs anywhere the environment is available
— useful for short debugging runs on a login node or inside a notebook terminal:

```bash
conda activate /quobyte/maikesgrp/laique/CONDA/conda_envs/nemi_env
python pipeline.py --input <features.zarr> --outdir ./scratch \
                   --hidden 64,32 --epochs 5 --train-years 2005 2006 \
                   --time-stride 3 --tag debug --device cpu --verbose
```


## Using a trained model in a Jupyter notebook

Two notebooks in [`notebooks/`](notebooks/), both of which fall back to a small
synthetic dataset when the real inputs are not on disk, so they run end to end
either way:

* **[`notebooks/analysis.ipynb`](notebooks/analysis.ipynb)** — the analysis
  workflow, runnable. It loads a checkpoint, opens (or computes) the prediction
  store, and works through sanity checks, the regime and entropy maps at three
  ambiguity thresholds, entropy calibration and per-regime skill, and
  area-weighted occupancy, persistence and ambiguity through time. It ends by
  writing its own artefacts to `outputs/notebook_outputs/`:
  `<tag>_analysis.nc`, `<tag>_per_regime_skill.csv`,
  `<tag>_accuracy_vs_entropy.csv`.
* **[`notebooks/run_pipeline.ipynb`](notebooks/run_pipeline.ipynb)** — the three
  ways to drive the pipeline from a kernel: the Python API (Route A),
  `pipeline.main(argv)` with an argument list (Route B), and `submit.sh` through
  `subprocess` (Route C, dry-run by default).

The rest of this section is the same material as a reference. Training stays on
Slurm; the notebook only loads the weights.

### 1. Point the notebook at the pipeline

Start the kernel from the same conda environment the job used, then:

```python
import sys
from pathlib import Path

# pipeline.py lives one level up from notebooks/
NN_DIR = Path("/home/djeutsch/Projects/seaLevelRegimes/nn")
if str(NN_DIR) not in sys.path:
    sys.path.insert(0, str(NN_DIR))

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
BASE = "/group/maikesgrp/laique/PPAN/CM4X/NN4X"
TAG  = "bvb_mlp_h256x128x64x32x16_k15_2005_2011"
CKPT = f"{BASE}/outputs/nn/models/{TAG}.pt"

model = BVBRegimeMLP.from_checkpoint(CKPT, device="cpu")   # or device="cuda"

print("features :", model.config.features)
print("hidden   :", model.config.hidden)
print("n_regimes:", model.config.n_regimes)
print("scaler mean:", np.round(model.scaler.mean_, 3))
```

`model.config.features` is authoritative: the new dataset must contain those
variables, and the model applies them in that order. The two feature sets in use
differ (9 vs 12 variables), so read it from the checkpoint rather than assuming.

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
DATA = f"{BASE}/inputs/global_NN4X_p25_clim_features_nc15.zarr"

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
| `regime_pred` | `(time, lat, lon)` | Most likely regime (argmax), numbered as `bvb_regime` |
| `regime_confidence` | `(time, lat, lon)` | Probability of that most likely regime |

> **Memory.** `model.predict` holds the whole result in memory. On the 1080×1440
> grid that is far too much for more than a couple of months — use
> `model.predict_to_store`, which streams the inference in blocks of months and
> appends them to a Zarr store:
>
> ```python
> model.predict_to_store(ds_new, f"{BASE}/outputs/nn/predictions/clim.zarr",
>                        entropy_unit="fraction", time_chunk=12)
> pred = xr.open_zarr(f"{BASE}/outputs/nn/predictions/clim.zarr", chunks=None)
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

# Expected regime occupancy through time, area-weighted by cos(lat)
w = np.cos(np.deg2rad(pred["lat"]))
occupancy = pred["regime_prob"].weighted(w).mean(("lat", "lon"))

# The most dynamically ambiguous months in the record
ambiguity_ts = pred["regime_entropy"].weighted(w).mean(("lat", "lon"))
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

Both fields use the same 1-based numbering, so no offset is needed here.

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
`analysis.ipynb` uses a discrete `K`-colour map so neighbouring regime indices
stay distinguishable and the colourbar ticks land on the regime numbers.

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
axes[0].set(title="Regime occupancy", ylabel="expected fraction of ocean area")

entropy_values = pred["regime_entropy"].values
axes[1].hist(entropy_values[np.isfinite(entropy_values)], bins=40)
axes[1].set(title="Entropy distribution", xlabel="normalised entropy")
fig.tight_layout()
```

### 7. Save what you derived

```python
analysis = xr.Dataset({"dominant_regime": dominant, "mean_entropy": mean_entropy})
analysis.to_netcdf(f"{BASE}/outputs/notebook_outputs/{TAG}_analysis.nc")
```


## Environment

The Slurm jobs run in the shared **`nemi_env`** at
`/quobyte/maikesgrp/laique/CONDA/conda_envs/nemi_env`, which carries the
`xarray`/`zarr`/`scikit-learn` stack *and* a CUDA build of PyTorch (torch
2.9.1+cu126, verified against the H100 NVL nodes). Nothing needs to be created
to reproduce the runs above — `./submit.sh` uses it by default.

### Creating your own

`environment.yml` and `requirements.txt` in this directory pin everything the
pipeline and the notebook analysis need. The conda route is the one to prefer:
`pytorch`, `netcdf4` and `cartopy` all wrap compiled libraries that conda
resolves as a set, and pip does not.

```bash
# From nn/ - creates an environment named slvp_nn
mamba env create -f environment.yml      # or: conda env create -f environment.yml
mamba activate slvp_nn
```

Environments used by Slurm jobs are conventionally kept on the shared
filesystem rather than in a home directory — they are large, and that is where
the other shared environments on this cluster live. Create it by path and point
the submitter at it:

```bash
mamba env create -f environment.yml -p /quobyte/maikesgrp/laique/CONDA/conda_envs/slvp_nn
./submit.sh --env /quobyte/maikesgrp/laique/CONDA/conda_envs/slvp_nn
```

`environment.yml` installs the **CPU** build of PyTorch. For the GPU partitions,
swap `pytorch` for `pytorch-gpu` in the file before creating the environment.

If you would rather add the dependencies to an environment you already have,
`requirements.txt` is the same list for pip:

```bash
mamba create -n slvp_nn python=3.12 -y
mamba activate slvp_nn
python -m pip install -r requirements.txt
```

On Linux the plain `torch` wheel is the CUDA build, ~2.5 GB with its NVIDIA
runtime dependencies. For a CPU-only environment, install it from the CPU index
first:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -r requirements.txt
```

Register the environment as a notebook kernel for the analysis workflow above:

```bash
python -m ipykernel install --user --name slvp_nn --display-name "Python (slvp_nn)"
```

### What is in it, and why

| Package | Needed for |
| --- | --- |
| `pytorch` | The MLP, the training loop and inference |
| `numpy` | Array maths throughout |
| `scikit-learn` | `StandardScaler`, fitted on the training months; the notebook's confusion matrix |
| `xarray` | The labelled arrays and the whole I/O layer |
| `zarr` | `open_zarr` / `to_zarr` — the default `--pred-format` |
| `dask` | `.chunk()`, applied before every `to_zarr` append |
| `netcdf4` | `open_dataset` / `to_netcdf` — `--pred-format nc`, and the notebook artefacts |
| `cftime` | Non-standard model calendars on the `time` axis |
| `pandas` | The training-history dataframes |
| `matplotlib`, `cartopy` | The maps and diagnostic figures |
| `jupyterlab`, `ipykernel` | The notebook workflow |
| `bottleneck` | Faster NaN-aware reductions over the land mask |

`zarr`, `dask` and `netcdf4` are reached through `xarray` rather than imported
directly, so they are easy to leave out — and the run then fails at the point
where it writes its results rather than at import. They are not optional.

### Pointing the job at an environment

`pipeline.sh` activates `$SLVP_CONDA_ENV` (default
`/quobyte/maikesgrp/laique/CONDA/conda_envs/nemi_env`). Any environment named
there must provide the core packages above; `--env` overrides it per submission,
and the job aborts with a clear message if the activation does not take.

Three environment variables are honoured, all exported automatically by
`submit.sh`:

| Variable | Purpose |
| --- | --- |
| `SLVP_NN_DIR` | Directory holding `pipeline.py` |
| `SLVP_CONDA_ENV` | Conda environment to activate |
| `SLVP_BASE_DIR` | Data root behind the default paths (also read by both notebooks) |

The notebooks additionally honour `SLVP_NB_SCRATCH` for where they write their
own artefacts (default `<base-dir>/outputs/notebook_outputs`).

### Troubleshooting

* **`could not activate conda env ...`** — `pipeline.sh` relaxes `errexit`
  around the `module`/`conda` hooks (they are not errexit-safe: the unload hook
  calls `conda deactivate`, which fails when `sbatch --export=ALL` carries an
  active env over) and then checks `CONDA_PREFIX` explicitly. If you see this,
  the environment path is wrong or unreadable from the compute node.
* **Job fails with empty logs** — `--logdir` must be on a shared filesystem that
  the compute nodes can see. Node-local scratch will not work.
* **`Illegal instruction (core dumped)`** — an old, pre-AVX CPU node; prebuilt
  PyTorch wheels abort there. Steer around it with `--exclude` or `--constraint`.
* **`WARNING: torch CUDA probe failed`** in the GPU-context block — the job got a
  GPU allocation but torch cannot see it; training silently falls back to CPU.
  Check the `nvidia-smi` line just above it.
* **`Variables missing from ...`** — the input store does not carry the expected
  feature names. Check `pipeline.BVB_TERMS` against `list(ds.data_vars)`, and
  override with `--features` if they differ (the 9-term runs on disk were made
  this way).
* **Nothing happened and the job exited immediately** — the outputs already
  exist; the run was skipped. Add `--overwrite`, or use a fresh `--tag`.

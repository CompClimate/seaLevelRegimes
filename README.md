# Predictability of Sea Level Regimes

Determining and predicting regimes of sea level variability as outlined with the vorticity budgets.

---

## Quick summary

Lightweight pipeline to create embeddings, clusterings, entropy maps and regime identification for sea-level / climate data. This repository implements a multi-step workflow (steps 1–4) of [NEMI](https://compclimate.github.io/NEMI/index.html) method (Sonnewald, 2023) and helpers to run locally or in parallel on an HPC (SLURM).

- Step 1 — embeddings: generate low-dimensional embeddings from input fields (UMAP).
- Step 2 — hierarchical/partition clustering: build clusterings from embeddings.
- Step 3 — entropy & baseline labels: compute entropy-like diagnostics per cluster.
- Step 4 — regimes identification: identify and export regimes using entropy + clustering.

The repo includes job launchers for SLURM and small utility modules under `src/`.

---

## Recommended project layout for data workspace

This project expects a structured data workspace. For example, my experiment is using the following layout:

```bash
nemis  
│   ├── CM4X-p125  
│   │   ├── figures  
│   │   ├── inputs  
│   │   └── outputs  
│   │       ├── dynamics  
│   │       │   ├── clusterings  
│   │       │   ├── embeddings  
│   │       │   ├── entropy  
│   │       │   └── regimes  
│   │       └── statics  
│   │           ├── clusterings  
│   │           ├── embeddings  
│   │           ├── entropy  
│   │           └── regimes  
│   └── CM4X-p25  
│       ├── figures  
│       ├── inputs  
│       └── outputs  
│           ├── dynamics  
│           │   ├── clusterings  
│           │   ├── embeddings  
│           │   ├── entropy  
│           │   └── regimes  
│           └── statics  
│               ├── clusterings  
│               ├── embeddings  
│               ├── entropy  
│               └── regimes
```
---

## Repository layout (important files)

- `step_1/` — embeddings pipeline
  - `embeddings.py` — main script to create embeddings
  - `run_embeddings.sh` — SLURM submission wrapper
  - `submit_jobs.sh` — helper to submit multiple jobs
  - `dumps/` — logs for python and slurm
- `step_2/` — clustering scripts & submit wrappers
- `step_3/` — entropy calculators & submit wrappers
- `step_4/` — regimes identifier & submit wrappers
- `src/` — reusable python functions (aux_func.py, nemi_func.py)
- `create_project_layout.py` — script to scaffold the recommended folder layout

---

## Prerequisites

- Linux (tested on HPC clusters)
- Python 3.10+ (uses conda in provided scripts)
- Conda environment (example name `proc_env`) with required packages:
  - numpy, scipy, pandas, xarray, scikit-learn, umap-learn, matplotlib, dask (optional), tensorflow (optional), joblib
- SLURM if running on cluster

Example minimal steps:

```bash
# Create environment (example)
conda create -n proc_env python=3.10 -y
conda activate proc_env
pip install -r requirements.txt   # create this file if desired
```

Note: SLURM scripts in `step_*` expect `module load conda` and `conda activate proc_env`.

---

## Create the recommended data workspace

The repository includes `create_project_layout.py`. Example to generate above data workspace:

```bash
python create_project_layout.py --root /path/to/nemis --models CM4X-p25 CM4X-p125
```

---

## How to run the pipeline

Local run (step 1):

```bash
# Example: run embeddings locally (python script accepts args: data_path member min_dist n_neighbors)
python step_1/embeddings.py /path/to/data.nc 1 0.1 15
```

Submit a SLURM job:

- Single job submission with one parameter combination
```bash
# Make sure script is executable
chmod +x step_1/run_embeddings.sh

# Submit (example args: MEMBER UMAP_MD UMAP_NN DATA_PATH)
sbatch step_1/run_embeddings.sh 0 0.1 15 /path/to/data.parquet
```
- Batch submission helpers are provided (e.g. `step_1/submit_jobs.sh`) and can be run as follows if you sufficient CPUs and RAMs for it can take hours and even days.
```bash
# Make sure script is executable
chmod +x step_1/submit_jobs.sh

# Submit (example args: MEMBER UMAP_MD UMAP_NN DATA_PATH)
cd step_1
./submit_jobs.sh
```
Steps 2–4 follow a similar pattern: inspect `run_*.sh` and `submit_*.sh` in each `step_*` directory.

---

## Logging and outputs

- Per-step logs are under `step_*/dumps/python/` and `step_*/dumps/slurm/`.
- Final processed outputs should be placed in your project `outputs/` tree following the recommended layout (clusterings, embeddings, entropy, regimes).
- The provided scripts write run-specific logs with member / UMAP parameters encoded in filenames.

---

## Troubleshooting / tips

- TensorFlow informational messages such as:
  "This TensorFlow binary is optimized to use available CPU instructions..."  
  are informational only and safe to ignore.
- If you hit `fatal: Authentication failed` when pushing to GitHub, use a PAT or SSH key:
  - git push -u origin your-branch
  - Configure credential helper or switch remote to `git@github.com:...`
- Ensure paths referenced in SLURM scripts (e.g. `conda activate proc_env`) match your environment.

---

## Contributing

- Use branches for features/fixes, push with:
  ```bash
  git checkout -b feat/your-feature
  git push -u origin feat/your-feature
  ```
- Add tests in `src/` if you refactor utility functions.
- Keep SLURM scripts generic; parameterize resources if adding new experiments.

---

## License & contact

- Preferred license file (e.g. MIT / BSD) coming (to be added at the repo root).
- For questions, open an Issue or contact the maintainer.

---

## Minimal checklist to get started

1. Create conda env and install deps.
2. Create project subdirs (use `create_project_layout.py` or manually go with `mkdir`).
3. Place input datasets under `inputs/`.
4. Run step 1 (embeddings) locally or via SLURM.
5. Run steps 2–4 in order, inspect `dumps/slurm` or `dumps/python` logs when debugging.

---

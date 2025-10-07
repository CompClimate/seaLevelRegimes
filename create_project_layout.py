# ...existing code...
#!/usr/bin/env python3
"""
Create the recommended project subdirectory tree from the CLI.

Example:
  python create_project_layout.py --root /quobyte/maikesgrp/laique/nemis --models CM4X-p25 CM4X-p125
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import argparse

# ensure local src/ is importable (keeps previous behaviour)
source = os.path.abspath("./")
if source not in sys.path:
    sys.path.insert(1, source)

from src import aux_func as af  # Importing the aux_func module
# ...existing code...

def create_project_structure(base_dir: str = "nemis", experiments: list[str] = ["CM4X-p25", "CM4X-p125"]):
    """
    Create a project folder structure for climate/ocean experiments.

    Parameters
    ----------
    base_dir : str
        The root project directory (default "nemis").
    experiments : list of str
        List of experiment names (defaults to ["CM4X-p25", "CM4X-p125"]).
    """
    # Define the sub-structure
    structure = {
        "figures": [],
        "inputs": [],
        "outputs": {
            "dynamics": ["clusterings", "embeddings", "entropy", "regimes"],
            "statics": ["clusterings", "embeddings", "entropy", "regimes"],
        },
    }

    # Recursive helper
    def make_dirs(path: str, tree: dict):
        for key, sub in tree.items():
            dir_path = os.path.join(path, key)
            os.makedirs(dir_path, exist_ok=True)
            if isinstance(sub, dict):
                make_dirs(dir_path, sub)
            elif isinstance(sub, list):
                for leaf in sub:
                    os.makedirs(os.path.join(dir_path, leaf), exist_ok=True)

    # Create base directory
    af.log_info(f"Creating base directory <{base_dir}> ...")
    os.makedirs(base_dir, exist_ok=True)

    # Create each experiment structure
    af.log_info(f"Creating and structuring experiment sub-folders: {experiments} ...")
    for exp in experiments:
        exp_dir = os.path.join(base_dir, exp)
        os.makedirs(exp_dir, exist_ok=True)
        make_dirs(exp_dir, structure)
    af.log_info(f"Project structure created under <{base_dir}>")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate recommended project subdirectories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", "-r", dest="root", default="nemis",
                   help="Root folder where experiment subdirs will be created.")
    p.add_argument("--models", "-m", dest="models", nargs="+", default=["CM4X-p25", "CM4X-p125"],
                   help="One or more model/experiment names (space separated).")
    p.add_argument("--dry-run", action="store_true", help="Show actions without creating directories.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base_dir = Path(args.root).resolve()
    models = args.models

    af.log_info(f"Requested root: {base_dir}")
    af.log_info(f"Models: {models}")
    if args.dry_run:
        print("Dry run: no directories will be created.")
        # Print the planned layout for user confirmation
        for m in models:
            print(f"{base_dir}/{m}/(figures, inputs, outputs/{{dynamics,statics}}/...)")
        return 0

    try:
        create_project_structure(base_dir=str(base_dir), experiments=models)
    except Exception as exc:
        af.log_info(f"Failed to create project structure: {exc}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


import os
import sys


# Add path to local modules and import them
source = os.path.abspath('./')
if source not in sys.path:
    sys.path.insert(1, source)

from src import aux_func as af # Importing the aux_func module


def create_project_structure(base_dir:str="nemis", experiments:list[str]=["CM4X-p25", "CM4X-p125"]):
    """
    Create a project folder structure for climate/ocean experiments.

    Parameters
    ----------
    base_dir : str
        The root project directory (default "nemis").
    experiments : list of str
        List of experiment names defaults to ["CM4X-p25", "CM4X-p125"]).
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
    def make_dirs(path, tree):
        for key, sub in tree.items():
            dir_path = os.path.join(path, key)
            os.makedirs(dir_path, exist_ok=True)
            if isinstance(sub, dict):
                make_dirs(dir_path, sub)
            elif isinstance(sub, list):
                for leaf in sub:
                    os.makedirs(os.path.join(dir_path, leaf), exist_ok=True)

    # Create base directory
    print()
    af.log_info(f"Creating base directory ...")
    os.makedirs(base_dir, exist_ok=True)

    # Create each experiment structure
    af.log_info(f"Creating and structuring experiment sub-folder: {experiments} ...")
    for exp in experiments:
        exp_dir = os.path.join(base_dir, exp)
        os.makedirs(exp_dir, exist_ok=True)
        make_dirs(exp_dir, structure)
    af.log_info(f"Project structure created under <{base_dir}>")
    print()


if __name__ == "__main__":
    # Parse command-line arguments
    base_dir = sys.argv[1]
    if len(sys.argv) > 2:
        experiments = [sys.argv[i] for i in range(2, len(sys.argv))]
    else:
        experiments = ["CM4X-p25", "CM4X-p125"]
    
    # Create the project structure
    create_project_structure(base_dir=base_dir,  experiments=experiments)


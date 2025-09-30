#!/bin/bash

# Usage: ./rename_files.sh /path/to/directory
TARGET_DIR=$1

if [ -z "$TARGET_DIR" ]; then
  echo "Please provide a target directory."
  exit 1
fi

cd "$TARGET_DIR" || exit

for file in *th_ensemble_min_dist_*_n_neighbors_*_nclusters_*_hclust_neighbors_*.npy; do
  if [[ -f "$file" ]]; then
    # Extract values using regex
    if [[ $file =~ ([0-9]+)th_ensemble_min_dist_([0-9.]+)_n_neighbors_([0-9]+)_nclusters_([0-9]+)_hclust_neighbors_([0-9]+).npy ]]; then
      ens="${BASH_REMATCH[1]}"
      md="${BASH_REMATCH[2]}"
      nn="${BASH_REMATCH[3]}"
      nc="${BASH_REMATCH[4]}"
      hclustn="${BASH_REMATCH[5]}"

      newname="emb_${ens}th_ensemble_md_${md}_nn_${nn}_nc_${nc}_hclustn_${hclustn}.npy"

      echo "Renaming $file → $newname"
      mv "$file" "$newname"
    fi
  fi
done
echo
echo "Renaming completed."
echo

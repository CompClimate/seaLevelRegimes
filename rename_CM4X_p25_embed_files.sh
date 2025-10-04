#!/bin/bash

# Usage: ./rename_files.sh /path/to/directory
DIR=$1

echo
if [ -z "$DIR" ]; then
  echo "Please provide a directory path."
  exit 1
fi

cd "$DIR" || exit

for file in *ensemble_min_dist_*_n_neighbors_*.npy; do
  if [[ -f "$file" ]]; then
    # Extract the parts we need
    prefix=$(echo "$file" | cut -d'_' -f1)      # e.g. 01th
    md=$(echo "$file" | sed -E 's/.*min_dist_([0-9.]+)_n_neighbors_.*/\1/')
    nn=$(echo "$file" | sed -E 's/.*n_neighbors_([0-9]+).*/\1/')

    # Build new filename
    new_name="emb_${prefix}_ensemble_md_${md}_nn_${nn}.npy"

    echo "Renaming $file -> $new_name"
    mv "$file" "$new_name"
  fi
done
echo
echo "Renaming completed."
echo
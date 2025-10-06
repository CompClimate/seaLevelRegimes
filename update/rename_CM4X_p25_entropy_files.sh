#!/bin/bash

# Usage: ./rename_entropy.sh /path/to/directory
DIR=$1

if [ -z "$DIR" ]; then
  echo "Usage: $0 /path/to/directory"
  exit 1
fi

cd "$DIR" || { echo "Directory not found"; exit 1; }

for file in entropy_n_neighbors_*_min_dist_*_nclusters_*.npy; do
  # Skip if no match
  [ -e "$file" ] || continue

  # Extract values
  nn=$(echo "$file" | sed -E 's/.*n_neighbors_([0-9]+)_min_dist_.*/\1/')
  md=$(echo "$file" | sed -E 's/.*min_dist_([0-9.]+)_nclusters_.*/\1/')
  nc=$(echo "$file" | sed -E 's/.*nclusters_([0-9]+)\.npy/\1/')

  # Construct new filename
  newname="entropy_emb_nn_${nn}_md_${md}_nc_${nc}.npy"

  echo "Renaming: $file → $newname"
  mv "$file" "$newname"
done
echo
echo "Renaming completed."
echo

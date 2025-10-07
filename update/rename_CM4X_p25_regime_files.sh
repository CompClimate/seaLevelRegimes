#!/bin/bash

# Usage: ./rename_files.sh /path/to/directory
DIR=$1
echo
if [ -z "$DIR" ]; then
  echo "Usage: $0 /path/to/directory"
  exit 1
fi

cd "$DIR" || { echo "Directory not found"; echo; exit 1; }
echo "Renaming files in directory: $DIR"
echo
for file in nemi_nclusters_*_mini_dist_*_n_neighbors_*.nc; do
  # Skip if no match
  [ -e "$file" ] || continue

  # Extract values using regex-friendly bash substitution
  nc=$(echo "$file" | sed -E 's/.*nclusters_([0-9]+)_mini_dist_.*/\1/')
  md=$(echo "$file" | sed -E 's/.*mini_dist_([0-9.]+)_n_neighbors_.*/\1/')
  nn=$(echo "$file" | sed -E 's/.*n_neighbors_([0-9]+)\.nc/\1/')

  # Construct new filename
  newname="regimes_from_md_${md}_nn_${nn}_nc_${nc}.nc"

  echo "Renaming: $file → $newname"
  mv "$file" "$newname"
done
echo
echo "Renaming completed."
echo
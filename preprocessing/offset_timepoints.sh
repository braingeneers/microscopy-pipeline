#!/bin/bash

# Usage: ./offset_timepoints.sh <folder> <offset>
# Example: ./offset_timepoints.sh ./data 5
# Example: ./offset_timepoints.sh ./data -3

FOLDER="$1"
OFFSET="$2"

if [[ -z "$FOLDER" || -z "$OFFSET" ]]; then
    echo "Usage: $0 <folder> <offset>"
    exit 1
fi

shopt -s nullglob

# Create an array to store files with their N values
declare -a files_with_n=()

# First pass: collect files and extract N values
for file in "$FOLDER"/*_zs*; do
    basename=$(basename "$file")
    if [[ "$basename" =~ ^([0-9]+)_zs(.*)$ ]]; then
        N="${BASH_REMATCH[1]}"
        files_with_n+=("$N:$file")
    fi
done

# Sort the array based on offset direction
if [[ $OFFSET -ge 0 ]]; then
    # Positive offset: sort in reverse order (highest N first)
    IFS=$'\n' sorted_files=($(sort -t: -k1,1nr <<< "${files_with_n[*]}"))
    echo "Processing files in reverse order (highest N first) for positive offset"
else
    # Negative offset: sort in forward order (lowest N first)
    IFS=$'\n' sorted_files=($(sort -t: -k1,1n <<< "${files_with_n[*]}"))
    echo "Processing files in forward order (lowest N first) for negative offset"
fi
unset IFS

# Second pass: rename files in appropriate order
for entry in "${sorted_files[@]}"; do
    N="${entry%%:*}"
    file="${entry#*:}"
    basename=$(basename "$file")
    if [[ "$basename" =~ ^([0-9]+)_zs(.*)$ ]]; then
        rest="${BASH_REMATCH[2]}"
        NEW_N=$((N + OFFSET))
        newname="${NEW_N}_zs${rest}"
        echo "Renaming: $basename -> $newname"
        mv "$file" "$FOLDER/$newname"
    fi
done
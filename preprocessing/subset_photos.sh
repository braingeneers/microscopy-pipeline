#!/bin/bash

# Check if correct number of arguments provided
if [[ $# -ne 5 ]]; then
    echo "Usage: $0 <photo_directory> <destination_directory> <minimum> <maximum> <increment>"
    echo "Example: $0 /path/to/photos /path/to/subset 0 100 10"
    echo "Example: $0 \"d:\\Documents\\photos\" \"d:\\Documents\\subset\" 5 50 5"
    exit 1
fi

# Set directories and parameters from command line arguments
PHOTO_DIR="$1"
DEST_DIR="$2"
minimum=$3
maximum=$4
increment=$5

# Check if the photo directory exists
if [[ ! -d "$PHOTO_DIR" ]]; then
    echo "Error: Photo directory '$PHOTO_DIR' does not exist"
    exit 1
fi

echo "Processing photos in: $PHOTO_DIR"
echo "Destination directory: $DEST_DIR"
echo "Range: $minimum to $maximum with increment $increment"

# Iterate through all files in the directory and rename '+-' to '-'
# echo "Renaming files with '+- ' to '-'..."
# for file in "$PHOTO_DIR"/*; do
#     if [[ "$(basename "$file")" == *"+-"* ]]; then
#         newfile="${file//+\-/-}"
#         mv "$file" "$newfile"
#         echo "  Renamed: $(basename "$file") -> $(basename "$newfile")"
#     fi
# done

# Create the destination folder if it doesn't exist
mkdir -p "$DEST_DIR"
echo "Created destination directory: $DEST_DIR"

echo "Copying subset of photos..."
copied_count=0

for ((i=minimum; i<=maximum; i+=increment)); do
    echo "Processing files with index: $i"
    for file in "$PHOTO_DIR"/*; do
        filename=$(basename "$file")
        if [[ "$filename" =~ .*_zs\+${i}\.png$ ]]; then
            cp "$file" "$DEST_DIR/"
            echo "  Copied: $filename"
            ((copied_count++))
        fi
    done
done

echo "Completed! Copied $copied_count files to $DEST_DIR"
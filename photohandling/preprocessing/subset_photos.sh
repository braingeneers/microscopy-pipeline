#!/bin/bash

# Check if correct number of arguments provided
# Defaults
TIMEPOINTINCREMENT=1
MINTIMEPOINT=""
MAXTIMEPOINT=""
 
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)               PHOTO_DIR="$2";           shift 2 ;;
        --output)              DEST_DIR="$2";             shift 2 ;;
        --minzstack)           minimum="$2";              shift 2 ;;
        --maxzstack)           maximum="$2";              shift 2 ;;
        --zstackincrement)     increment="$2";            shift 2 ;;
        --mintimepoint)        MINTIMEPOINT="$2";         shift 2 ;;
        --maxtimepoint)        MAXTIMEPOINT="$2";         shift 2 ;;
        --timepointincrement)  TIMEPOINTINCREMENT="$2";   shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --input DIR --output DIR --minzstack N --maxzstack N --zstackincrement N [--mintimepoint N] [--maxtimepoint N] [--timepointincrement N]"
            exit 1 ;;
    esac
done
 
# Check required arguments
if [[ -z "$PHOTO_DIR" || -z "$DEST_DIR" || -z "$minimum" || -z "$maximum" || -z "$increment" ]]; then
    echo "Usage: $0 --input DIR --output DIR --minzstack N --maxzstack N --zstackincrement N [--mintimepoint N] [--maxtimepoint N] [--timepointincrement N]"
    exit 1
fi
 
# Check if the photo directory exists
if [[ ! -d "$PHOTO_DIR" ]]; then
    echo "Error: Photo directory '$PHOTO_DIR' does not exist"
    exit 1
fi
 
echo "Processing photos in: $PHOTO_DIR"
echo "Destination directory: $DEST_DIR"
echo "Z-stack range: $minimum to $maximum with increment $increment"
if [[ -n "$MINTIMEPOINT" || -n "$MAXTIMEPOINT" ]]; then
    echo "Timepoint range: ${MINTIMEPOINT:-any} to ${MAXTIMEPOINT:-any} with increment $TIMEPOINTINCREMENT"
fi
 
# Create the destination folder if it doesn't exist
mkdir -p "$DEST_DIR"
echo "Created destination directory: $DEST_DIR"
 
echo "Copying subset of photos..."
copied_count=0
 
for ((i=minimum; i<=maximum; i+=increment)); do
    echo "Processing files with z-stack index: $i"
    for file in "$PHOTO_DIR"/*; do
        filename=$(basename "$file")
        if [[ "$filename" =~ ^([0-9]+)_zs\+${i}\.png$ ]]; then
            tp=$((10#${BASH_REMATCH[1]}))
 
            # Check timepoint bounds if specified
            if [[ -n "$MINTIMEPOINT" && $tp -lt $MINTIMEPOINT ]]; then continue; fi
            if [[ -n "$MAXTIMEPOINT" && $tp -gt $MAXTIMEPOINT ]]; then continue; fi
 
            # Check timepoint increment if a min was given
            if [[ -n "$MINTIMEPOINT" ]]; then
                if (( (tp - MINTIMEPOINT) % TIMEPOINTINCREMENT != 0 )); then continue; fi
            fi
 
            cp "$file" "$DEST_DIR/"
            echo "  Copied: $filename"
            ((copied_count++))
        fi
    done
done
 
echo "Completed! Copied $copied_count files to $DEST_DIR"
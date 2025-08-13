#!/bin/bash

INPUT_DIR="capture"
OUTPUT_DIR="capture_data"
FILTER="wlan.fc.type == 2"

mkdir -p "$OUTPUT_DIR"

# Loop over subdirectories
find "$INPUT_DIR" -type f -name "virtual_ap.pcap" | while read -r INPUT_FILE; do
    # Determine relative subdirectory
    REL_PATH="${INPUT_FILE#$INPUT_DIR/}"
    SUBDIR=$(dirname "$REL_PATH")

    # Create corresponding output subdir
    OUT_SUBDIR="$OUTPUT_DIR/$SUBDIR"
    mkdir -p "$OUT_SUBDIR"

    # Set output file path
    OUTPUT_FILE="$OUT_SUBDIR/virtual_ap.pcap"

    echo "Processing $INPUT_FILE -> $OUTPUT_FILE"
    tshark -r "$INPUT_FILE" -Y "$FILTER" -w "$OUTPUT_FILE"
done

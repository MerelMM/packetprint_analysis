#!/bin/bash
# This script was developed with assistance from ChatGPT (OpenAI) and Github Copilot
# Final implementation and adaptation by Merel Haenaets.
# Default input/output directories
INPUT_DIR="${INPUT_DIR:-capture}"
OUTPUT_DIR="${OUTPUT_DIR:-capture_data}"

# Device and AP MAC addresses
MAC1="fa:97:c8:94:78:8d"
MAC2="dc:4e:f4:0a:42:6f"

# Combined filter:
# - Between MAC1 and MAC2
# - 802.11 data frames only (type 2)
# - Exclude subtype 0x24 (Null function)
FILTER="(wlan.sa == $MAC1 && wlan.da == $MAC2 || wlan.sa == $MAC2 && wlan.da == $MAC1) && wlan.fc.type == 2 && wlan.fc.type_subtype != 0x24"

LOG_FILE="filter_data_frames_output.log"
: > "$LOG_FILE"  # Clear previous log file

# Loop over all virtual_ap.pcap files
find "$INPUT_DIR" -type f -name "virtual_ap.pcap" | while read -r INPUT_FILE; do
    REL_PATH="${INPUT_FILE#$INPUT_DIR/}"
    SUBDIR=$(dirname "$REL_PATH")
    OUT_SUBDIR="$OUTPUT_DIR/$SUBDIR"
    mkdir -p "$OUT_SUBDIR"

    OUTPUT_FILE="$OUT_SUBDIR/virtual_ap_filtered.pcap"

    echo "Processing: $INPUT_FILE -> $OUTPUT_FILE" | tee -a "$LOG_FILE"

    tshark -r "$INPUT_FILE" -Y "$FILTER" -w "$OUTPUT_FILE" 2>>"$LOG_FILE"

    FRAME_COUNT=$(tshark -r "$OUTPUT_FILE" -T fields -e frame.number 2>/dev/null | wc -l)
    echo "Filtered frame count: $FRAME_COUNT" | tee -a "$LOG_FILE"

    if [ "$FRAME_COUNT" -eq 0 ]; then
        echo "No frames found after filtering for $INPUT_FILE. Removing empty output." | tee -a "$LOG_FILE"
        rm -f "$OUTPUT_FILE"
    fi
done

#!/bin/bash

# Configuration (must be passed in)
BASE_DIR="$BASE_DIR"
OUTPUT_FILE="all_packet_timings.txt"
> "$OUTPUT_FILE"

echo "Extracting relative timestamps from pcap files..."

for dir in "$BASE_DIR"/*; do
    pcap_file="$dir/virtual_ap_filtered.pcap"
    if [[ -f "$pcap_file" ]]; then
        echo "Processing: $pcap_file"

        tshark -r "$pcap_file" -T fields -e frame.time_relative > "$dir/packet_timings.txt"

        awk -v name="$(basename "$dir")" '{print name, $0}' "$dir/packet_timings.txt" >> "$OUTPUT_FILE"
    fi
done

echo "All per-session timestamps saved. Combined output: $OUTPUT_FILE"


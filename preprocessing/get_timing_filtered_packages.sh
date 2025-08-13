#!/bin/bash

# Base directory containing all capture folders
BASE_DIR="capture_data"

# Output file where we collect timings for each folder
OUTPUT_FILE="all_packet_timings.txt"
> "$OUTPUT_FILE"  # clear or create the file

# Iterate over all subdirectories
for dir in "$BASE_DIR"/*; do
    pcap_file="$dir/virtual_ap_filtered.pcap"
    if [[ -f "$pcap_file" ]]; then
        echo "Processing $pcap_file"
        
        # Extract relative timestamps of all packets (seconds since beginning of capture)
        tshark -r "$pcap_file" -T fields -e frame.time_relative > "$dir/packet_timings.txt"
        
        # Optionally, append results to a global file with folder name
        awk -v name="$(basename "$dir")" '{print name, $0}' "$dir/packet_timings.txt" >> "$OUTPUT_FILE"
    fi
done

echo "Done. Per-directory timings saved as packet_timings.txt, combined output in $OUTPUT_FILE"

#!/bin/bash

# Ensure required tool is installed
if ! command -v tshark &> /dev/null; then
    echo "tshark not found. Please install Wireshark/tshark."
    exit 1
fi

# Loop over all virtual_ap_filtered.pcap files
find capture_data/ -type f -name "virtual_ap_filtered.pcap" | while read -r file; do
    echo "Processing $file..."

    first_time=$(tshark -r "$file" -T fields -e frame.time_epoch -c 1 2>/dev/null)
    last_time=$(tshark -r "$file" -T fields -e frame.time_epoch | tail -n 1 2>/dev/null)

    if [[ -z "$first_time" || -z "$last_time" ]]; then
        echo "  Skipping (empty or unreadable)"
        continue
    fi

    # Calculate duration
    duration=$(echo "$last_time - $first_time" | bc -l)

    # Output to console
    echo "  First packet: $first_time"
    echo "  Last packet:  $last_time"
    echo "  Duration:     $duration seconds"

    # Save to file
    folder=$(dirname "$file")
    echo "$first_time $last_time $duration" > "$folder/capture_time_range.txt"
done

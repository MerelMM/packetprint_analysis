#!/bin/bash
# This script was developed with assistance from ChatGPT (OpenAI) and Github Copilot
# Final implementation and adaptation by Merel Haenaets.
MAC1="fa:97:c8:94:78:8d"
MAC2="dc:4e:f4:0a:42:6f"
LOG_FILE="filter_mac_pairs_output.log"

# Empty the log file at the start
: > "$LOG_FILE"

for dir in "${OUTPUT_DIR:-capture_data}"/*/; do
    PCAP_FILE="${dir}virtual_ap.pcap"
    OUTPUT_FILE="${dir}virtual_ap_filtered.pcap"

    if [ ! -f "$PCAP_FILE" ]; then
        echo "❌ Skipping $dir — virtual_ap.pcap not found" | tee -a "$LOG_FILE"
        continue
    fi

    echo "🔍 Processing $PCAP_FILE..." | tee -a "$LOG_FILE"

    # Apply the MAC address filter and write output
    tshark -r "$PCAP_FILE" \
        -Y "(wlan.sa == $MAC1 && wlan.da == $MAC2) || (wlan.sa == $MAC2 && wlan.da == $MAC1)" \
        -w "$OUTPUT_FILE" 2>/dev/null

    # Count frames in filtered output
    FRAME_COUNT=$(tshark -r "$OUTPUT_FILE" -T fields -e frame.number | wc -l)

    echo "➡️  $FRAME_COUNT frames between $MAC1 and $MAC2" | tee -a "$LOG_FILE"

    if [ "$FRAME_COUNT" -eq 0 ]; then
        echo "⚠️  Warning: No frames found in $PCAP_FILE" | tee -a "$LOG_FILE"
        rm -f "$OUTPUT_FILE"  # Optional: remove empty output
    fi
done

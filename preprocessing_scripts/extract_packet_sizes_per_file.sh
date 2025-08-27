#!/bin/bash

# Configuration (must be passed in by the caller)
ROOT_DIR="$ROOT_DIR"
PCAP_NAME="${PCAP_NAME:-virtual_ap_filtered.pcap}"
OUTPUT_NAME="packet_sizes.txt"
AP_MAC="$AP_MAC"
DEVICE_MAC="$DEVICE_MAC"
WPA2_OVERHEAD="${WPA2_OVERHEAD:-16}"

echo "Extracting packet sizes from filtered pcap files..."

find "$ROOT_DIR" -type f -name "$PCAP_NAME" | while read -r PCAP; do
    DIR=$(dirname "$PCAP")
    OUTPUT_FILE="$DIR/$OUTPUT_NAME"

    echo "Processing: $PCAP"
    > "$OUTPUT_FILE"

    tshark -r "$PCAP" \
        -Y "wlan.fc.type == 2 && (wlan.sa == $AP_MAC || wlan.sa == $DEVICE_MAC)" \
        -T fields -e wlan.sa -e frame.len |
    while read -r SRC SIZE; do
        SIZE=$((SIZE - WPA2_OVERHEAD))
        if (( SIZE > 0 )); then
            if [[ "$SRC" == "$DEVICE_MAC" ]]; then
                echo "$SIZE" >> "$OUTPUT_FILE"
            elif [[ "$SRC" == "$AP_MAC" ]]; then
                echo "-$SIZE" >> "$OUTPUT_FILE"
            fi
        fi
    done

    echo "Saved to: $OUTPUT_FILE"
done

echo "Done processing all pcap files."

#!/bin/bash

# Configuration
ROOT_DIR="../code/capture_data"
OUTPUT_FILE="all_packet_sizes.txt"
AP_MAC="dc:4e:f4:0a:42:6f"
DEVICE_MAC="fa:97:c8:94:78:8d"
WPA2_OVERHEAD=16

# Reset output file
> "$OUTPUT_FILE"

echo "📡 Extracting packet sizes from PCAPs..."

find "$ROOT_DIR" -type f -name "virtual_ap_filtered.pcap" | while read -r PCAP; do
    echo "Processing: $PCAP"

    # Extract relevant packet info with tshark
    # In wireshark you can filter all Data frame using below filter (Type =_2_) _wlan_._fc_._type == 2_ https://mrncciew.com/2014/10/13/cwap-802-11-data-frame-types/
    tshark -r "$PCAP" \
        -Y "wlan.fc.type == 2 && (wlan.sa == $AP_MAC || wlan.sa == $DEVICE_MAC)" \
        -T fields -e wlan.sa -e frame.len |
    while read -r SRC SIZE; do
        # Apply WPA2 normalization
        SIZE=$((SIZE - WPA2_OVERHEAD))
        if (( SIZE > 0 )); then
            if [[ "$SRC" == "$DEVICE_MAC" ]]; then
                echo "outbound $SIZE" >> "$OUTPUT_FILE"
            elif [[ "$SRC" == "$AP_MAC" ]]; then
                echo "inbound $SIZE" >> "$OUTPUT_FILE"
            fi
        fi
    done
done

echo "✅ Done. Results saved to $OUTPUT_FILE"

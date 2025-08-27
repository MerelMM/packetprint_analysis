#!/bin/bash

set -e  # Exit on error

# Absolute path to this script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Shared config (can be overridden when calling the script)
INPUT_DIR="${INPUT_DIR:-./capture_data}"
OUTPUT_DIR="${OUTPUT_DIR:-./capture_filtered_data}"
PCAP_NAME="virtual_ap_filtered.pcap"
AP_MAC="${AP_MAC:-dc:4e:f4:0a:42:6f}"
DEVICE_MAC="${DEVICE_MAC:-fa:97:c8:94:78:8d}"
WPA2_OVERHEAD="${WPA2_OVERHEAD:-16}"

echo "Starting preprocessing pipeline..."
echo "Input directory:  $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "AP MAC address:   $AP_MAC"
echo "Device MAC:       $DEVICE_MAC"

# Step 1 — filter data frames and MAC addresses
echo "Step 1: Filtering 802.11 data frames between AP and device..."
INPUT_DIR="$INPUT_DIR" OUTPUT_DIR="$OUTPUT_DIR" bash "$SCRIPT_DIR/filter_data_frames.sh"

# Step 2 — extract packet sizes
echo "Step 2: Extracting packet sizes..."
ROOT_DIR="$OUTPUT_DIR" \
PCAP_NAME="$PCAP_NAME" \
AP_MAC="$AP_MAC" \
DEVICE_MAC="$DEVICE_MAC" \
WPA2_OVERHEAD="$WPA2_OVERHEAD" \
bash "$SCRIPT_DIR/extract_packet_sizes_per_file.sh"

# Step 3 — extract relative packet timings
echo "Step 3: Extracting relative timestamps..."
BASE_DIR="$OUTPUT_DIR" bash "$SCRIPT_DIR/get_timing_filtered_packages.sh"

# Step 4 — extract total capture time
echo "Step 4: Extracting capture time ranges..."
BASE_DIR="$OUTPUT_DIR" bash "$SCRIPT_DIR/get_capture_time_range.sh"

echo "Preprocessing pipeline completed successfully."

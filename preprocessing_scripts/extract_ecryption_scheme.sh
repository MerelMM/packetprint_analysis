#!/bin/bash
# This script was developed with assistance from ChatGPT (OpenAI) and Github Copilot
# Final implementation and adaptation by Merel Haenaets.
CAPTURE_DIR="capture"
OUTPUT_DIR="capture_data"
SSID="MerelAP"

for DIR in "$CAPTURE_DIR"/*; do
    PCAP_FILE="$DIR/virtual_ap.pcap"
    OUT_FILE="$OUTPUT_DIR/$(basename "$DIR")/encryption.txt"
    mkdir -p "$(dirname "$OUT_FILE")"

    # Zoek beacon frames met juiste SSID en extract akms.type en privacy bit
    BEACON_INFO=$(tshark -r "$PCAP_FILE" -Y "wlan.fc.type_subtype == 0x08 && wlan.ssid == \"$SSID\"" \
        -T fields -e wlan.rsn.akms.type -e wlan.fixed.capabilities.privacy -E separator=,)

    if [[ -z "$BEACON_INFO" ]]; then
        echo "No beacon frames with SSID '$SSID' found." > "$OUT_FILE"
        echo "$(basename "$DIR"): No beacon frames with SSID '$SSID' found."
    else
        AKM_TYPE=$(echo "$BEACON_INFO" | awk -F',' '{print $1; exit}')
        PRIVACY_BIT=$(echo "$BEACON_INFO" | awk -F',' '{print $2; exit}')

        if [[ "$AKM_TYPE" == "1" ]]; then
            RESULT="WPA"
        elif [[ "$AKM_TYPE" =~ ^[2-7]$ ]]; then
            RESULT="WPA2"
        elif [[ "$AKM_TYPE" =~ ^[8-9]|1[0-9]$ ]]; then
            RESULT="WPA3"
        elif [[ "$PRIVACY_BIT" == "True" ]]; then
            RESULT="WEP (privacy bit set, no RSN)"
        else
            RESULT="Open (no encryption)"
        fi

        echo "$RESULT" > "$OUT_FILE"
        echo "$(basename "$DIR"): $RESULT"
    fi
done

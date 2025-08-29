#!/bin/bash
# This script was developed with assistance from ChatGPT (OpenAI) and Github Copilot
# Final implementation and adaptation by Merel Haenaets.
ulimit -n 165536

# Configuration
PHY_INTERFACE="wlxdc4ef40a426f"
AP_SSID="MerelAP"
AP_PASSWORD="merel123"
MONITOR_IFACE="wlan0mon"

echo "[*] Starting virtual AP on $PHY_INTERFACE..."
sudo ./lnxrouter --ap "$PHY_INTERFACE" "$AP_SSID" -p "$AP_PASSWORD" &

# Wait a bit to ensure AP is up
sleep 3

echo "[*] Creating monitor interface $MONITOR_IFACE..."
sudo iw "$PHY_INTERFACE" interface add "$MONITOR_IFACE" type monitor

echo "[*] Bringing up monitor interface..."
sudo ip link set "$MONITOR_IFACE" up

echo "[✓] AP and monitor interface setup complete."

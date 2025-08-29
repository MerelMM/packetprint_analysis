#!/bin/bash
# This script was developed with assistance from ChatGPT (OpenAI) and Github Copilot
# Final implementation and adaptation by Merel Haenaets.
echo "📦 Phase 1: Running with selected_apps.txt (40 runs)"
for i in {1..40}; do #40
    echo "=== Run $i of 40 with selected_apps.txt ==="
    APP_LIST="selected_apps.txt" ./scripts/wifi_enabled2.sh
done # hetzelfde

echo "📦 Phase 2: Running with all_packages.txt (15 runs)"
for i in {1..15}; do #15
    echo "=== Run $i of 15 with all_packages.txt ==="
    APP_LIST="all_packages.txt" ./scripts/wifi_enabled2.sh
done #all_packages

#!/bin/bash 
# This script was developed with assistance from ChatGPT (OpenAI) and Github Copilot
# Final implementation and adaptation by Merel Haenaets.
# capturing traffic, explicitely turning the other apps off when starting capturing new app traffic (in the hope of reducing noise and less app multiplexing since spotify stops)

# === Config ===
CAPTURE_DIR="${CAPTURE_DIR:-capture_selected}"
MONITOR_IFACE="wlan0mon"
AP_INTERFACE="wlxdc4ef40a426f"
APP_LIST="${APP_LIST:-selected_apps.txt}"
LOGFILE="batched_run_log.txt"
BATCH_EVENT_COUNT=500 #500
TOTAL_BATCHES=5
# ==============

adb wait-for-device

exec 3< "$APP_LIST"
while IFS= read -r APP_PACKAGE <&3 || [ -n "$APP_PACKAGE" ]; do
    echo -e "\n==============================" | tee -a "$LOGFILE"
    echo "[*] Processing $APP_PACKAGE" | tee -a "$LOGFILE"

    # Check if app is installed
    if ! adb shell pm list packages | grep -q "$APP_PACKAGE"; then
        echo "[!] $APP_PACKAGE not installed — skipping" | tee -a "$LOGFILE"
        continue
    fi

    # Determine launchable activity using dumpsys
    MAIN_ACTIVITY=$(adb shell dumpsys package "$APP_PACKAGE" | grep -A 1 "MAIN" | grep "Activity" | head -n1 | awk '{print $2}')
    if [ -z "$MAIN_ACTIVITY" ]; then
        echo "[!] Could not determine main activity — skipping $APP_PACKAGE" | tee -a "$LOGFILE"
        continue
    fi
    if [[ "$MAIN_ACTIVITY" != "$APP_PACKAGE"* ]]; then
        MAIN_ACTIVITY="${APP_PACKAGE}/${MAIN_ACTIVITY}"
    fi

    # Session folder
    mkdir -p "$CAPTURE_DIR"
    SESSION_ID=$(date +%Y%m%d_%H%M%S)
    APP_NAME=$(basename "$APP_PACKAGE")
    SESSION_DIR="$CAPTURE_DIR/session_${APP_NAME}_${SESSION_ID}"
    mkdir -p "$SESSION_DIR"
    echo "[*] Starting session: $SESSION_DIR" | tee -a "$LOGFILE"

    # Stop previously running app (foreground)
    PREV_APP=$(adb shell dumpsys activity activities | grep "mResumedActivity" | awk '{print $4}' | cut -d '/' -f1)
    if [[ -n "$PREV_APP" && "$PREV_APP" != "$APP_PACKAGE" ]]; then
        echo "[*] Force-stopping previously resumed app: $PREV_APP" | tee -a "$LOGFILE"
        adb shell am force-stop "$PREV_APP"
    fi

    # Start captures
    echo "[*] Starting tcpdump..." | tee -a "$LOGFILE"
    sudo tcpdump -i "$MONITOR_IFACE" -s 0 -w "$SESSION_DIR/virtual_ap.pcap" &
    MON_PID1=$!
    sudo tcpdump -i "$AP_INTERFACE" -s 0 -w "$SESSION_DIR/ap.pcap" &
    MON_PID2=$!

    echo "[*] Capturing logcat..." | tee -a "$LOGFILE"
    adb logcat -v time > "$SESSION_DIR/logcat.txt" &
    LOG_PID=$!

    TOTAL_EVENTS_DONE=0

    # Force-stop all other user-installed apps
    echo "[*] Force-stopping all other user apps..." | tee -a "$LOGFILE"
    USER_APPS=$(adb shell pm list packages -3 | cut -f2 -d':')
    for pkg in $USER_APPS; do
        if [[ "$pkg" != "$APP_PACKAGE" ]]; then
            echo "  🔹 Killing $pkg" | tee -a "$LOGFILE"
            adb shell am force-stop "$pkg"
        fi
    done

    for i in $(seq 1 $TOTAL_BATCHES); do
        echo "[*] Monkey batch $i at $(date +%H:%M:%S)" | tee -a "$LOGFILE"

        # Ensures the wifi is on
        adb shell svc wifi enable

        # Start app explicitly
        adb shell am start -n "$MAIN_ACTIVITY"

        sleep 2

        # Run monkey and capture output
        MONKEY_OUTPUT=$(adb shell monkey -p "$APP_PACKAGE" \
            --throttle 300 \
            --pct-motion 70 \
            --pct-touch 25 \
            --pct-appswitch 0 \
            --pct-nav 0 \
            --pct-syskeys 0 \
            --pct-majornav 0 \
            --ignore-crashes \
            --ignore-timeouts \
            --monitor-native-crashes \
            --kill-process-after-error \
            -s $RANDOM \
            -v -v -v $BATCH_EVENT_COUNT)

        echo "$MONKEY_OUTPUT" >> "$SESSION_DIR/monkey_log.txt"

        # Extract and sum number of events actually injected
        EVENTS_DONE=$(echo "$MONKEY_OUTPUT" | grep -o "Events injected: [0-9]*" | awk '{sum += $3} END {print sum}')
        TOTAL_EVENTS_DONE=$((TOTAL_EVENTS_DONE + EVENTS_DONE))
        echo "[*] Events done in batch $i: $EVENTS_DONE — Total: $TOTAL_EVENTS_DONE" | tee -a "$LOGFILE"

        sleep 5

        # Check if still in foreground
        TOP_APP=$(adb shell dumpsys activity activities | grep "mResumedActivity" | grep "$APP_PACKAGE")
        if [ -z "$TOP_APP" ]; then
            echo "[!] App not in foreground after batch $i" | tee -a "$LOGFILE"
        else
            echo "[✓] App remained in foreground" | tee -a "$LOGFILE"
        fi
    done

    echo "[✓] Monkey session total events: $TOTAL_EVENTS_DONE (expected: $((BATCH_EVENT_COUNT * TOTAL_BATCHES)))" | tee -a "$LOGFILE"

    # Stop background processes
    echo "[*] Stopping captures..." | tee -a "$LOGFILE"
    sudo kill $MON_PID1 $MON_PID2
    kill $LOG_PID

    echo "[✓] Finished $APP_PACKAGE" | tee -a "$LOGFILE"
done
exec 3<&-

echo -e "\n[✓] All apps processed." | tee -a "$LOGFILE"

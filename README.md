This repository contains a full open source reimplementation of the **PacketPrint (PP)** attack framework[1], originally proposed for **passive app fingerprinting** on encrypted Wi-Fi traffic. 

PacketPrint is a **three-phase architecture**:

1. **Preprocessing** – filters and normalizes raw wireless packet traces.
2. **Segmentation** – detects potential app-related segments using statistical and temporal features.
3. **Recognition** – classifies which app generated each proposed segment using learned traffic patterns.

This implementation includes the entire workflow: from collecting and preprocessing data, through segment proposal, to final app classification.

For capturing of the data the following tools were used:

Creating a WiFi hotspot:
https://github.com/garywill/linux-router

Creating a virtual interface on the AP:
https://github.com/vanhoefm/libwifi/blob/master/docs/linux_tutorial.md#virtual-interfaces

#### References

[1] PacketPrint: Li, J., Wu, S., Zhou, H., Luo, X., Wang, T., Liu, Y., & Ma, X. (2022). Packet-Level Open-World App Fingerprinting on Wireless Traffic. Proceedings 2022 Network and Distributed System Security Symposium.

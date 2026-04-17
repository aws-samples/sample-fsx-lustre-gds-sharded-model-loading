#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# configure-lustre-client.sh — Configure NUMA-aware Lustre client networking
#
# This script detects the instance type (P5 vs P5e/P5en) based on NUMA topology
# and generates an optimized LNet configuration that aligns EFA interfaces with
# CPU partitions for maximum GDS throughput.
#
# Prerequisites:
#   - EFA driver installed and instance REBOOTED (run 1-setup-gds.sh, then reboot)
#   - Run as root or with sudo
#
# Usage: sudo ./configure-lustre-client.sh

set -euo pipefail

echo "============================================"
echo " FSx for Lustre GDS Setup"
echo " Step 2: NUMA-Aware Lustre Client Config"
echo "============================================"

# -----------------------------------------------------------
# Detect instance topology
# -----------------------------------------------------------
NUMA_NODES=$(numactl -H | grep "available:" | awk '{print $2}')
echo "Detected ${NUMA_NODES} NUMA nodes"

# Get default network interface
DEF_IF=$(ip route | grep default | head -1 | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}')
echo "Default network interface: ${DEF_IF}"
echo ""

# -----------------------------------------------------------
# Define CPU Partition Table (CPT) mappings per instance type
# Maps EFA interface PCI bus IDs to NUMA-aligned CPU partitions
# -----------------------------------------------------------

# P5 (2 NUMA nodes, 8 partitions, 24 CPUs per partition)
declare -A P5_CPT=(
    [79]=0 [80]=1 [81]=4 [82]=5 [96]=0 [97]=1 [98]=4 [99]=5
    [113]=0 [114]=1 [115]=4 [116]=5 [130]=0 [131]=1 [132]=4 [133]=5
    [147]=2 [148]=3 [149]=6 [150]=7 [164]=2 [165]=3 [166]=6 [167]=7
    [181]=2 [182]=3 [183]=6 [184]=7 [198]=2 [199]=3 [200]=6 [201]=7
)

# P5e/P5en (4 NUMA nodes, 16 partitions, 12 CPUs per partition)
declare -A P5E_CPT=(
    [79]=2 [80]=3 [81]=10 [82]=11 [96]=2 [97]=3 [98]=10 [99]=11
    [113]=0 [114]=1 [115]=8 [116]=9 [130]=0 [131]=1 [132]=8 [133]=9
    [147]=6 [148]=7 [149]=14 [150]=15 [164]=6 [165]=7 [166]=14 [167]=15
    [181]=4 [182]=5 [183]=12 [184]=13 [198]=4 [199]=5 [200]=12 [201]=13
)

# EFA interface PCI bus IDs (same across P5/P5e/P5en)
EFA_BUS_IDS=(79 80 81 82 96 97 98 99 113 114 115 116 130 131 132 133
             147 148 149 150 164 165 166 167 181 182 183 184 198 199 200 201)

# -----------------------------------------------------------
# Generate CPU pattern and select CPT map based on NUMA count
# -----------------------------------------------------------
PATTERN=""
if [[ $NUMA_NODES -eq 2 ]]; then
    echo "Instance type: P5 (2 NUMA nodes, 8 CPU partitions)"
    declare -n CPT_MAP=P5_CPT
    NPART=8
    for i in {0..7}; do
        start=$((i * 24))
        end=$((start + 23))
        PATTERN+=" ${i}[$(seq -s, $start $end)]"
    done
else
    echo "Instance type: P5e/P5en (${NUMA_NODES} NUMA nodes, 16 CPU partitions)"
    declare -n CPT_MAP=P5E_CPT
    NPART=16
    for i in {0..15}; do
        start=$((i * 12))
        end=$((start + 11))
        PATTERN+=" ${i}[$(seq -s, $start $end)]"
    done
fi

echo ""

# -----------------------------------------------------------
# Apply the Lustre client configuration
# -----------------------------------------------------------
echo ">>> Cleaning existing LNet configuration..."
sudo lnetctl net del --net tcp 2>/dev/null || true
sudo lustre_rmmod 2>/dev/null || true

echo ">>> Loading Lustre modules with ${NPART} CPU partitions..."
sudo modprobe lnet cpu_npartitions=${NPART} cpu_pattern="${PATTERN# }"
sudo modprobe ptlrpc ptlrpcd_per_cpt_max=24
sudo modprobe ksocklnd
sudo modprobe kefalnd ipif_name="${DEF_IF}"

echo ">>> Configuring LNet..."
sudo lnetctl lnet configure
sudo lnetctl net add --net tcp --if "${DEF_IF}"

echo ">>> Adding EFA interfaces with NUMA-aware CPT mapping..."
for bus_id in "${EFA_BUS_IDS[@]}"; do
    cpt=${CPT_MAP[$bus_id]}
    sudo lnetctl net add --net efa --if "rdmap${bus_id}s0" --cpt "${cpt}" --peer-credits 128
done

echo ">>> Enabling routing and EFA priority..."
sudo lnetctl set discovery 1
sudo lnetctl udsp add --src efa --priority 0

echo ""
echo ">>> Verifying LNet configuration..."
sudo lnetctl net show | head -20
echo "... (truncated — run 'sudo lnetctl net show' for full output)"

echo ""
echo "============================================"
echo " Lustre client configuration complete."
echo " Next: run mount-and-tune.sh"
echo "============================================"

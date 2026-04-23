#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# configure-lustre-client.sh — Configure NUMA-aware Lustre client networking
#
# Auto-detects EFA RDMA devices and their NUMA topology, then generates an
# optimized LNet configuration that aligns EFA interfaces with CPU partitions
# for maximum GDS throughput.
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
TOTAL_CPUS=$(nproc)
echo "Detected ${NUMA_NODES} NUMA nodes, ${TOTAL_CPUS} CPUs"

# Get default network interface
DEF_IF=$(ip route | grep default | head -1 | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}')
echo "Default network interface: ${DEF_IF}"

# -----------------------------------------------------------
# Discover EFA RDMA devices and their NUMA nodes
# -----------------------------------------------------------
echo ""
echo ">>> Discovering EFA interfaces..."

EFA_DEVICES=()
EFA_NUMA=()
for dev in $(ls /sys/class/infiniband/ | sort -t'p' -k2 -n); do
    numa=$(cat "/sys/class/infiniband/${dev}/device/numa_node")
    EFA_DEVICES+=("$dev")
    EFA_NUMA+=("$numa")
    echo "    ${dev} → NUMA ${numa}"
done

NUM_EFA=${#EFA_DEVICES[@]}
if [[ $NUM_EFA -eq 0 ]]; then
    echo "ERROR: No EFA RDMA devices found. Did you reboot after running 1-setup-gds.sh?"
    exit 1
fi
echo "Found ${NUM_EFA} EFA interfaces"

# -----------------------------------------------------------
# Generate CPU partitions based on NUMA topology
#
# Strategy: create PARTITIONS_PER_NUMA partitions per NUMA node,
# distributing that node's CPUs evenly. EFA interfaces on each
# NUMA node are round-robin assigned across that node's partitions.
# -----------------------------------------------------------
PARTITIONS_PER_NUMA=4
NPART=$((NUMA_NODES * PARTITIONS_PER_NUMA))
echo ""
echo "Configuring ${NPART} CPU partitions (${PARTITIONS_PER_NUMA} per NUMA node)"

PATTERN=""
PART_IDX=0
for (( n=0; n<NUMA_NODES; n++ )); do
    # Get CPUs for this NUMA node
    CPUS=$(numactl -H | grep "node ${n} cpus:" | sed "s/node ${n} cpus: //")
    read -ra CPU_LIST <<< "$CPUS"
    NCPUS=${#CPU_LIST[@]}
    CPUS_PER_PART=$((NCPUS / PARTITIONS_PER_NUMA))

    for (( p=0; p<PARTITIONS_PER_NUMA; p++ )); do
        start=$((p * CPUS_PER_PART))
        end=$((start + CPUS_PER_PART - 1))
        cpu_range=$(IFS=,; echo "${CPU_LIST[*]:$start:$CPUS_PER_PART}")
        PATTERN+=" ${PART_IDX}[${cpu_range}]"
        PART_IDX=$((PART_IDX + 1))
    done
done

echo ""

# -----------------------------------------------------------
# Build NUMA-to-CPT mapping for EFA interfaces
# Round-robin each NUMA node's EFA devices across its partitions
# -----------------------------------------------------------
declare -A NUMA_COUNTER
for (( n=0; n<NUMA_NODES; n++ )); do
    NUMA_COUNTER[$n]=0
done

EFA_CPT=()
for (( i=0; i<NUM_EFA; i++ )); do
    n=${EFA_NUMA[$i]}
    local_idx=${NUMA_COUNTER[$n]}
    cpt=$(( n * PARTITIONS_PER_NUMA + (local_idx % PARTITIONS_PER_NUMA) ))
    EFA_CPT+=("$cpt")
    NUMA_COUNTER[$n]=$((local_idx + 1))
done

# -----------------------------------------------------------
# Apply the Lustre client configuration
# -----------------------------------------------------------
echo ">>> Cleaning existing LNet configuration..."
sudo lnetctl net del --net efa 2>/dev/null || true
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
for (( i=0; i<NUM_EFA; i++ )); do
    dev=${EFA_DEVICES[$i]}
    cpt=${EFA_CPT[$i]}
    echo "    ${dev} → CPT ${cpt}"
    sudo lnetctl net add --net efa --if "${dev}" --cpt "${cpt}" --peer-credits 128
done

echo ">>> Enabling routing and EFA priority..."
sudo lnetctl set discovery 1
sudo lnetctl udsp add --src efa --priority 0

echo ""
echo ">>> Verifying LNet configuration..."
EFA_COUNT=$(sudo lnetctl net show --net efa | grep -c "status: up" || true)
echo "${EFA_COUNT} EFA interfaces configured and up"

echo ""
echo "============================================"
echo " Lustre client configuration complete."
echo " Next: run 3-mount-and-tune.sh"
echo "============================================"

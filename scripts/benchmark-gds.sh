#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# benchmark-gds.sh — Validate GDS performance with NVIDIA GDSIO benchmark
#
# Runs one job per GPU with NUMA-aligned threads to verify GDS throughput
# from FSx for Lustre directly to GPU memory.
#
# Prerequisites:
#   - GDS configured and FSx mounted (run previous scripts first)
#   - CUDA toolkit installed (for gdsio tool)
#   - Run as root or with sudo
#
# Usage: sudo ./benchmark-gds.sh [threads_per_gpu]
#   threads_per_gpu — threads per GPU job (default: 32)

set -euo pipefail

THREADS=${1:-32}
MOUNT=${MOUNT:-/fsx}
NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "============================================"
echo " GDS Performance Benchmark"
echo " GPUs: ${NUM_GPUS}, Threads per GPU: ${THREADS}"
echo " Mount: ${MOUNT}"
echo "============================================"
echo ""

# -----------------------------------------------------------
# Find GDSIO tool
# -----------------------------------------------------------
GDSIO=""
for p in /usr/local/cuda-*/gds/tools/gdsio /usr/local/cuda/gds/tools/gdsio; do
    [ -f "$p" ] && GDSIO="$p" && break
done

if [ -z "$GDSIO" ]; then
    echo "ERROR: gdsio not found in /usr/local/cuda-*/gds/tools/"
    exit 1
fi
echo "Using: ${GDSIO}"
echo ""

# -----------------------------------------------------------
# Detect GPU NUMA topology
# -----------------------------------------------------------
echo "GPU topology:"
declare -a GPU_NUMA
for (( i=0; i<NUM_GPUS; i++ )); do
    pci=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i $i \
        | tr '[:upper:]' '[:lower:]' | sed 's/^0000://')
    numa_file="/sys/bus/pci/devices/${pci}/numa_node"
    if [ -f "$numa_file" ]; then
        numa=$(cat "$numa_file")
        # -1 means no NUMA affinity; default to 0
        [ "$numa" -lt 0 ] 2>/dev/null && numa=0
    else
        numa=0
    fi
    GPU_NUMA[$i]=$numa
    echo "  GPU${i} (${pci}) → NUMA ${numa}"
done
echo ""

# -----------------------------------------------------------
# Create benchmark directories with Lustre striping
# -----------------------------------------------------------
echo ">>> Creating benchmark directories..."
for (( i=0; i<NUM_GPUS; i++ )); do
    dir="${MOUNT}/gds_benchmark/gpu${i}"
    mkdir -p "$dir"
    lfs setstripe -S 1M -c -1 "$dir"
done

# -----------------------------------------------------------
# Generate GDSIO configs (one job per GPU)
# -----------------------------------------------------------
for test in write read; do
    config="/tmp/${test}_benchmark.gdsio"
    cat > "${config}" << EOF
[global]
name=gds_${test}
xfer_type=0
bs=1MB
size=20G
runtime=60
do_verify=0
enable_nvlinks=0
rw=${test}
EOF
    for (( i=0; i<NUM_GPUS; i++ )); do
        cat >> "${config}" << EOF

[job${i}]
numa_node=${GPU_NUMA[$i]}
gpu_dev_id=${i}
num_threads=${THREADS}
directory=${MOUNT}/gds_benchmark/gpu${i}
EOF
    done
done

echo ""
ulimit -n 100000

# -----------------------------------------------------------
# Run benchmarks
# -----------------------------------------------------------
echo ">>> WRITE benchmark (${NUM_GPUS} GPUs × ${THREADS} threads, 60s)..."
${GDSIO} /tmp/write_benchmark.gdsio
echo ""

echo ">>> READ benchmark (${NUM_GPUS} GPUs × ${THREADS} threads, 60s)..."
${GDSIO} /tmp/read_benchmark.gdsio
echo ""

# -----------------------------------------------------------
# Show GDS stats
# -----------------------------------------------------------
echo ">>> GDS kernel stats:"
cat /proc/driver/nvidia-fs/stats | grep -E "Reads|Writes|Ops"
echo ""

# -----------------------------------------------------------
# Cleanup
# -----------------------------------------------------------
rm -rf "${MOUNT}/gds_benchmark" /tmp/write_benchmark.gdsio /tmp/read_benchmark.gdsio

echo "============================================"
echo " Benchmark complete."
echo "============================================"

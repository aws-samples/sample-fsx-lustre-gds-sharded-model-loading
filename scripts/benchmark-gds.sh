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
# Usage: sudo ./benchmark-gds.sh [threads_per_gpu] [block_size]
#   threads_per_gpu — threads per GPU job (default: 8)
#   block_size      — I/O block size (default: 16MB)
#
# The defaults (8 threads × 16MB) were tuned on p5en.48xlarge + FSx Persistent_2
# EFA (96 TiB, 20 OSTs, 1000 MBps/TiB) and deliver ~94 GiB/s read with ~10 ms
# avg latency — close to the theoretical filesystem maximum. Smaller block
# sizes underutilize parallelism (bs=1MB → ~17 GiB/s); larger block sizes
# (bs=64MB) match peak throughput but drive latency into the 100+ ms range.

set -euo pipefail

THREADS=${1:-8}
BLOCK_SIZE=${2:-16MB}
MOUNT=${MOUNT:-/fsx}
NUM_GPUS=$(nvidia-smi -L | wc -l)

echo "============================================"
echo " GDS Performance Benchmark"
echo " GPUs: ${NUM_GPUS}, Threads per GPU: ${THREADS}, Block size: ${BLOCK_SIZE}"
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
# Stripe size should match (or be a divisor of) the I/O block size
# so each request fits cleanly into OST chunks; 16M stripe pairs
# with the default 16MB I/O block size.
# -----------------------------------------------------------
echo ">>> Creating benchmark directories (stripe: all OSTs, 16M)..."
for (( i=0; i<NUM_GPUS; i++ )); do
    dir="${MOUNT}/gds_benchmark/gpu${i}"
    mkdir -p "$dir"
    lfs setstripe -S 16M -c -1 "$dir"
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
bs=${BLOCK_SIZE}
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
echo ">>> WRITE benchmark (${NUM_GPUS} GPUs × ${THREADS} threads × ${BLOCK_SIZE}, 60s)..."
${GDSIO} /tmp/write_benchmark.gdsio
echo ""

echo ">>> READ benchmark (${NUM_GPUS} GPUs × ${THREADS} threads × ${BLOCK_SIZE}, 60s)..."
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

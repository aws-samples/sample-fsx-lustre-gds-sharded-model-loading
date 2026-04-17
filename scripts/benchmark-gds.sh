#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# benchmark-gds.sh — Validate GDS performance with NVIDIA GDSIO benchmark
#
# Creates a NUMA-aware benchmark configuration and runs read/write tests
# to verify GDS throughput from FSx for Lustre to GPU memory.
#
# Prerequisites:
#   - GDS configured and FSx mounted (run previous scripts first)
#   - CUDA toolkit installed (for gdsio tool)
#   - Run as root or with sudo
#
# Usage: sudo ./benchmark-gds.sh [jobs] [threads]
#   jobs    — number of parallel benchmark jobs (default: 32)
#   threads — threads per job (default: 32)
#
# Example: sudo ./benchmark-gds.sh 32 32

set -euo pipefail

JOBS=${1:-32}
THREADS=${2:-32}
MOUNT=${MOUNT:-/fsx}

echo "============================================"
echo " GDS Performance Benchmark"
echo " Jobs: ${JOBS}, Threads per job: ${THREADS}"
echo " Mount: ${MOUNT}"
echo "============================================"
echo ""

# -----------------------------------------------------------
# Find GDSIO tool
# -----------------------------------------------------------
GDSIO=""
for cuda_dir in /usr/local/cuda-*/gds/tools/gdsio /usr/local/cuda/gds/tools/gdsio; do
    if [ -f "$cuda_dir" ]; then
        GDSIO="$cuda_dir"
        break
    fi
done

if [ -z "$GDSIO" ]; then
    echo "ERROR: gdsio benchmark tool not found."
    echo "Ensure CUDA toolkit is installed. Look for it in /usr/local/cuda-*/gds/tools/"
    exit 1
fi

echo "Using GDSIO: ${GDSIO}"
echo ""

# -----------------------------------------------------------
# Create benchmark directories with Lustre striping
# -----------------------------------------------------------
echo ">>> Creating benchmark directories..."
for ((i = 0; i < JOBS; i++)); do
    dir="${MOUNT}/gds_benchmark/job${i}/data"
    mkdir -p "$dir"
    lfs setstripe -S 1M -c -1 "$dir"
done

# -----------------------------------------------------------
# Get GPU-to-NUMA mapping from nvidia-smi
# -----------------------------------------------------------
declare -A GPU_NUMA
for i in {0..7}; do
    numa=$(nvidia-smi topo -m | grep "^GPU${i}" | awk '{print $NF}')
    GPU_NUMA[$i]=${numa:-0}
done

echo "GPU-to-NUMA mapping:"
for i in {0..7}; do
    echo "  GPU${i} -> NUMA ${GPU_NUMA[$i]}"
done
echo ""

# -----------------------------------------------------------
# Generate GDSIO benchmark configs
# -----------------------------------------------------------
for test in read write; do
    config="${test}_benchmark.gdsio"
    cat > "${config}" << EOF
[global]
name=1MB_seq
xfer_type=0
bs=1MB
size=100G
runtime=60
do_verify=0
enable_nvlinks=0
rw=${test}
EOF

    for ((i = 0; i < JOBS; i++)); do
        gpu=$((i % 8))
        numa=${GPU_NUMA[$gpu]}
        cat >> "${config}" << EOF

[job${i}]
numa_node=${numa}
gpu_dev_id=${gpu}
num_threads=${THREADS}
directory=${MOUNT}/gds_benchmark/job${i}/data
EOF
    done

    echo "Generated: ${config}"
done

echo ""

# -----------------------------------------------------------
# Run benchmarks
# -----------------------------------------------------------
echo ">>> Running WRITE benchmark (60 seconds)..."
echo ""
ulimit -n 100000
${GDSIO} write_benchmark.gdsio
echo ""

echo ">>> Running READ benchmark (60 seconds)..."
echo ""
${GDSIO} read_benchmark.gdsio
echo ""

# -----------------------------------------------------------
# Cleanup
# -----------------------------------------------------------
echo ">>> Cleaning up benchmark data..."
rm -rf "${MOUNT}/gds_benchmark"
rm -f read_benchmark.gdsio write_benchmark.gdsio

echo ""
echo "============================================"
echo " Benchmark complete."
echo " Expected results on P5en:"
echo "   Write: ~158 GiB/s (~1300 Gbps)"
echo "   Read:  ~190 GiB/s (~1600 Gbps)"
echo "============================================"

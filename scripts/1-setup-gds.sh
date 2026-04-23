#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# setup-gds.sh — Install EFA driver and build NVIDIA GPUDirect Storage kernel module
#
# The nvidia-fs.ko module is installed into the kernel module tree and configured
# to load automatically on boot via /etc/modules-load.d, so it survives reboots.
#
# Prerequisites:
#   - NVIDIA drivers and CUDA must be installed (use a Deep Learning AMI)
#   - Run as root or with sudo
#
# Usage: sudo ./setup-gds.sh

set -euo pipefail

echo "============================================"
echo " FSx for Lustre GDS Setup"
echo " Step 1: EFA Driver + GPUDirect Storage"
echo "============================================"

# Verify NVIDIA drivers are installed
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first (use a Deep Learning AMI)."
    exit 1
fi

echo ""
echo "NVIDIA driver detected:"
nvidia-smi --query-gpu=driver_version,name --format=csv,noheader | head -1 || true
echo ""

KERNEL_VER=$(uname -r)
MODULE_DIR="/lib/modules/${KERNEL_VER}/extra"

# -----------------------------------------------------------
# 1. Install EFA driver
# -----------------------------------------------------------
echo ">>> Installing EFA driver..."
cd /tmp
curl -O https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz
tar -xf aws-efa-installer-latest.tar.gz
cd aws-efa-installer
sudo ./efa_installer.sh -y
cd /tmp
rm -rf aws-efa-installer aws-efa-installer-latest.tar.gz

echo ">>> EFA driver installed."
echo ""

# -----------------------------------------------------------
# 2. Build and install GDS kernel module
# -----------------------------------------------------------
echo ">>> Building GPUDirect Storage kernel module..."

# Unload existing module if present
sudo rmmod nvidia_fs 2>/dev/null || true

# Clone NVIDIA GDS kernel module source
BUILD_DIR=$(mktemp -d)
git clone https://github.com/NVIDIA/gds-nvidia-fs.git "${BUILD_DIR}/gds-nvidia-fs"
cd "${BUILD_DIR}/gds-nvidia-fs/src"

# Configure for P5/P5e/P5en/P6e instances (large PCI topology)
export NVFS_MAX_PEER_DEVS=128
export NVFS_MAX_PCI_DEPTH=16

# Build the module
make

# Install into kernel module tree so it survives reboots
sudo mkdir -p "${MODULE_DIR}"
sudo cp nvidia-fs.ko "${MODULE_DIR}/nvidia-fs.ko"
sudo depmod -a

# Clean up build directory
rm -rf "${BUILD_DIR}"

# Load the module
sudo modprobe nvidia_fs

echo ""
echo ">>> GDS kernel module built and installed to ${MODULE_DIR}"
echo ""

# -----------------------------------------------------------
# 3. Configure nvidia_fs to load on boot
# -----------------------------------------------------------
echo ">>> Configuring nvidia_fs to load on boot..."
echo "nvidia_fs" | sudo tee /etc/modules-load.d/nvidia-fs.conf > /dev/null

# Set build options so they apply on modprobe too
echo "options nvidia_fs nvfs_max_peer_devs=128 nvfs_max_pci_depth=16" \
    | sudo tee /etc/modprobe.d/nvidia-fs.conf > /dev/null

echo ">>> Auto-load configured."
echo ""

# -----------------------------------------------------------
# 4. Verify GDS is active
# -----------------------------------------------------------
echo ">>> Verifying GDS installation..."
echo ""

if lsmod | grep -q nvidia_fs; then
    echo "PASS: nvidia_fs kernel module is loaded"
else
    echo "FAIL: nvidia_fs kernel module is NOT loaded"
    exit 1
fi

if [ -f /proc/driver/nvidia-fs/stats ]; then
    echo "PASS: GDS stats interface is available"
    echo ""
    echo "GDS stats:"
    cat /proc/driver/nvidia-fs/stats
else
    echo "FAIL: /proc/driver/nvidia-fs/stats not found"
    exit 1
fi

# -----------------------------------------------------------
# 5. Create GDS configuration file
# -----------------------------------------------------------
echo ""
echo ">>> Creating GDS configuration (/etc/cufile.json)..."

sudo tee /etc/cufile.json > /dev/null << 'CUFILEEOF'
{
  "logging": {
    "level": "ERROR"
  },
  "profile": {
    "nvtx": false,
    "cufile_stats": 3
  },
  "execution": {
    "max_io_queue_depth": 128,
    "max_io_threads": 4,
    "parallel_io": true,
    "min_io_threshold_size_kb": 8192,
    "max_request_parallelism": 4
  },
  "properties": {
    "max_direct_io_size_kb": 16384,
    "max_device_cache_size_kb": 131072,
    "max_device_pinned_mem_size_kb": 33554432,
    "use_poll_mode": false,
    "poll_mode_max_size_kb": 4,
    "allow_compat_mode": false,
    "rdma_dev_addr_list": [],
    "rdma_dynamic_routing": false
  },
  "fs": {
    "generic": {
      "posix_unaligned_writes": false
    },
    "lustre": {
      "posix_gds_min_kb": 0
    }
  }
}
CUFILEEOF

echo ">>> GDS configuration created."
echo ""
echo "============================================"
echo " GDS setup complete."
echo ""
echo " *** REBOOT REQUIRED ***"
echo " The EFA driver needs a reboot to activate"
echo " all EFA interfaces. After reboot, nvidia_fs"
echo " will load automatically. Run:"
echo ""
echo "   sudo reboot"
echo ""
echo " After reboot, run: 2-configure-lustre-client.sh"
echo "============================================"

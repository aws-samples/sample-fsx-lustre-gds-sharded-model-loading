#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# configure-lustre-client.sh — Configure EFA for the Lustre client using the
# official AWS setup script, optimized for GPUDirect Storage.
#
# This downloads and runs the AWS-provided configure-efa-fsx-lustre-client
# script, which:
#   - Detects instance type and NUMA topology
#   - Loads Lustre and EFA kernel modules with optimal CPU partitioning
#   - Configures the correct subset of EFA interfaces for FSx for Lustre
#   - Sets EFA as the preferred Lustre network transport
#   - Creates a systemd service for automatic configuration on reboot
#
# Prerequisites:
#   - EFA driver installed and instance REBOOTED (run 1-setup-gds.sh, then reboot)
#   - Run as root or with sudo
#
# Usage: sudo ./configure-lustre-client.sh

set -euo pipefail

echo "============================================"
echo " FSx for Lustre GDS Setup"
echo " Step 2: EFA Lustre Client Configuration"
echo "============================================"
echo ""

# -----------------------------------------------------------
# Download the official AWS EFA setup script
# -----------------------------------------------------------
WORK_DIR=$(mktemp -d)
cd "${WORK_DIR}"

echo ">>> Downloading AWS EFA Lustre client configuration script..."
curl -sO https://docs.aws.amazon.com/fsx/latest/LustreGuide/samples/configure-efa-fsx-lustre-client.zip
unzip -q configure-efa-fsx-lustre-client.zip
cd configure-efa-fsx-lustre-client

# -----------------------------------------------------------
# Run the official setup with GDS optimization
# -----------------------------------------------------------
echo ">>> Running official AWS setup (--optimized-for-gds)..."
echo ""
sudo ./setup.sh --optimized-for-gds

echo ""

# -----------------------------------------------------------
# Verify
# -----------------------------------------------------------
echo ">>> Verifying EFA configuration..."
EFA_COUNT=$(sudo lnetctl net show --net efa 2>/dev/null | grep -c "status: up" || true)
echo "${EFA_COUNT} EFA interface(s) configured and up"

echo ""
echo "============================================"
echo " Lustre client configuration complete."
echo " A systemd service has been created for"
echo " automatic EFA configuration on reboot."
echo ""
echo " Next: run 3-mount-and-tune.sh"
echo "============================================"

# Cleanup
rm -rf "${WORK_DIR}"

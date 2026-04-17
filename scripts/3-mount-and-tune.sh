#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# mount-and-tune.sh — Mount FSx for Lustre and apply performance tuning
#
# Prerequisites:
#   - Lustre client configured (run 2-configure-lustre-client.sh first)
#   - FSx filesystem must be in AVAILABLE state
#   - Run as root or with sudo
#
# Usage: sudo ./mount-and-tune.sh <fsx-dns-name> <mount-name>
#
# Example:
#   sudo ./mount-and-tune.sh fs-0123456789abcdef0.fsx.us-east-1.amazonaws.com abcd1234
#
# If /etc/fsx-config.env exists (created by CloudFormation UserData), the script
# will attempt to auto-detect the DNS name and mount name.

set -euo pipefail

echo "============================================"
echo " FSx for Lustre GDS Setup"
echo " Step 3: Mount and Tune"
echo "============================================"

MOUNT_POINT="/fsx"

# -----------------------------------------------------------
# Determine FSx DNS name and mount name
# -----------------------------------------------------------
if [[ $# -ge 2 ]]; then
    FSX_DNS="$1"
    MOUNT_NAME="$2"
elif [[ -f /etc/fsx-config.env ]]; then
    source /etc/fsx-config.env
    echo ">>> Auto-detecting FSx filesystem details..."

    # Query the filesystem for DNS name and mount name
    FS_INFO=$(aws fsx describe-file-systems \
        --file-system-ids "${FSX_FILESYSTEM_ID}" \
        --region "${FSX_REGION}" \
        --query 'FileSystems[0]' \
        --output json)

    LIFECYCLE=$(echo "$FS_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin)['Lifecycle'])")
    if [[ "$LIFECYCLE" != "AVAILABLE" ]]; then
        echo "ERROR: FSx filesystem ${FSX_FILESYSTEM_ID} is in state '${LIFECYCLE}'. Wait until AVAILABLE."
        echo "Check status: aws fsx describe-file-systems --file-system-ids ${FSX_FILESYSTEM_ID} --query 'FileSystems[0].Lifecycle'"
        exit 1
    fi

    FSX_DNS=$(echo "$FS_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin)['DNSName'])")
    MOUNT_NAME=$(echo "$FS_INFO" | python3 -c "import sys,json; print(json.load(sys.stdin)['LustreConfiguration']['MountName'])")
    echo "    Filesystem ID: ${FSX_FILESYSTEM_ID}"
    echo "    DNS Name:      ${FSX_DNS}"
    echo "    Mount Name:    ${MOUNT_NAME}"
else
    echo "Usage: $0 <fsx-dns-name> <mount-name>"
    echo ""
    echo "Example: $0 fs-0123456789abcdef0.fsx.us-east-1.amazonaws.com abcd1234"
    echo ""
    echo "Find these values with:"
    echo "  aws fsx describe-file-systems --file-system-ids <fs-id> \\"
    echo "    --query 'FileSystems[0].[DNSName,LustreConfiguration.MountName]'"
    exit 1
fi

echo ""

# -----------------------------------------------------------
# Mount the filesystem
# -----------------------------------------------------------
echo ">>> Mounting FSx for Lustre at ${MOUNT_POINT}..."
sudo mkdir -p "${MOUNT_POINT}"
sudo mount -t lustre -o relatime,flock "${FSX_DNS}@tcp:/${MOUNT_NAME}" "${MOUNT_POINT}"

echo ">>> Filesystem mounted."
df -h "${MOUNT_POINT}"
echo ""

# -----------------------------------------------------------
# Apply Lustre performance tuning
# -----------------------------------------------------------
echo ">>> Applying Lustre performance tuning..."

# Increase max RPCs in flight for metadata and data operations
sudo lctl set_param mdc.*.max_rpcs_in_flight=64
sudo lctl set_param mdc.*.max_mod_rpcs_in_flight=50
sudo lctl set_param osc.*OST*.max_rpcs_in_flight=64

echo ">>> Tuning applied."
echo ""

# -----------------------------------------------------------
# Create model shards directory with optimal striping
# -----------------------------------------------------------
echo ">>> Creating model shards directory with optimal Lustre striping..."
sudo mkdir -p "${MOUNT_POINT}/model_shards"

# Stripe across all OSTs with 64 MB stripe size for large sequential reads
sudo lfs setstripe -c -1 -S 64M "${MOUNT_POINT}/model_shards"

echo ">>> Striping configuration:"
lfs getstripe "${MOUNT_POINT}/model_shards"

echo ""
echo "============================================"
echo " FSx for Lustre mounted and tuned."
echo " Mount point: ${MOUNT_POINT}"
echo " Model shards directory: ${MOUNT_POINT}/model_shards/"
echo ""
echo " Next: optionally run benchmark-gds.sh to"
echo " validate GDS throughput, then use"
echo " model-loading/load-sharded-model.py to"
echo " load your model."
echo "============================================"

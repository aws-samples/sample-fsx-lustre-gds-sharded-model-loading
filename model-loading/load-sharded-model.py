# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# load-sharded-model.py — Load a pre-sharded, pre-quantized LLM via GDS
#
# This script demonstrates the sharded parallel model loading pattern
# described in the blog post. It:
#   1. Verifies GDS is active
#   2. Loads a pre-sharded model using vLLM with tensor parallelism
#   3. Runs a quick inference test to verify the model works
#
# Prerequisites:
#   - GDS configured and Amazon FSx for Lustre mounted (run the setup scripts first)
#   - Pre-sharded model on Amazon FSx for Lustre (run pre-shard step below first)
#   - vLLM installed: pip install vllm
#
# Pre-shard your model (one-time offline step):
#   python -m vllm.utils.save_sharded_state \
#     --model /fsx/models/Llama-3.1-405B \
#     --quantization fp8 \
#     --tensor-parallel-size 8 \
#     --output /fsx/model_shards/Llama-3.1-405B-FP8-8way
#
# Usage:
#   python load-sharded-model.py \
#     --model-path /fsx/model_shards/Llama-3.1-405B-FP8-8way \
#     --tensor-parallel-size 8 \
#     --quantization fp8

import argparse
import os
import subprocess
import sys
import time


def check_gds_active():
    """Verify that NVIDIA GPUDirect Storage is active."""
    print(">>> Checking GDS status...")

    # Check kernel module
    result = subprocess.run(
        ["lsmod"], capture_output=True, text=True
    )
    if "nvidia_fs" not in result.stdout:
        print("WARNING: nvidia_fs kernel module is not loaded.")
        print("GDS is not active — reads will fall back to CPU bounce buffer path.")
        print("Load the module with: sudo modprobe nvidia_fs")
        return False

    # Check stats interface
    stats_path = "/proc/driver/nvidia-fs/stats"
    if not os.path.exists(stats_path):
        print(f"WARNING: {stats_path} not found. GDS may not be active.")
        return False

    with open(stats_path, "r") as f:
        stats = f.read()
    print(f"GDS stats:\n{stats}")
    print("PASS: GDS is active.\n")
    return True


def load_model(model_path, tensor_parallel_size, quantization):
    """Load a pre-sharded model using vLLM."""
    from vllm import LLM

    print(f">>> Loading model from: {model_path}")
    print(f"    Tensor parallel size: {tensor_parallel_size}")
    print(f"    Quantization: {quantization}")
    print(f"    Load format: sharded_state")
    print()

    t0 = time.perf_counter()
    llm = LLM(
        model=model_path,
        load_format="sharded_state",
        quantization=quantization,
        tensor_parallel_size=tensor_parallel_size,
    )
    t1 = time.perf_counter()

    load_time = t1 - t0
    print(f">>> Model loaded in {load_time:.2f} seconds")
    return llm, load_time


def verify_model(llm):
    """Run a quick inference test to verify the model works."""
    print("\n>>> Running verification inference...")
    outputs = llm.generate(["Hello, world! Tell me a fun fact about"])
    generated_text = outputs[0].outputs[0].text
    print(f"    Prompt: 'Hello, world! Tell me a fun fact about'")
    print(f"    Output: '{generated_text[:200]}...'")
    print(">>> Verification passed.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Load a pre-sharded LLM via GDS from Amazon FSx for Lustre"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to pre-sharded model on Amazon FSx for Lustre (e.g. /fsx/model_shards/Llama-3.1-405B-FP8-8way)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=8,
        help="Number of GPUs for tensor parallelism (default: 8)",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="fp8",
        choices=["fp8", "awq", "gptq", "squeezellm", None],
        help="Quantization method (default: fp8)",
    )
    parser.add_argument(
        "--skip-gds-check",
        action="store_true",
        help="Skip the GDS verification check",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the inference verification step",
    )
    args = parser.parse_args()

    # Validate model path exists
    if not os.path.isdir(args.model_path):
        print(f"ERROR: Model path not found: {args.model_path}")
        sys.exit(1)

    # Check GDS
    if not args.skip_gds_check:
        gds_ok = check_gds_active()
        if not gds_ok:
            print("Continuing without GDS (performance will be degraded).\n")

    # Load model
    llm, load_time = load_model(
        args.model_path, args.tensor_parallel_size, args.quantization
    )

    # Verify
    if not args.skip_verify:
        verify_model(llm)

    # Summary
    print("============================================")
    print(f" Model loaded in {load_time:.2f} seconds")
    print(f" Path: {args.model_path}")
    print(f" TP size: {args.tensor_parallel_size}")
    print(f" Quantization: {args.quantization}")
    print("============================================")
    print()
    print("To serve this model:")
    print(f"  vllm serve {args.model_path} \\")
    print(f"    --load-format sharded_state \\")
    print(f"    --quantization {args.quantization} \\")
    print(f"    --tensor-parallel-size {args.tensor_parallel_size}")


if __name__ == "__main__":
    main()

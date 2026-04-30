#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0
#
# gds_load_shards.py — Load pre-sharded safetensors via GDS into GPU memory
#
# Uses fastsafetensors to read each shard directly into GPU HBM via
# NVIDIA GPUDirect Storage, bypassing CPU memory entirely.
#
# Prerequisites:
#   pip install fastsafetensors
#   nvidia-fs kernel module loaded (run 1-setup-gds.sh)
#   FSx for Lustre mounted with EFA (run scripts 2 and 3)
#
# Usage:
#   python gds_load_shards.py /fsx/model_shards/Llama-3.1-405B-BF16-8way
#   python gds_load_shards.py /fsx/model_shards/Llama-3.1-70B-FP8-8way --num-gpus 8

import argparse
import os
import time
import torch
import torch.multiprocessing as mp
from fastsafetensors import SafeTensorsFileLoader


def load_rank(rank, shard_dir, barrier, results):
    """Load one shard into a single GPU via GDS."""
    torch.cuda.set_device(rank)

    # Find this rank's shard file
    shard = os.path.join(shard_dir, f"model-rank-{rank}-part-0.safetensors")
    if not os.path.exists(shard):
        print(f"  Rank {rank}: shard not found: {shard}", flush=True)
        return
    sz_gb = os.path.getsize(shard) / 1e9

    # Open GDS file handle and register GPU buffer
    loader = SafeTensorsFileLoader(pg=None, device=f"cuda:{rank}", nogds=False)
    loader.add_filenames({0: [shard]})
    torch.cuda.synchronize(rank)
    barrier.wait()

    # GDS read: storage → GPU HBM directly
    t0 = time.perf_counter()
    fbuf = loader.copy_files_to_device()
    torch.cuda.synchronize(rank)
    copy_time = time.perf_counter() - t0

    # Extract tensors (sub-millisecond — pointer math into GPU buffer)
    keys = loader.get_keys()
    t1 = time.perf_counter()
    tensors = {k: fbuf.get_tensor(k) for k in keys}
    extract_time = time.perf_counter() - t1

    total = copy_time + extract_time
    print(
        f"  Rank {rank}: {sz_gb:.2f} GB in {total:.3f}s = {sz_gb/total:.2f} GB/s "
        f"({len(tensors)} tensors, extract {extract_time*1000:.1f}ms)",
        flush=True,
    )
    results[rank] = (sz_gb, total, len(tensors))
    fbuf.close()


def main():
    parser = argparse.ArgumentParser(description="Load pre-sharded model via GDS")
    parser.add_argument("shard_dir", help="Path to directory with model-rank-N-part-0.safetensors files")
    parser.add_argument("--num-gpus", type=int, default=8, help="Number of GPUs (default: 8)")
    parser.add_argument("--drop-cache", action="store_true", help="Drop page cache before loading (requires sudo)")
    args = parser.parse_args()

    # Validate
    shards = sorted(f for f in os.listdir(args.shard_dir) if f.endswith(".safetensors"))
    total_gb = sum(os.path.getsize(os.path.join(args.shard_dir, f)) for f in shards) / 1e9
    print(f"Loading {len(shards)} shard files from {args.shard_dir}")
    print(f"Total size: {total_gb:.1f} GB across {args.num_gpus} GPUs\n")

    if args.drop_cache:
        os.system("sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'")

    # Parallel load
    mp.set_start_method("spawn", force=True)
    barrier = mp.Barrier(args.num_gpus)
    manager = mp.Manager()
    results = manager.dict()

    procs = []
    t0 = time.perf_counter()
    for r in range(args.num_gpus):
        p = mp.Process(target=load_rank, args=(r, args.shard_dir, barrier, results))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    wall = time.perf_counter() - t0

    # Summary
    if results:
        max_time = max(v[1] for v in results.values())
        total_tensors = sum(v[2] for v in results.values())
        agg_gbps = total_gb / max_time
        print(f"\nMax per-rank time: {max_time:.3f}s")
        print(f"Aggregate throughput: {agg_gbps:.1f} GB/s")
        print(f"Total tensors loaded: {total_tensors:,}")
        print(f"Wall time (incl process spawn): {wall:.1f}s")


if __name__ == "__main__":
    main()

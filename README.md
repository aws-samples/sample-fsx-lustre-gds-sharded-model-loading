# Accelerate LLM Model Loading with GPUDirect Storage on Amazon FSx for Lustre

This repository contains the infrastructure and setup scripts for the AWS blog post:
**[Accelerate LLM Model Loading and Increase Context Windows with GPU Direct on Amazon FSx for Lustre and Advanced Turbo Quantization](TODO: insert blog URL)**

Load Llama 3.1 405B model weights to 8 GPUs in 6 seconds instead of nearly 3 minutes — a **25x speedup** — by combining Amazon FSx for Lustre, NVIDIA GPUDirect Storage (GDS), and pre-sharded tensor-parallel checkpoints.

## Overview

Traditional LLM model loading is single-threaded and CPU-bound — reading weights from storage, deserializing, optionally quantizing, and copying to each GPU sequentially over PCIe.

This solution eliminates every bottleneck by:

1. **Pre-sharding and pre-quantizing** the model offline into per-GPU tensor-parallel shards
2. **Reading all shards in parallel** via GDS directly from FSx for Lustre into GPU HBM using [fastsafetensors](https://github.com/IBM/fastsafetensors), bypassing CPU memory entirely
3. **Serving immediately** — each GPU has exactly the weights it needs, no all-gather required

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  FSx for Lustre (EFA)                   │
│         Persistent_2 | 1000 MBps/TiB | 20 OSTs         │
│                                                         │
│  shard-0.safetensors  shard-1.safetensors  ...  shard-7 │
└──────┬──────────────────┬─────────────────────────┬─────┘
       │ GDS              │ GDS                     │ GDS
       │ (EFA/SRD)        │ (EFA/SRD)               │ (EFA/SRD)
       ▼                  ▼                         ▼
   ┌──────┐          ┌──────┐                  ┌──────┐
   │ GPU 0│          │ GPU 1│       ...        │ GPU 7│
   │ HBM  │◄─NVLink─►│ HBM  │◄─────NVLink─────►│ HBM  │
   └──────┘          └──────┘                  └──────┘
              P5en / P6e Instance
```

## Measured Performance

Tested on P5en (p5en.48xlarge, 8x H200) with 96 TiB FSx for Lustre Persistent_2 EFA (20 OSTs, ~78 GiB/s GDS read throughput).

### Llama 3.1 405B Instruct (8-way TP, cold cache)

| Loading Method | Total Load Time | Speedup |
|---|---|---|
| Standard vLLM from HF checkpoint (BF16→FP8 quantize at load) | **162.4 s** (2.7 min) | 1x |
| vLLM sharded_state from BF16 shards (no GDS) | **111.0 s** (1.8 min) | 1.5x |
| **GDS parallel load — BF16 shards (812 GB)** | **10.4 s** | **16x** |
| **GDS parallel load — FP8 shards (408 GB)** | **6.4 s** | **25x** |

### Llama 3.1 70B Instruct (8-way TP, cold cache)

| Loading Method | Total Load Time | Speedup |
|---|---|---|
| Standard vLLM from HF checkpoint (BF16→FP8 quantize at load) | **66.1 s** | 1x |
| vLLM sharded_state from BF16 shards (no GDS) | **50.5 s** | 1.3x |
| **GDS parallel load — BF16 shards (141 GB)** | **2.17 s** | **30x** |
| **GDS parallel load — FP8 shards (72 GB)** | **1.28 s** | **52x** |

GDS load times use [fastsafetensors](https://github.com/IBM/fastsafetensors) for direct storage-to-GPU transfer with tensor reconstruction. Throughput scales linearly with filesystem size.

## Repository Structure

```
├── cloudformation/
│   ├── 1-gpu-instance.yaml          # Stack 1: VPC, SG, IAM, GPU instance (deploy first)
│   └── 2-fsx-filesystem.yaml        # Stack 2: FSx Persistent_2 EFA (deploy after GPU stack)
├── scripts/
│   ├── 1-setup-gds.sh               # Build nvidia-fs.ko GDS module → REBOOT REQUIRED
│   ├── 2-configure-lustre-client.sh  # Official AWS EFA setup (--optimized-for-gds)
│   ├── 3-mount-and-tune.sh          # Mount FSx, Lustre tuning, striping
│   └── benchmark-gds.sh             # GDSIO benchmark (8 threads × 16MB per GPU)
├── model-loading/
│   ├── load-sharded-model.py        # vLLM sharded load example
│   ├── quantize_shards_fp8.py       # Offline FP8 quantizer (2D weights only)
│   └── ...                          # GDS loading utilities
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

## Deployment

### Step 1: Deploy Infrastructure (two-stack approach)

```bash
# Deploy GPU stack first (VPC, SG, IAM, instance)
aws cloudformation create-stack --stack-name fsx-lustre-gds-gpu \
    --template-body file://cloudformation/1-gpu-instance.yaml \
    --capabilities CAPABILITY_IAM \
    --parameters ParameterKey=KeyPairName,ParameterValue=my-key-pair \
                 ParameterKey=AvailabilityZone,ParameterValue=us-west-2d \
                 ParameterKey=AmiId,ParameterValue=<DLAMI-ID> \
    --region us-west-2

# Deploy FSx stack (imports subnet/SG from GPU stack)
aws cloudformation create-stack --stack-name fsx-lustre-gds-fsx \
    --template-body file://cloudformation/2-fsx-filesystem.yaml \
    --parameters ParameterKey=GPUStackName,ParameterValue=fsx-lustre-gds-gpu \
    --region us-west-2
```

### Step 2: Configure the Instance

```bash
# 1. Build and install GDS module (survives reboot)
sudo bash scripts/1-setup-gds.sh
sudo reboot

# 2. Configure EFA for Lustre (uses official AWS script)
sudo bash scripts/2-configure-lustre-client.sh

# 3. Mount FSx and tune
sudo bash scripts/3-mount-and-tune.sh <fsx-dns> <mount-name>
```

### Step 3: Validate GDS (Optional)

```bash
sudo bash scripts/benchmark-gds.sh
# Expected on P5en with 96 TiB: ~78-94 GiB/s read
```

### Step 4: Pre-shard and Load

```bash
# Create striped output directory
mkdir -p /fsx/model_shards/Llama-3.1-405B-BF16-8way
lfs setstripe -c -1 -S 16M /fsx/model_shards/Llama-3.1-405B-BF16-8way

# Shard with vLLM (TP=8)
python save_sharded_state.py --model /fsx/models/Llama-3.1-405B-Instruct \
  --output /fsx/model_shards/Llama-3.1-405B-BF16-8way --tensor-parallel-size 8

# Serve with vLLM
vllm serve /fsx/model_shards/Llama-3.1-405B-BF16-8way \
  --load-format sharded_state --tensor-parallel-size 8
```

## Cleanup

```bash
aws cloudformation delete-stack --stack-name fsx-lustre-gds-fsx --region us-west-2
aws cloudformation delete-stack --stack-name fsx-lustre-gds-gpu --region us-west-2
```

## Related Resources

- [Amazon FSx for Lustre](https://aws.amazon.com/fsx/lustre/)
- [Configuring EFA clients for FSx for Lustre](https://docs.aws.amazon.com/fsx/latest/LustreGuide/configure-efa-clients.html)
- [Build and deploy a 1 TB/s file system in under an hour](https://aws.amazon.com/blogs/hpc/build-and-deploy-a-1-tb-s-file-system-in-under-an-hour/)
- [NVIDIA GPUDirect Storage documentation](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html)
- [Amazon EC2 P5en instances](https://aws.amazon.com/ec2/instance-types/p5/)
- [fastsafetensors](https://github.com/IBM/fastsafetensors) — GDS-enabled safetensors loader

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.

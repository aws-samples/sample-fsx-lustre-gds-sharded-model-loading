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
│   ├── gds_load_shards.py           # fastsafetensors GDS parallel loader (main example)
│   ├── save_sharded_state.py        # vLLM TP-aware sharding script
│   ├── quantize_shards_fp8.py       # Offline FP8 quantizer (2D weights only)
│   ├── benchmark_load.py            # vLLM baseline load timer
│   └── load-sharded-model.py        # vLLM sharded serve example
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

## Infrastructure and Setup

This section describes what each component does and how they fit together.

> **💡 Tip:** We recommend using [Kiro CLI](https://kiro.dev) (`kiro-cli chat`) to deploy the
> CloudFormation stacks and run the setup scripts. Kiro can read the templates and scripts,
> execute them on your behalf, and assist with troubleshooting if any errors are encountered
> during deployment — such as capacity issues, security group misconfigurations, or module
> build failures.

### CloudFormation Templates

The infrastructure is split into two stacks so you can secure scarce GPU capacity first, then
create the FSx filesystem (which takes ~25 minutes) without blocking on instance availability.

**`1-gpu-instance.yaml`** (deploy first) creates:
- VPC with a /16 CIDR (required for FSx EFA)
- Public subnet in your chosen AZ
- Security group with self-referencing all-traffic rules (required for FSx EFA validation) plus scoped SSH
- IAM role and instance profile with FSx and S3 access
- Launch Template with all EFA interfaces (16 for P5en, 32 for P6e — auto-detected via conditions)
- GPU instance (on-demand or spot)

**`2-fsx-filesystem.yaml`** (deploy second) creates:
- FSx for Lustre Persistent_2 filesystem with `EfaEnabled: true`
- Imports the subnet and security group from the GPU stack via CloudFormation exports
- Parameterized capacity (default 96 TiB) and throughput (default 1000 MBps/TiB)

> **⚠️ Capacity Block Reservations are NOT supported by CloudFormation.**
>
> P5en and P6e instances are extremely scarce. [EC2 Capacity Blocks](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-blocks-using.html)
> are often the only reliable way to get GPU capacity. However, Capacity Blocks require
> `MarketType=capacity-block` in `InstanceMarketOptions`, which CloudFormation does not
> support as of May 2026.
>
> **If using a Capacity Block:** Deploy the GPU stack with `--disable-rollback` to create
> the networking resources (the instance creation will fail — that's expected). Then launch
> the instance via the AWS CLI as shown in [Option B](#option-b-cli-launch-with-capacity-block-reservation) below.
>
> **If using on-demand or spot:** The CloudFormation template works end-to-end.

### Setup Scripts

The scripts are numbered and must be run in order on the GPU instance. There is a **mandatory reboot** between scripts 1 and 2.

| Script | What it does | Why |
|---|---|---|
| `1-setup-gds.sh` | Builds and installs `nvidia-fs.ko` (GDS kernel module) from source, writes `/etc/cufile.json`, configures module to auto-load on boot | The DLAMI doesn't ship with GDS pre-installed. The module enables direct DMA from EFA to GPU HBM. **Reboot required** after this script for the module to load cleanly. |
| `2-configure-lustre-client.sh` | Downloads and runs the official AWS `configure-efa-fsx-lustre-client` setup script with `--optimized-for-gds` | Configures LNet to use EFA interfaces with NUMA-aware CPU partitioning, creates a systemd service for reboot persistence. Must run **after** reboot so nvidia-fs is loaded first. |
| `3-mount-and-tune.sh` | Mounts the FSx filesystem, applies Lustre client tuning (max_rpcs_in_flight, max_cached_mb), creates model_shards directory with 16M stripe across all OSTs | Tuning parameters and stripe size are matched to the optimal GDS block size (16 MB). |
| `benchmark-gds.sh` | Runs GDSIO with 8 threads × 16 MB per GPU across all 8 GPUs for 60 seconds | Validates GDS is working end-to-end. Expected: ~78–94 GiB/s read on 96 TiB. If you see < 20 GiB/s, EFA routing or module load order is wrong. |

### Model Loading Scripts

| Script | Purpose |
|---|---|
| `gds_load_shards.py` | **Main example.** Parallel GDS load using fastsafetensors — reads safetensors shards directly into GPU HBM and reconstructs tensors. This is what produces the 6.4s / 25x speedup numbers. |
| `save_sharded_state.py` | Shards a HuggingFace model into per-GPU tensor-parallel safetensors files using vLLM. |
| `quantize_shards_fp8.py` | Offline FP8 quantizer — converts BF16 shards to FP8 (2D weight tensors only, preserves norms in BF16). |
| `benchmark_load.py` | Measures vLLM baseline load time (no GDS) for comparison. |
| `load-sharded-model.py` | Example of serving pre-sharded weights with vLLM's `--load-format sharded_state`. |

## Deployment

### Step 1: Deploy Infrastructure (two-stack approach)

> **⚠️ Important: Capacity Block Reservations are NOT supported by CloudFormation.**
>
> P5en and P6e instances are scarce. If you are using an [EC2 Capacity Block reservation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-blocks-using.html)
> (the most reliable way to get GPU capacity), you **cannot** launch the instance via
> CloudFormation. Capacity Blocks require `MarketType=capacity-block` in the
> `InstanceMarketOptions`, which is not supported by `AWS::EC2::Instance` or Launch Templates
> in CloudFormation as of May 2026.
>
> **If using a Capacity Block:** Deploy the GPU CloudFormation stack with `--disable-rollback`
> to create the networking (VPC, subnet, security group, IAM role) — the instance will fail
> but the networking resources will persist. Then launch the instance separately via the
> AWS CLI `run-instances` command as shown in Option B below.
>
> **If using on-demand or spot:** The CloudFormation template works as-is (Option A).

#### Option A: CloudFormation (on-demand or spot only)

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

#### Option B: CLI launch with Capacity Block reservation

Use this when you have a Capacity Block reservation. First deploy the GPU stack with
`--disable-rollback` to create networking, then launch the instance via CLI:

```bash
# 1. Deploy GPU stack for networking only (instance will fail — that's expected)
aws cloudformation create-stack --stack-name fsx-lustre-gds-gpu \
    --template-body file://cloudformation/1-gpu-instance.yaml \
    --capabilities CAPABILITY_IAM --disable-rollback \
    --parameters ParameterKey=KeyPairName,ParameterValue=my-key-pair \
                 ParameterKey=AvailabilityZone,ParameterValue=us-west-2d \
                 ParameterKey=AmiId,ParameterValue=<DLAMI-ID> \
    --region us-west-2

# 2. Get the subnet and security group from the stack outputs
SUBNET=$(aws cloudformation describe-stacks --stack-name fsx-lustre-gds-gpu \
    --query 'Stacks[0].Outputs[?OutputKey==`SubnetId`].OutputValue' --output text --region us-west-2)
SG=$(aws cloudformation describe-stacks --stack-name fsx-lustre-gds-gpu \
    --query 'Stacks[0].Outputs[?OutputKey==`SecurityGroupId`].OutputValue' --output text --region us-west-2)
IAM_PROFILE=$(aws cloudformation describe-stacks --stack-name fsx-lustre-gds-gpu \
    --query 'Stacks[0].Outputs[?OutputKey==`InstanceProfileName`].OutputValue' --output text --region us-west-2)

# 3. Build 16-interface EFA network config (P5en; use 32 for P6e)
python3 -c "
import json
enis = [{'DeviceIndex':0,'NetworkCardIndex':0,'InterfaceType':'efa',
         'SubnetId':'$SUBNET','Groups':['$SG'],'DeleteOnTermination':True}]
for i in range(1, 16):
    enis.append({'DeviceIndex':1,'NetworkCardIndex':i,'InterfaceType':'efa-only',
                 'SubnetId':'$SUBNET','Groups':['$SG'],'DeleteOnTermination':True})
json.dump(enis, open('/tmp/enis.json','w'), indent=2)
"

# 4. Launch with capacity-block market type
aws ec2 run-instances --region us-west-2 \
    --image-id <DLAMI-ID> --instance-type p5en.48xlarge --key-name my-key-pair \
    --iam-instance-profile Name=$IAM_PROFILE \
    --instance-market-options 'MarketType=capacity-block' \
    --capacity-reservation-specification "CapacityReservationTarget={CapacityReservationId=cr-XXXXXXXXXXXXXXXXX}" \
    --network-interfaces file:///tmp/enis.json \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=fsx-gds-blog-p5en}]'

# 5. Allocate and associate an Elastic IP (multi-ENI instances don't auto-assign public IPs)
INSTANCE_ID=<from step 4>
PRIMARY_ENI=$(aws ec2 describe-instances --region us-west-2 --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].NetworkInterfaces[?Attachment.DeviceIndex==`0`].NetworkInterfaceId | [0]' --output text)
ALLOC=$(aws ec2 allocate-address --region us-west-2 --domain vpc --query 'AllocationId' --output text)
aws ec2 associate-address --region us-west-2 --allocation-id $ALLOC --network-interface-id $PRIMARY_ENI

# 6. Deploy FSx stack
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

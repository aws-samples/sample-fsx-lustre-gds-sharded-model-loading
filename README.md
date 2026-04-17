# Accelerate LLM Model Loading with GPUDirect Storage on Amazon FSx for Lustre

This repository contains the infrastructure and setup scripts for the AWS blog post:
**[Accelerate LLM Model Loading with GPU Direct on Amazon FSx for Lustre](TODO: insert blog URL)**

Load large language models (LLMs) like Llama 3.1 405B in under 10 seconds instead of 20+ minutes by combining Amazon FSx for Lustre, NVIDIA GPUDirect Storage (GDS), and pre-sharded tensor-parallel checkpoints.

## Overview

Traditional LLM model loading is single-threaded and CPU-bound — reading weights from storage, deserializing, optionally quantizing on the CPU, and copying to each GPU sequentially over PCIe. For an 800 GB model, this takes ~20 minutes.

This solution eliminates every bottleneck by:

1. **Pre-sharding and pre-quantizing** the model offline into per-GPU tensor-parallel shards
2. **Reading all shards in parallel** via GDS directly from FSx for Lustre into GPU HBM, bypassing CPU memory entirely
3. **Serving immediately** — each GPU has exactly the weights it needs, no all-gather required

The result is a ~120x speedup in model load time on P5en and P6e instances.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  FSx for Lustre (EFA)                   │
│         Persistent_2 | 1000 MBps/TiB | 24 OSTs         │
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

## Prerequisites

- An AWS account with permissions to create VPCs, EC2 instances, FSx filesystems, and IAM roles
- An [Amazon EC2 key pair](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html) for SSH access
- A [Deep Learning AMI](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html) (Ubuntu 22.04/24.04 or Amazon Linux 2023) with NVIDIA drivers and CUDA pre-installed
- Familiarity with Linux command line and AWS services

## Supported Configurations

| Component | Supported Options |
|---|---|
| Instance types | p5.48xlarge, p5e.48xlarge, p5en.48xlarge |
| Operating systems | Amazon Linux 2023 (kernel 6.1+), Ubuntu 22.04/24.04 (kernel 6.8+) |
| FSx for Lustre | Persistent_2 SSD with EFA enabled |
| Networking | Instance and FSx filesystem must be in the same VPC and Availability Zone |

## Repository Structure

```
├── cloudformation/
│   └── fsx-lustre-gds-infrastructure.yaml   # VPC, SG, FSx, GPU instance
├── scripts/
│   ├── 1-setup-gds.sh                       # EFA driver + GDS kernel module (reboot after)
│   ├── 2-configure-lustre-client.sh         # NUMA-aware Lustre networking (post-reboot)
│   ├── 3-mount-and-tune.sh                  # Mount FSx + performance tuning
│   └── benchmark-gds.sh                     # GDSIO performance validation
├── model-loading/
│   └── load-sharded-model.py                # vLLM sharded model load example
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
└── README.md
```

## Deployment

### Step 1: Deploy Infrastructure

Deploy the CloudFormation stack to create the VPC, security group, FSx for Lustre filesystem, and GPU instance:

```bash
# Find the latest Deep Learning AMI in your region
AMI_ID=$(aws ec2 describe-images --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*" \
            "Name=state,Values=available" \
  --query "sort_by(Images, &CreationDate)[-1].ImageId" --output text)
echo "Using AMI: $AMI_ID"

# Find an AZ with P5en capacity in your region
aws ec2 describe-instance-type-offerings \
  --filters Name=instance-type,Values=p5en.48xlarge \
  --location-type availability-zone \
  --query 'InstanceTypeOfferings[].Location' --output text
```

Then deploy the stack, substituting your values:

```bash
aws cloudformation create-stack \
  --stack-name fsx-lustre-gds \
  --template-body file://cloudformation/fsx-lustre-gds-infrastructure.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameters \
    ParameterKey=AvailabilityZone,ParameterValue=us-east-1d \
    ParameterKey=AmiId,ParameterValue=$AMI_ID \
    ParameterKey=KeyPairName,ParameterValue=my-key-pair \
    ParameterKey=SSHCidrBlock,ParameterValue=203.0.113.0/32
```

Wait for the stack to complete (~25-30 minutes, primarily FSx filesystem creation):

```bash
aws cloudformation wait stack-create-complete --stack-name fsx-lustre-gds
```

### Step 2: Configure the Instance

SSH into the GPU instance and copy the scripts:

```bash
# Get the instance IP from stack outputs
INSTANCE_ID=$(aws cloudformation describe-stacks --stack-name fsx-lustre-gds \
  --query 'Stacks[0].Outputs[?OutputKey==`InstanceId`].OutputValue' --output text)

INSTANCE_IP=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

# Copy scripts to the instance
scp -i my-key-pair.pem -r scripts/ model-loading/ ubuntu@${INSTANCE_IP}:~/

# SSH in
ssh -i my-key-pair.pem ubuntu@${INSTANCE_IP}
```

Run the setup scripts in order:

```bash
# 1. Install EFA driver and build GDS kernel module
sudo bash scripts/1-setup-gds.sh

# 2. REBOOT — required for EFA interfaces to activate
sudo reboot
```

After the instance comes back up (~1-2 minutes), SSH back in and continue:

```bash
# 3. Configure NUMA-aware Lustre client networking
sudo bash scripts/2-configure-lustre-client.sh

# 4. Mount FSx for Lustre and apply performance tuning
#    (auto-detects FSx details from CloudFormation if /etc/fsx-config.env exists)
sudo bash scripts/3-mount-and-tune.sh
```

### Step 3: Validate GDS Performance (Optional)

Run the GDSIO benchmark to confirm GDS is delivering expected throughput:

```bash
sudo bash scripts/benchmark-gds.sh 32 32
```

Expected results on P5en: ~190 GiB/s read, ~158 GiB/s write.

### Step 4: Pre-shard Your Model

Pre-shard and pre-quantize the model (one-time offline step):

```bash
# Download or copy your model to FSx
# Example: Llama 3.1 405B in FP16 (~800 GB)

# Create output directory with optimal striping
sudo mkdir -p /fsx/model_shards/Llama-3.1-405B-FP8-8way
sudo lfs setstripe -c -1 -S 64M /fsx/model_shards/Llama-3.1-405B-FP8-8way

# Pre-shard with vLLM (adjust tensor-parallel-size for your GPU count)
python -m vllm.utils.save_sharded_state \
  --model /fsx/models/Llama-3.1-405B \
  --quantization fp8 \
  --tensor-parallel-size 8 \
  --output /fsx/model_shards/Llama-3.1-405B-FP8-8way
```

### Step 5: Load and Serve

```bash
# Load the model and verify
python model-loading/load-sharded-model.py \
  --model-path /fsx/model_shards/Llama-3.1-405B-FP8-8way \
  --tensor-parallel-size 8 \
  --quantization fp8

# Serve for production inference
vllm serve /fsx/model_shards/Llama-3.1-405B-FP8-8way \
  --load-format sharded_state \
  --quantization fp8 \
  --tensor-parallel-size 8
```

## Performance

| Loading Method | Total Load Time | Speedup |
|---|---|---|
| Single-threaded vLLM (FP16→FP8 quantize at load) | ~20 min | 1x |
| Single-threaded vLLM (pre-quantized, no GDS) | ~5 min | ~4x |
| **FSx for Lustre GDS sharded parallel load** | **< 10 s** | **~120x** |

Measured on P5en (p5en.48xlarge) with FSx for Lustre Persistent_2 EFA at 1000 MBps/TiB, 24 OSTs.

## Cleanup

Delete the CloudFormation stack to remove all resources:

```bash
aws cloudformation delete-stack --stack-name fsx-lustre-gds
```

**Note:** The FSx for Lustre filesystem and all data on it will be deleted. Back up any data you want to keep before deleting the stack.

## Related Resources

- [Amazon FSx for Lustre](https://aws.amazon.com/fsx/lustre/)
- [Configuring EFA clients for FSx for Lustre](https://docs.aws.amazon.com/fsx/latest/LustreGuide/configure-efa-clients.html)
- [NVIDIA GPUDirect Storage documentation](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html)
- [Amazon EC2 P5en instances](https://aws.amazon.com/ec2/instance-types/p5/)
- [Amazon EC2 P6e instances](https://aws.amazon.com/ec2/instance-types/p6/)
- [vLLM documentation](https://docs.vllm.ai/)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.

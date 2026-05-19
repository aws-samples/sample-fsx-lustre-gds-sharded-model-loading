# Accelerate LLM Loading with GPUDirect Storage on Amazon FSx for Lustre

This repository contains the infrastructure and setup scripts that accompany the AWS blog post **"Accelerate LLM Loading and Increase Context Windows with GPUDirect on Amazon FSx for Lustre and Advanced Turbo Quantization"**. The link to the published post will be added here once it is live.

Load Llama 3.1 405B model weights to 8 GPUs in approximately 6 seconds instead of nearly 18 minutes — up to a **169x speedup** in our testing — by combining Amazon FSx for Lustre, NVIDIA GPUDirect Storage (GDS), and pre-sharded tensor-parallel checkpoints.

## Introduction

Standard LLM model loading reads weights from storage, deserializes them, optionally quantizes, and copies to each GPU sequentially over PCIe. The default path is single-threaded and CPU-bound.

This solution addresses the key bottlenecks by:

1. **Pre-sharding and pre-quantizing** the model offline into per-GPU tensor-parallel shards
2. **Reading all shards in parallel** via GDS directly from Amazon FSx for Lustre into GPU HBM using [fastsafetensors](https://github.com/foundation-model-stack/fastsafetensors), bypassing CPU memory entirely
3. **Serving immediately** — each GPU loads only its assigned shard, which avoids a separate all-gather step during startup

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Amazon FSx for Lustre (EFA)                   │
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
              p5en.48xlarge Instance
```

## Measured Performance

Tested on P5en (p5en.48xlarge, 8x H200) with 96 TiB Amazon FSx for Lustre Persistent_2 EFA (20 OSTs, ~78 GiB/s GDS read throughput).

### Llama 3.1 405B Instruct (8-way TP, cold cache)

| **Loading Method** | **Total Load Time** | **Speedup** |
|:---|:---|:---|
| Standard vLLM load (BF16 checkpoint, FP8 quantize-at-load, no GDS) | **~18 min** | 1x |
| **GDS parallel load — BF16 shards (812 GB)** | **10.4 s** | **~104x** |
| **GDS parallel load — FP8 shards (408 GB)** | **6.4 s** | **~169x** |

### Llama 3.1 70B Instruct (8-way TP, cold cache)

| **Loading Method** | **Total Load Time** | **Speedup** |
|:---|:---|:---|
| Standard vLLM load (BF16 checkpoint, FP8 quantize-at-load, no GDS) | **~3 min** | 1x |
| **GDS parallel load — BF16 shards (141 GB)** | **2.17 s** | **~83x** |
| **GDS parallel load — FP8 shards (72 GB)** | **1.28 s** | **~141x** |

GDS load times use [fastsafetensors](https://github.com/foundation-model-stack/fastsafetensors) for direct storage-to-GPU transfer with tensor reconstruction. Throughput generally scales with filesystem size in our testing.

> **Note:** The baseline "Standard vLLM load" times were measured with the older vLLM weight loader (pre-V1 engine). The [vLLM V1 engine](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html) (default since vLLM 0.19) introduced parallel weight loading that significantly reduces standard load times. The GDS speedup vs. the current vLLM loader will be lower, but GDS still eliminates the CPU bounce-buffer bottleneck entirely.

## Repository Structure

```
├── cloudformation/
│   ├── 1-gpu-instance.yaml          # Stack 1: VPC, SG, IAM, GPU instance (deploy first)
│   └── 2-fsx-filesystem.yaml        # Stack 2: Amazon FSx for Lustre Persistent_2 EFA (deploy after GPU stack)
├── scripts/
│   ├── 1-setup-gds.sh               # Build nvidia-fs.ko GDS module → REBOOT REQUIRED
│   ├── 2-configure-lustre-client.sh  # Official AWS EFA setup (--optimized-for-gds)
│   ├── 3-mount-and-tune.sh          # Mount Amazon FSx for Lustre, Lustre tuning, striping
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

This section describes the AWS resources required for the solution and the
configuration steps that prepare the instance for GPUDirect Storage. The infrastructure is
deployed using two AWS CloudFormation templates — a GPU instance stack and an
Amazon FSx for Lustre stack — followed by instance configuration scripts
that build the GDS kernel module, configure Elastic Fabric Adapter (EFA)
routing for Lustre, and mount the filesystem with GDS-optimized tuning. The
subsections below cover each component in order.

> **💡 Tip:** Use [Kiro CLI](https://kiro.dev) (`kiro-cli chat`) to deploy the
> CloudFormation stacks and run the setup scripts. Kiro can read the templates and scripts,
> run them on your behalf, and assist with troubleshooting if any errors are encountered
> during deployment — such as capacity issues, security group misconfigurations, or module
> build failures.

### AWS CloudFormation Templates

The infrastructure is split into two stacks so you can secure scarce GPU capacity first, then
create the Amazon FSx for Lustre filesystem (which takes ~25 minutes) without blocking on instance availability.

**`1-gpu-instance.yaml`** (deploy first) creates:
- Amazon Virtual Private Cloud (Amazon VPC) with a /16 CIDR (required for Amazon FSx for Lustre Elastic Fabric Adapter (EFA))
- Public subnet in your chosen AZ
- Security group with self-referencing all-traffic rules (required for Amazon FSx for Lustre EFA validation) plus scoped SSH
- AWS Identity and Access Management (AWS IAM) role and instance profile with Amazon FSx for Lustre and Amazon S3 access
- Launch Template that attaches all 16 Elastic Fabric Adapter (EFA) interfaces available on `p5en.48xlarge`. The count matches the `MaximumNetworkCards` value that `aws ec2 describe-instance-types` reports for this instance type.
- GPU instance (on-demand or spot)

**`2-fsx-filesystem.yaml`** (deploy second) creates:
- Amazon FSx for Lustre Persistent_2 filesystem with `EfaEnabled: true`
- Imports the subnet and security group from the GPU stack via CloudFormation exports
- Parameterized capacity (default 96 TiB) and throughput (default 1000 MBps/TiB)

> **ℹ️ Capacity Block reservations with CloudFormation**
>
> P5en instances have limited on-demand availability. [Capacity Blocks for ML](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-blocks-using.html)
> can be a reliable way to reserve GPU capacity in advance.
>
> CloudFormation supports Capacity Blocks through `AWS::EC2::LaunchTemplate` by setting
> [`InstanceMarketOptions.MarketType`](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-properties-ec2-launchtemplate-instancemarketoptions.html)
> to `capacity-block` along with a `CapacityReservationSpecification`. To use this template with
> a Capacity Block, add those fields to the Launch Template and provide your reservation ID.
>
> If you prefer to launch via the AWS CLI (useful for ad-hoc testing), [Option B](#option-b-cli-launch-with-capacity-block-reservation)
> below walks through that workflow.

### Setup Scripts

The scripts are numbered and must be run in order on the GPU instance. There is a **mandatory reboot** between scripts 1 and 2.

| **Script** | **What it does** | **Why** |
|:---|:---|:---|
| `1-setup-gds.sh` | Builds and installs `nvidia-fs.ko` (GDS kernel module) from source, writes `/etc/cufile.json`, configures module to auto-load on boot | The AWS Deep Learning AMIs (DLAMI) does not include GDS pre-installed. The module provides direct DMA from EFA to GPU HBM. **Reboot required** after this script for the module to load cleanly. |
| `2-configure-lustre-client.sh` | Downloads and runs the official AWS `configure-efa-fsx-lustre-client` setup script with `--optimized-for-gds` | Configures LNet to use EFA interfaces with NUMA-aware CPU partitioning, creates a systemd service for reboot persistence. Must run **after** reboot so nvidia-fs is loaded first. |
| `3-mount-and-tune.sh` | Mounts Amazon FSx for Lustre on the instance with Lustre client parameters tuned for GDS (16 MB stripe, matched `max_rpcs_in_flight`/`max_cached_mb` values). | Tuning parameters and stripe size are matched to the optimal GDS block size (16 MB). |
| `benchmark-gds.sh` | Runs GDSIO with 8 threads × 16 MB per GPU across all 8 GPUs for 60 seconds | Validates GDS is working end-to-end. Expected: ~78–94 GiB/s read on 96 TiB. If you see < 20 GiB/s, EFA routing or module load order is wrong. |

### Model Loading Scripts

The following table lists the Python helpers that drive the sharded load and baseline comparisons on the instance.

| **Script** | **Purpose** |
|:---|:---|
| `gds_load_shards.py` | **Main example.** Parallel GDS load using fastsafetensors — reads safetensors shards directly into GPU HBM and reconstructs tensors. This is what produces the 6.4s / 169x speedup numbers. |
| `save_sharded_state.py` | Shards a HuggingFace model into per-GPU tensor-parallel safetensors files using vLLM. |
| `quantize_shards_fp8.py` | Offline FP8 quantizer — converts BF16 shards to FP8 (2D weight tensors only, preserves norms in BF16). |
| `benchmark_load.py` | Measures vLLM baseline load time (no GDS) for comparison. |
| `load-sharded-model.py` | Example of serving pre-sharded weights with vLLM's `--load-format sharded_state`. |

## Prerequisites

Before deploying this solution, verify you have:

- An AWS account with permissions to create Amazon Virtual Private Cloud (Amazon VPC) networks, Amazon Elastic Compute Cloud (Amazon EC2) instances, Amazon FSx for Lustre filesystems, AWS Identity and Access Management (AWS IAM) roles, and AWS CloudFormation stacks
- AWS Command Line Interface (AWS CLI) version 2.x or later, installed and configured with appropriate credentials
- Python 3.8 or later for running the model loading scripts
- An [Amazon Elastic Compute Cloud (Amazon EC2) key pair](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-key-pairs.html) in your target region
- [AWS Deep Learning AMIs](https://docs.aws.amazon.com/dlami/latest/devguide/what-is-dlami.html) (DLAMI) with NVIDIA drivers and CUDA 12.x (Ubuntu 24.04 recommended)
- Access to `p5en.48xlarge` GPU instances — consider requesting a [Capacity Blocks for ML](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-blocks-using.html) reservation due to limited on-demand availability
- Sufficient service quotas for your target instance type and Amazon FSx for Lustre storage capacity in your target region
- At least 1 TB of Amazon FSx for Lustre storage for model weights (Llama 3.1 405B requires ~800 GB in BF16)

## Deployment

Deploy the solution in two phases. First, provision the GPU instance and the surrounding networking with Stack 1 so you can secure scarce GPU capacity as soon as it is available. Then create the Amazon FSx for Lustre filesystem in that same subnet and security group with Stack 2. Running these in sequence means the filesystem build (which takes roughly 20 to 30 minutes) does not block the instance launch, and the instance does not sit idle waiting for the filesystem to come online.

### Step 1: Deploy Infrastructure (two-stack approach)

This step creates the networking, security, Amazon FSx for Lustre
filesystem, and the GPU instance. Two deployment options are available
depending on how you plan to source GPU capacity: Option A uses AWS
CloudFormation for on-demand or spot instances, and Option B uses the
AWS CLI to launch against a Capacity Blocks for ML reservation.

> **ℹ️ Capacity Block options**
>
> P5en on-demand capacity is limited. If you have a [Capacity Blocks for ML](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-blocks-using.html) reservation
> (a reliable way to reserve GPU capacity in advance), you have two options:
>
> 1. **Extend the CloudFormation template** to set
>    [`InstanceMarketOptions.MarketType`](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-properties-ec2-launchtemplate-instancemarketoptions.html)
>    to `capacity-block` on the Launch Template and pass your reservation ID via
>    `CapacityReservationSpecification`.
> 2. **Use the AWS CLI workflow** in Option B below — deploy the CloudFormation networking
>    stack first, then launch the instance via `aws ec2 run-instances`. This is handy for ad-hoc
>    testing without editing templates.
>
> For on-demand or spot, Option A works as-is.

#### Option A: AWS CloudFormation (on-demand or spot only)

```bash
# Find the latest AWS Deep Learning AMIs (DLAMI) for Ubuntu 24.04 with NVIDIA drivers:
DLAMI_ID=$(aws ec2 describe-images --region us-west-2 \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*" \
            "Name=state,Values=available" \
  --query "sort_by(Images, &CreationDate)[-1].ImageId" --output text)
echo "Using DLAMI: $DLAMI_ID"

# Deploy GPU stack first (VPC, SG, IAM, instance)
aws cloudformation create-stack --stack-name fsx-lustre-gds-gpu \
    --template-body file://cloudformation/1-gpu-instance.yaml \
    --capabilities CAPABILITY_IAM \
    --parameters ParameterKey=KeyPairName,ParameterValue=my-key-pair \
                 ParameterKey=AvailabilityZone,ParameterValue=us-west-2d \
                 ParameterKey=AmiId,ParameterValue=$DLAMI_ID \
    --region us-west-2

# Deploy FSx for Lustre stack (imports subnet/SG from GPU stack)
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
DLAMI_ID=$(aws ec2 describe-images --region us-west-2 \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*" \
            "Name=state,Values=available" \
  --query "sort_by(Images, &CreationDate)[-1].ImageId" --output text)
aws cloudformation create-stack --stack-name fsx-lustre-gds-gpu \
    --template-body file://cloudformation/1-gpu-instance.yaml \
    --capabilities CAPABILITY_IAM --disable-rollback \
    --parameters ParameterKey=KeyPairName,ParameterValue=my-key-pair \
                 ParameterKey=AvailabilityZone,ParameterValue=us-west-2d \
                 ParameterKey=AmiId,ParameterValue=$DLAMI_ID \
    --region us-west-2

# 2. Get the subnet ID from the stack outputs
SUBNET=$(aws cloudformation describe-stacks --stack-name fsx-lustre-gds-gpu \
    --query 'Stacks[0].Outputs[?OutputKey==`SubnetId`].OutputValue' --output text --region us-west-2)

# 3. Get the security group ID from the stack outputs
SG=$(aws cloudformation describe-stacks --stack-name fsx-lustre-gds-gpu \
    --query 'Stacks[0].Outputs[?OutputKey==`SecurityGroupId`].OutputValue' --output text --region us-west-2)

# 4. Get the IAM instance profile name from the stack outputs
IAM_PROFILE=$(aws cloudformation describe-stacks --stack-name fsx-lustre-gds-gpu \
    --query 'Stacks[0].Outputs[?OutputKey==`InstanceProfileName`].OutputValue' --output text --region us-west-2)

# 5. Build the 16-interface EFA network config for p5en.48xlarge
#    (matches MaximumNetworkCards=16 on this instance type).
python3 -c "
import json
enis = [{'DeviceIndex':0,'NetworkCardIndex':0,'InterfaceType':'efa',
         'SubnetId':'$SUBNET','Groups':['$SG'],'DeleteOnTermination':True}]
for i in range(1, 16):
    enis.append({'DeviceIndex':1,'NetworkCardIndex':i,'InterfaceType':'efa-only',
                 'SubnetId':'$SUBNET','Groups':['$SG'],'DeleteOnTermination':True})
json.dump(enis, open('/tmp/enis.json','w'), indent=2)
"

# 6. Launch with capacity-block market type and capture the instance ID
INSTANCE_ID=$(aws ec2 run-instances --region us-west-2 \
    --image-id $DLAMI_ID --instance-type p5en.48xlarge --key-name my-key-pair \
    --iam-instance-profile Name=$IAM_PROFILE \
    --instance-market-options 'MarketType=capacity-block' \
    --capacity-reservation-specification "CapacityReservationTarget={CapacityReservationId=cr-XXXXXXXXXXXXXXXXX}" \
    --network-interfaces file:///tmp/enis.json \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=fsx-gds-blog-p5en}]' \
    --query 'Instances[0].InstanceId' --output text)

# 7. Get the primary network interface ID
PRIMARY_ENI=$(aws ec2 describe-instances --region us-west-2 --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].NetworkInterfaces[?Attachment.DeviceIndex==`0`].NetworkInterfaceId | [0]' --output text)

# 8. Allocate an Elastic IP (multi-ENI instances don't auto-assign public IPs)
ALLOC=$(aws ec2 allocate-address --region us-west-2 --domain vpc --query 'AllocationId' --output text)
# 9. Associate the Elastic IP with the primary network interface
aws ec2 associate-address --region us-west-2 --allocation-id $ALLOC --network-interface-id $PRIMARY_ENI

# 10. Deploy FSx for Lustre stack
aws cloudformation create-stack --stack-name fsx-lustre-gds-fsx \
    --template-body file://cloudformation/2-fsx-filesystem.yaml \
    --parameters ParameterKey=GPUStackName,ParameterValue=fsx-lustre-gds-gpu \
    --region us-west-2
```

### Step 2: Configure the Instance

```bash
# 1. Build and install the GDS module (survives reboot):
sudo bash scripts/1-setup-gds.sh

# 2. Reboot the instance so the module loads cleanly:
sudo reboot

# 3. After the instance is back, configure EFA for Lustre
#    (uses the official AWS script):
sudo bash scripts/2-configure-lustre-client.sh

# 4. Retrieve the Amazon FSx for Lustre DNS name from the stack outputs:
FSX_DNS=$(aws cloudformation describe-stacks --stack-name fsx-lustre-gds-fsx \
  --query 'Stacks[0].Outputs[?OutputKey==`FSxFilesystemDNS`].OutputValue' --output text --region us-west-2)

# 5. Retrieve the mount name (required by the Lustre client):
MOUNT_NAME=$(aws cloudformation describe-stacks --stack-name fsx-lustre-gds-fsx \
  --query 'Stacks[0].Outputs[?OutputKey==`FSxMountName`].OutputValue' --output text --region us-west-2)

# 6. Mount Amazon FSx for Lustre on the instance:
sudo bash scripts/3-mount-and-tune.sh $FSX_DNS $MOUNT_NAME
```

### Step 3: Validate GDS (Optional)

```bash
sudo bash scripts/benchmark-gds.sh
# Expected on P5en with 96 TiB: ~78-94 GiB/s read
```

### Step 4: Download the Model

Llama 3.1 405B is a gated model on Hugging Face. Request access on the
[model page](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct)
before running the commands below. The download is approximately 800 GB
and can take several hours depending on network throughput.

```bash
# 1. Install the Hugging Face CLI:
pip install huggingface-hub

# 2. Install the faster transfer backend:
pip install hf_transfer

# 3. Authenticate with Hugging Face (paste your token when prompted):
huggingface-cli login

# 4. Download the model to the mounted Amazon FSx for Lustre filesystem:
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
    meta-llama/Llama-3.1-405B-Instruct \
    --local-dir /fsx/models/Llama-3.1-405B-Instruct
```

### Step 5: Pre-shard and Load

```bash
# 1. Create the output directory:
mkdir -p /fsx/model_shards/Llama-3.1-405B-BF16-8way

# 2. Configure Lustre striping on the directory (files inherit the stripe
#    configuration at creation time):
lfs setstripe -c -1 -S 16M /fsx/model_shards/Llama-3.1-405B-BF16-8way

# 3. Shard the model with vLLM (TP=8):
python save_sharded_state.py --model /fsx/models/Llama-3.1-405B-Instruct \
  --output /fsx/model_shards/Llama-3.1-405B-BF16-8way --tensor-parallel-size 8

# 4. Serve with vLLM:
vllm serve /fsx/model_shards/Llama-3.1-405B-BF16-8way \
  --load-format sharded_state --tensor-parallel-size 8
```

## Cleanup

**💰 Cost Note:** The Amazon FSx for Lustre filesystem and GPU instances incur significant charges while running. P5en is among the higher-priced EC2 instance types; see [Amazon EC2 P5 pricing](https://aws.amazon.com/ec2/instance-types/p5/) and [Amazon FSx for Lustre pricing](https://aws.amazon.com/fsx/lustre/pricing/) for current rates in your target region. Delete resources promptly when not in use.

**⚠️ WARNING:** The following commands permanently delete all resources, including the Amazon FSx for Lustre filesystem and any model shards stored on it. Back up any data you want to retain before proceeding.

### If you used Option B (Capacity Block CLI launch):

```bash
# 1. Terminate the manually launched instance
aws ec2 terminate-instances --instance-ids <INSTANCE_ID> --region us-west-2

# 2. Wait for the instance to finish terminating
aws ec2 wait instance-terminated --instance-ids <INSTANCE_ID> --region us-west-2

# 3. Release the Elastic IP
aws ec2 release-address --allocation-id <ALLOC_ID> --region us-west-2
```

> **Note on EBS volumes:** The root EBS volume attached to the instance is
> created with `DeleteOnTermination=true` in the Launch Template, so it is
> automatically removed when the instance terminates. No separate cleanup
> step is required for the volume.

### Delete CloudFormation stacks (all deployments):

```bash
# 1. Delete the Amazon FSx for Lustre stack first (filesystem deletion takes ~5 minutes)
aws cloudformation delete-stack --stack-name fsx-lustre-gds-fsx --region us-west-2

# 2. Wait for the Amazon FSx for Lustre stack deletion to complete
aws cloudformation wait stack-delete-complete --stack-name fsx-lustre-gds-fsx --region us-west-2

# 3. Delete the GPU and networking stack
aws cloudformation delete-stack --stack-name fsx-lustre-gds-gpu --region us-west-2
```

## Conclusion

This solution demonstrates how combining Amazon FSx for Lustre with NVIDIA GPUDirect Storage can accelerate LLM model loading by up to 169x in our testing — reducing Llama 3.1 405B load times from nearly 18 minutes to 6 seconds on a 96 TiB filesystem. By pre-sharding models and loading directly to GPU memory in parallel via [fastsafetensors](https://github.com/foundation-model-stack/fastsafetensors), you can reduce CPU bottlenecks and help achieve faster iteration during development and lower cold-start latency in production.

Throughput generally scales with filesystem size in our testing — a larger filesystem can deliver proportionally faster loads.

To get started, deploy the CloudFormation templates in this repository and follow the setup scripts. For questions or contributions, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Related Resources

- [Amazon FSx for Lustre](https://aws.amazon.com/fsx/lustre/)
- [Configuring EFA clients for Amazon FSx for Lustre](https://docs.aws.amazon.com/fsx/latest/LustreGuide/configure-efa-clients.html)
- [Build and deploy a 1 TB/s file system in under an hour](https://aws.amazon.com/blogs/hpc/build-and-deploy-a-1-tb-s-file-system-in-under-an-hour/)
- [NVIDIA GPUDirect Storage documentation](https://docs.nvidia.com/gpudirect-storage/release-notes/index.html)
- [Amazon EC2 P5en instances](https://aws.amazon.com/ec2/instance-types/p5/)
- [fastsafetensors](https://github.com/foundation-model-stack/fastsafetensors) — GDS-enabled safetensors loader

## Important

This is sample code, for non-production usage. You should work with your security and legal teams to meet your organizational security, regulatory and compliance requirements before deployment.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.

# Known Issues and Workarounds

## P5en Capacity

P5en on-demand instances are rarely available. You will likely need a
**capacity block reservation** to get one. Capacity blocks use a distinct
market type that the CloudFormation `AWS::EC2::Instance` resource does not
support directly. Use the AWS CLI instead:

```bash
# 1. Deploy the GPU CFT with --disable-rollback so networking is created
#    even though the instance will fail without capacity.
aws cloudformation create-stack --stack-name fsx-lustre-gds-gpu \
    --template-body file://cloudformation/1-gpu-instance.yaml \
    --capabilities CAPABILITY_IAM --disable-rollback \
    --parameters ParameterKey=KeyPairName,ParameterValue=my-key \
                 ParameterKey=AvailabilityZone,ParameterValue=us-west-2d \
                 ParameterKey=AmiId,ParameterValue=<AMI> \
    --region us-west-2

# 2. After the stack creates networking but fails on the instance,
#    launch via CLI with the capacity block market type:
aws ec2 run-instances --region us-west-2 \
  --image-id <AMI> --instance-type p5en.48xlarge --key-name my-key \
  --iam-instance-profile Name=<stack-name>-instance-role \
  --instance-market-options 'MarketType=capacity-block' \
  --capacity-reservation-specification \
    "CapacityReservationTarget={CapacityReservationId=cr-XXXX}" \
  --network-interfaces file:///tmp/enis.json \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3"}}]'
```

You must build the `enis.json` file with 16 EFA interfaces referencing the
subnet and security group created by the CFT. See the project context or
README for the full Python snippet.

## cufile.json Pinned Memory

The default `max_device_pinned_mem_size_kb` in `1-setup-gds.sh` is set to
512 GB (536870912 KB). If you see `cuFileBufRegister error 5030` when loading
large shards, the per-GPU buffer exceeds this limit. Increase it in
`/etc/cufile.json` and reload nvidia-fs:

```bash
sudo rmmod nvidia_fs && sudo modprobe nvidia_fs
```

## fastsafetensors Process Exit Segfault

fastsafetensors may segfault at Python process exit (during garbage
collection). This is a cleanup ordering issue between Python GC and the CUDA
driver — it does not affect the loaded tensors or a long-running serving
process. You can safely ignore it.

## GDS Stats Counters Show Zero

`/proc/driver/nvidia-fs/stats` may show `Ops: Read=0 Write=0` even when GDS
is actively working. This is a counting artifact in nvidia-fs, not a fallback
to the CPU bounce-buffer path. To verify GDS is truly active, enable cufile
DEBUG logging:

```json
{
  "logging": { "dir": "/tmp", "level": "DEBUG" }
}
```

Then look for `nvfs_io_submit ... is_unaligned 0` and
`Compatibility Mode: 0` in `/tmp/cufile_*.log`.

## EFA Interface Count

The official AWS `setup.sh --optimized-for-gds` script now configures all 16
EFA interfaces on P5en (earlier versions used 8). Both configurations work.
The `osc.*.import` field will show `current_connection: <ip>@tcp` — this is
normal and does not mean EFA is inactive. Verify EFA traffic with:

```bash
sudo lnetctl net show -v 2 | grep -A2 send_count
```

EFA NIDs should show send/recv counts much higher than TCP.

## vLLM Does Not Use GDS Natively

vLLM's `--load-format sharded_state` uses standard CPU-based safetensors I/O.
The GDS acceleration comes from using `fastsafetensors` (see
`model-loading/gds_load_shards.py`). The vLLM sharded_state path is still
faster than loading from the original HF checkpoint because it eliminates
deserialization and per-GPU splitting overhead.

## Lustre Stripe Size

The optimal GDS block size on FSx for Lustre is 16 MB (validated via GDSIO
sweep). The `3-mount-and-tune.sh` script sets the model_shards directory to
`-S 16M`. If you created the directory with a different stripe size, recreate
it — Lustre stripe settings are set at directory creation time and inherited
by new files.

## FP8 Quantization Compatibility

The included `quantize_shards_fp8.py` produces FP8 shards that work with
fastsafetensors GDS loading but are **not compatible** with vLLM's
`--quantization fp8` flag. vLLM expects specific tensor naming conventions
for FP8 weights. For vLLM serving, use the BF16 shards with
`--load-format sharded_state` (no `--quantization` flag).

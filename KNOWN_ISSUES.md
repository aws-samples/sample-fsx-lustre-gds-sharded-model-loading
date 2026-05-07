# Known Issues and Workarounds

## P5en Capacity

Capacity Blocks for ML use a distinct market type. You can provision a
capacity-block instance with AWS CloudFormation by using `AWS::EC2::LaunchTemplate`
and setting [`InstanceMarketOptions.MarketType`](https://docs.aws.amazon.com/AWSCloudFormation/latest/TemplateReference/aws-properties-ec2-launchtemplate-instancemarketoptions.html)
to `capacity-block` along with a `CapacityReservationSpecification`. The
`AWS::EC2::Instance` resource does not accept this market type inline, so the
LaunchTemplate path is the supported way to do it in CloudFormation.

If you prefer to avoid editing the template, launch the instance through the
AWS CLI instead:

```bash
# 1. Deploy the GPU CFT with --disable-rollback so networking is created
#    even though the instance will fail without capacity.
AMI=$(aws ec2 describe-images --region us-west-2 \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)*" \
            "Name=state,Values=available" \
  --query "sort_by(Images, &CreationDate)[-1].ImageId" --output text)
aws cloudformation create-stack --stack-name fsx-lustre-gds-gpu \
    --template-body file://cloudformation/1-gpu-instance.yaml \
    --capabilities CAPABILITY_IAM --disable-rollback \
    --parameters ParameterKey=KeyPairName,ParameterValue=my-key \
                 ParameterKey=AvailabilityZone,ParameterValue=us-west-2d \
                 ParameterKey=AmiId,ParameterValue=$AMI \
    --region us-west-2

# 2. Get the subnet ID from the (partially failed) stack:
SUBNET=$(aws cloudformation describe-stacks --stack-name fsx-lustre-gds-gpu \
  --query 'Stacks[0].Outputs[?OutputKey==`SubnetId`].OutputValue' --output text --region us-west-2)

# 3. Get the security group ID:
SG=$(aws cloudformation describe-stacks --stack-name fsx-lustre-gds-gpu \
  --query 'Stacks[0].Outputs[?OutputKey==`SecurityGroupId`].OutputValue' --output text --region us-west-2)

# 4. Build the 16 Elastic Fabric Adapter (EFA) network interfaces JSON:
python3 -c "
import json
enis = [{'DeviceIndex':0,'NetworkCardIndex':0,'InterfaceType':'efa',
         'SubnetId':'$SUBNET','Groups':['$SG'],'DeleteOnTermination':True}]
for i in range(1, 16):
    enis.append({'DeviceIndex':1,'NetworkCardIndex':i,'InterfaceType':'efa-only',
                 'SubnetId':'$SUBNET','Groups':['$SG'],'DeleteOnTermination':True})
json.dump(enis, open('/tmp/enis.json','w'), indent=2)
print(f'Wrote {len(enis)} interfaces to /tmp/enis.json')
"

# 5. Launch the instance with capacity block market type:
INSTANCE_ID=$(aws ec2 run-instances --region us-west-2 \
  --image-id $AMI --instance-type p5en.48xlarge --key-name my-key \
  --iam-instance-profile Name=fsx-lustre-gds-gpu-instance-role \
  --instance-market-options 'MarketType=capacity-block' \
  --capacity-reservation-specification \
    "CapacityReservationTarget={CapacityReservationId=cr-XXXX}" \
  --network-interfaces file:///tmp/enis.json \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":500,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=fsx-gds-p5en}]' \
  --query 'Instances[0].InstanceId' --output text)
echo "Instance: $INSTANCE_ID"

# 6. Look up the primary network interface (required for the EIP association
#    in the next step — AssociatePublicIpAddress is not supported with multiple
#    network interfaces):
PRIMARY_ENI=$(aws ec2 describe-instances --region us-west-2 \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].NetworkInterfaces[?Attachment.DeviceIndex==`0`].NetworkInterfaceId | [0]' \
  --output text)

# 7. Allocate an Elastic IP:
ALLOC_ID=$(aws ec2 allocate-address --region us-west-2 --domain vpc --query 'AllocationId' --output text)

# 8. Associate the Elastic IP with the primary network interface:
aws ec2 associate-address --region us-west-2 --allocation-id $ALLOC_ID --network-interface-id $PRIMARY_ENI

# 9. Retrieve the public IP address:
PUBLIC_IP=$(aws ec2 describe-addresses --region us-west-2 --allocation-ids $ALLOC_ID --query 'Addresses[0].PublicIp' --output text)

# 10. Display the SSH connection command:
echo "SSH: ssh -i my-key.pem ubuntu@$PUBLIC_IP"
```

**Important:** The instance launched this way is NOT managed by CloudFormation.
You must terminate it manually (`aws ec2 terminate-instances`) and release the
EIP (`aws ec2 release-address`) when done. The FSx for Lustre stack and GPU stack
networking can still be deleted via `aws cloudformation delete-stack`.

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
driver — the segfault occurs after all tensor data has finished loading and
does not affect the loaded tensors or a long-running serving process.

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
faster than loading from the original HF checkpoint because it avoids
deserialization and per-GPU splitting overhead.

## Lustre Stripe Size

The optimal GDS block size on Amazon FSx for Lustre is 16 MB (validated via GDSIO
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

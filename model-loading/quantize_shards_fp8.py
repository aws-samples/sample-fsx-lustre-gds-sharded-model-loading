"""Convert BF16 sharded safetensors to FP8 (E4M3) with per-tensor scales.
Only quantizes 2D weight tensors (linear layers). Leaves norms, biases,
and embeddings in their original dtype."""
import os, sys, time, glob
import torch
from safetensors.torch import load_file, save_file
import shutil

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)

for f in glob.glob(os.path.join(src, "*")):
    if os.path.isdir(f) or f.endswith(".safetensors"):
        continue
    shutil.copy2(f, dst)
    print(f"Copied {os.path.basename(f)}")

shard_files = sorted(glob.glob(os.path.join(src, "*.safetensors")))
print(f"\nQuantizing {len(shard_files)} shard files...")
t0 = time.time()

for i, sf in enumerate(shard_files):
    tensors = load_file(sf)
    quantized = {}
    for name, tensor in tensors.items():
        # Only quantize 2D weight tensors (linear layers)
        if tensor.ndim == 2 and tensor.dtype in (torch.bfloat16, torch.float16, torch.float32):
            amax = tensor.abs().max().float()
            scale = amax / torch.finfo(torch.float8_e4m3fn).max
            fp8_tensor = (tensor.float() / scale).to(torch.float8_e4m3fn)
            quantized[name] = fp8_tensor
            quantized[name.replace(".weight", ".weight_scale")] = scale.unsqueeze(0)
        else:
            quantized[name] = tensor
    save_file(quantized, os.path.join(dst, os.path.basename(sf)))
    print(f"  [{i+1}/{len(shard_files)}] {os.path.basename(sf)} -> {time.time()-t0:.0f}s")

print(f"\nFP8 quantization completed in {time.time()-t0:.0f} seconds")

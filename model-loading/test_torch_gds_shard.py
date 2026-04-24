import torch
import torch.cuda.gds as gds
import time, os

shard = "/fsx/model_shards/Llama-3.1-405B-BF16-8way/model-rank-0-part-0.safetensors"
file_size = os.path.getsize(shard)
print(f"File: {file_size/1e9:.2f} GB")

os.system("sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'")

storage = torch.UntypedStorage(file_size, device=torch.device("cuda:0"))
gds.gds_register_buffer(storage)

t0 = time.perf_counter()
gds_file = gds.GdsFile(shard, os.O_RDONLY)
gds_file.load_storage(storage)
t1 = time.perf_counter()

gds.gds_deregister_buffer(storage)
print(f"PyTorch GDS: {file_size/1e9:.2f} GB in {t1-t0:.3f}s = {file_size/1e9/(t1-t0):.2f} GB/s")

with open("/proc/driver/nvidia-fs/stats") as sf:
    for line in sf:
        l = line.strip()
        if "Reads" in l or "Ops" in l or "err" in l:
            print(l)

os._exit(0)

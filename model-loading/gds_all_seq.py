import torch, torch.cuda.gds as gds, os, glob, time, sys

model_dir = sys.argv[1]
total_bytes = 0
t0 = time.perf_counter()

for rank in range(8):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    files = sorted(glob.glob(os.path.join(model_dir, f"model-rank-{rank}-part-*.safetensors")))
    rank_bytes = 0
    for f in files:
        fsize = os.path.getsize(f)
        storage = torch.UntypedStorage(fsize, device=device)
        gds.gds_register_buffer(storage)
        gds.GdsFile(f, os.O_RDONLY).load_storage(storage)
        gds.gds_deregister_buffer(storage)
        rank_bytes += fsize
        del storage
    total_bytes += rank_bytes
    elapsed = time.perf_counter() - t0
    print(f"Rank {rank}: {rank_bytes/1e9:.2f} GB ({elapsed:.1f}s elapsed)")

t1 = time.perf_counter()
print(f"\nAll 8 GPUs: {total_bytes/1e9:.2f} GB in {t1-t0:.3f}s = {total_bytes/1e9/(t1-t0):.2f} GB/s")

with open("/proc/driver/nvidia-fs/stats") as sf:
    for line in sf:
        l = line.strip()
        if "Reads" in l or "Ops" in l or "err" in l:
            print(l)
os._exit(0)

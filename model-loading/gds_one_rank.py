import torch, torch.cuda.gds as gds, os, glob, sys, time

rank = int(sys.argv[1])
model_dir = sys.argv[2]
torch.cuda.set_device(rank)
device = torch.device(f"cuda:{rank}")

files = sorted(glob.glob(os.path.join(model_dir, f"model-rank-{rank:05d}-of-*.safetensors")))
if not files:
    files = sorted(glob.glob(os.path.join(model_dir, f"model-rank-{rank}-part-*.safetensors")))

total = 0
t0 = time.perf_counter()
for f in files:
    fsize = os.path.getsize(f)
    storage = torch.UntypedStorage(fsize, device=device)
    gds.gds_register_buffer(storage)
    gds.GdsFile(f, os.O_RDONLY).load_storage(storage)
    gds.gds_deregister_buffer(storage)
    total += fsize
    del storage
t1 = time.perf_counter()
print(f"Rank {rank}: {total/1e9:.2f} GB in {t1-t0:.1f}s = {total/1e9/(t1-t0):.2f} GB/s")
os._exit(0)

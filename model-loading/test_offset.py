import torch, torch.cuda.gds as gds, time, os, struct

f = "/fsx/model_shards/Llama-3.1-405B-BF16-8way/model-rank-0-part-0.safetensors"
fsize = os.path.getsize(f)

# Test 1: whole file, no offset
storage = torch.UntypedStorage(fsize, device=torch.device("cuda:0"))
gds.gds_register_buffer(storage)
t0 = time.perf_counter()
gds.GdsFile(f, os.O_RDONLY).load_storage(storage)
t1 = time.perf_counter()
gds.gds_deregister_buffer(storage)
del storage
print(f"Whole file: {fsize/1e9:.2f} GB in {t1-t0:.3f}s = {fsize/1e9/(t1-t0):.2f} GB/s")

# Test 2: with offset
with open(f, "rb") as fh:
    hlen = struct.unpack("<Q", fh.read(8))[0]
data_offset = 8 + hlen
data_size = fsize - data_offset

storage2 = torch.UntypedStorage(data_size, device=torch.device("cuda:0"))
gds.gds_register_buffer(storage2)
t0 = time.perf_counter()
gds.GdsFile(f, os.O_RDONLY).load_storage(storage2, offset=data_offset)
t1 = time.perf_counter()
gds.gds_deregister_buffer(storage2)
del storage2
print(f"With offset ({data_offset}): {data_size/1e9:.2f} GB in {t1-t0:.3f}s = {data_size/1e9/(t1-t0):.2f} GB/s")

os._exit(0)

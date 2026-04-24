"""Consolidate multi-part shards into one safetensors file per rank."""
import glob, json, os, struct, sys, time

model_dir = sys.argv[1]
out_dir = sys.argv[2] if len(sys.argv) > 2 else model_dir + "-consolidated"
os.makedirs(out_dir, exist_ok=True)

for rank in range(8):
    files = sorted(glob.glob(os.path.join(model_dir, f"model-rank-{rank}-part-*.safetensors")))
    if not files:
        continue

    # Collect all tensor metadata and data
    all_meta = {}
    all_data = bytearray()
    for f in files:
        with open(f, "rb") as fh:
            hlen = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(hlen))
            data = fh.read()
        base = len(all_data)
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            all_meta[name] = {
                "dtype": meta["dtype"],
                "shape": meta["shape"],
                "data_offsets": [base + start, base + end],
            }
        all_data.extend(data)

    # Write consolidated file
    header_json = json.dumps(all_meta).encode("utf-8")
    out_path = os.path.join(out_dir, f"model-rank-{rank:05d}-of-00008.safetensors")
    with open(out_path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header_json)))
        fh.write(header_json)
        fh.write(all_data)

    print(f"Rank {rank}: {len(files)} parts -> {os.path.getsize(out_path)/1e9:.2f} GB, {len(all_meta)} tensors")

# Copy non-safetensors files
for f in glob.glob(os.path.join(model_dir, "*")):
    base = os.path.basename(f)
    if not base.endswith(".safetensors") and not os.path.isdir(f):
        import shutil
        shutil.copy2(f, out_dir)
        print(f"Copied {base}")

print("Done")

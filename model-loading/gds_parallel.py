import torch, torch.cuda.gds as gds, os, glob, time, sys
import torch.multiprocessing as mp

def load_rank_worker(rank, model_dir, result_dict):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    # Support both naming conventions
    files = sorted(glob.glob(os.path.join(model_dir, f"model-rank-{rank:05d}-of-*.safetensors")))
    if not files:
        files = sorted(glob.glob(os.path.join(model_dir, f"model-rank-{rank}-part-*.safetensors")))
    total = 0
    for f in files:
        fsize = os.path.getsize(f)
        storage = torch.UntypedStorage(fsize, device=device)
        gds.gds_register_buffer(storage)
        gds.GdsFile(f, os.O_RDONLY).load_storage(storage)
        gds.gds_deregister_buffer(storage)
        total += fsize
        del storage
    result_dict[rank] = total

def main():
    model_dir = sys.argv[1]
    os.system("sudo rmmod nvidia_fs; sudo modprobe nvidia_fs; sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'")

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    result_dict = manager.dict()

    t0 = time.perf_counter()
    processes = []
    for rank in range(8):
        p = mp.Process(target=load_rank_worker, args=(rank, model_dir, result_dict))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    t1 = time.perf_counter()

    total = sum(result_dict.values())
    for rank in range(8):
        print(f"  Rank {rank}: {result_dict[rank]/1e9:.2f} GB")
    print(f"\nAll 8 GPUs: {total/1e9:.2f} GB in {t1-t0:.3f}s = {total/1e9/(t1-t0):.2f} GB/s")

    with open("/proc/driver/nvidia-fs/stats") as sf:
        for line in sf:
            l = line.strip()
            if "Reads" in l or "Ops" in l or "err" in l:
                print(l)

if __name__ == "__main__":
    main()

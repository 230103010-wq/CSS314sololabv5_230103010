import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory

from harness import run_benchmark, emit_verify_tag

STUDENT_ID = 230103010
PPHYS = 4
PLOG = 8
M = 24_000_000
SHM_SIZE = 4096


def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    acc = arr[slot_idx]
    mask = 0xFFFFFFFFFFFFFFFF
    mult = 6364136223846793005
    for i in range(iters):
        acc = (acc * mult + (i | 1)) & mask
        arr[slot_idx] = acc  
    del arr  
    shm.close()


def run_once(shm: shared_memory.SharedMemory, p: int, stride: int) -> None:
    shm.buf[:SHM_SIZE] = bytes(SHM_SIZE)  # reset slots -> deterministic
    iters = M // p
    procs = [
        mp.Process(target=fs_worker, args=(shm.name, t * stride, iters))
        for t in range(p)
    ]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()


def bench(shm, p: int, stride: int):
    return run_benchmark(lambda: run_once(shm, p, stride), warmup_runs=2, timed_runs=7)


def main() -> None:
    shm = shared_memory.SharedMemory(create=True, size=SHM_SIZE)
    try:
        print(f"\n## Table 1A: stride sweep, p = {PPHYS}")
        print("| stride | bytes | median_ms | IQR_ms | Mops/s |")
        print("|---|---|---|---|---|")
        med = {}
        for stride in (1, 2, 4, 8, 16, 32):
            s = bench(shm, PPHYS, stride)
            med[stride] = s.median_ms
            mops = M / (s.median_ms / 1000) / 1e6
            print(f"| {stride} | {stride*8} | {s.median_ms:.2f} | {s.iqr_ms:.2f} | {mops:.2f} |", flush=True)

        # ---- Table 1B: process scaling at stride 1 and 16 ----
        plist = [1, 2, 4, PLOG]
        for stride in (1, 16):
            print(f"\n## Table 1B: scaling, stride = {stride} ({stride*8} B)")
            print("| p | T_p_ms | S_p | E_p | Karp-Flatt e(p) |")
            print("|---|---|---|---|---|")
            t1 = None
            for p in plist:
                s = bench(shm, p, stride)
                if t1 is None:
                    t1 = s.median_ms
                sp = t1 / s.median_ms
                ep = sp / p
                if p >= 2:
                    e = (1 / sp - 1 / p) / (1 - 1 / p)
                    es = f"{e:.4f}"
                else:
                    es = "n/a"
                print(f"| {p} | {s.median_ms:.2f} | {sp:.3f} | {ep:.3f} | {es} |", flush=True)

        print()
        print(emit_verify_tag("PHASE1", STUDENT_ID, med[1], med[16]))
    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()
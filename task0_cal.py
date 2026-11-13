import multiprocessing as mp
import time

from harness import emit_verify_tag

STUDENT_ID = 230103010
READS = 2_000_000


def main() -> None:
    plog = mp.cpu_count()
    clock = time.perf_counter_ns

    # warm-up
    for _ in range(100_000):
        clock()

    t0 = clock()
    for _ in range(READS):
        clock()
    t1 = clock()

    avg_ns = (t1 - t0) / READS

    print(f"mp.cpu_count() = {plog}")
    print(f"Average perf_counter_ns call latency over {READS} reads: {avg_ns:.2f} ns")
    print(emit_verify_tag("PHASE0", STUDENT_ID, avg_ns, float(plog)))


if __name__ == "__main__":
    main()
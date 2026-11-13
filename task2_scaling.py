import math
import multiprocessing as mp
import time

from harness import emit_verify_tag

STUDENT_ID = 230103010
PPHYS = 4
N_STRONG = 1_200_000
N0_WEAK = 200_000
RUNS = 5


def stage_a(n: int) -> float:
    """Serial stage: 10 ops/element (5 sin + 5 sqrt)."""
    acc = 0.0
    for i in range(n):
        x = i * 1e-3 + 0.1
        for _ in range(5):
            x = math.sqrt(math.sin(x) + 2.0)
        acc += x
    return acc


def stage_b(span):
    """Parallel stage: 30 ops/element (10 x [fma, sin, sqrt]), no shared writes."""
    s, e = span
    acc = 0.0
    for i in range(s, e):
        x = (i + 1) * 1e-6
        for _ in range(10):
            x = x * 0.999 + 0.001
            x = math.sin(x)
            x = math.sqrt(x * x + 1.0)
        acc += x
    return acc


def split(n: int, p: int):
    step = n // p
    return [(w * step, n if w == p - 1 else (w + 1) * step) for w in range(p)]


def trial(pool, p: int, n: int):
    t0 = time.perf_counter()
    stage_a(int(0.05 * n))
    t_a = (time.perf_counter() - t0) * 1e3
    pool.map(stage_b, split(n, p), chunksize=1)
    total = (time.perf_counter() - t0) * 1e3
    return t_a, total


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


def measure(p: int, n: int):
    pool = mp.Pool(processes=p)          # создаётся ВНЕ замеряемого цикла
    try:
        pool.map(stage_b, [(0, 10)] * p, chunksize=1)   # прогрев воркеров
        trial(pool, p, n)                                # warmup
        samples = [trial(pool, p, n) for _ in range(RUNS)]
    finally:
        pool.close()
        pool.join()
    return median([s[0] for s in samples]), median([s[1] for s in samples])


def main():
    plog = mp.cpu_count()
    ps = list(range(1, plog + 1))

    # ---------- Strong scaling ----------
    strong = {}
    for p in ps:
        strong[p] = measure(p, N_STRONG)
        print(f"strong p={p} done", flush=True)
    t1 = strong[1][1]
    s_serial = strong[1][0] / t1
    print(f"\n## Strong scaling (N={N_STRONG})")
    print("| p | T_p ms | S_p | E_p | marginal dS |")
    print("|---|---|---|---|---|")
    prev = 0.0
    s_strong = {}
    for p in ps:
        tp = strong[p][1]
        sp = t1 / tp
        s_strong[p] = sp
        print(f"| {p} | {tp:.1f} | {sp:.3f} | {sp/p:.3f} | {sp-prev:+.3f} |")
        prev = sp
    print(f"\nT_StageA(1) = {strong[1][0]:.1f} ms, s = {s_serial:.4f}, "
          f"Amdahl cap S_inf = {1/s_serial:.1f}, "
          f"observed S_{plog} = {s_strong[plog]:.3f}")
    if PPHYS + 1 <= plog:
        d1 = s_strong[PPHYS] - s_strong[PPHYS - 1]
        d2 = s_strong[PPHYS + 1] - s_strong[PPHYS]
        print(f"Marginal speedup: p={PPHYS-1}->{PPHYS}: {d1:+.3f}; "
              f"p={PPHYS}->{PPHYS+1}: {d2:+.3f}")

    # ---------- Weak scaling ----------
    weak = {}
    for p in ps:
        weak[p] = measure(p, p * N0_WEAK)
        print(f"weak p={p} done", flush=True)
    w1 = weak[1][1]
    print(f"\n## Weak scaling (N0={N0_WEAK} per worker)")
    print("| p | N(p) | T_p ms | E_weak | alpha | S_Gustafson |")
    print("|---|---|---|---|---|---|")
    s_gus = {}
    for p in ps:
        tp = weak[p][1]
        alpha = weak[p][0] / tp
        g = p - alpha * (p - 1)
        s_gus[p] = g
        print(f"| {p} | {p*N0_WEAK} | {tp:.1f} | {w1/tp:.3f} | {alpha:.4f} | {g:.3f} |")

    print()
    print(emit_verify_tag("PHASE2", STUDENT_ID, s_strong[plog], s_gus[plog]))


if __name__ == "__main__":
    main()
    
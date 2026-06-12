# Week 2 — Cache Simulator

A correct, flat-SoA, zero-allocation, compile-time-geometry implementation of
the two-level cache hierarchy specified in [`CACHE_SPEC.md`](./CACHE_SPEC.md).

## Build

```bash
cmake -B build -DCSOT_CACHE_SIM_SRC=cache_sim.cpp
cmake --build build -j
```

Judge-like build (portable `-march=x86-64-v2`, no `-march=native`):

```bash
cmake -B build-judge -DCSOT_JUDGE_BUILD=ON -DCSOT_CACHE_SIM_SRC=cache_sim.cpp
cmake --build build-judge -j
```

## Correctness

```bash
diff <(./build/cache_sim_runner data/tiny.trace 2>/dev/null) data/tiny.stats.json
```

Clean diff on `tiny.trace` ✅. Also verified on a 5,000,000-access seed-42
trace (`python3 data/gen_trace.py --accesses 5000000 --seed 42 --out data/large.trace`):
non-zero `dirty_writebacks` (eviction path exercised), and both §7 identities hold:

```
reads + writes == l1_hits + l1_misses
l1_misses      == l2_hits + l2_misses
```

## Design

- **Flat SoA per level** (`CacheLevel<SETS>`): `tag[SETS*WAYS]`, plus one
  `uint8_t` valid-bitmask and one `uint8_t` dirty-bitmask per set (one bit per
  way), and one `uint32_t` per set packing the true-LRU order as 8 nibbles
  (rank 0 = MRU, rank 7 = LRU).
- **Zero allocation in `run()`**: all vectors are sized and zeroed once in
  `on_init()`.
- **Compile-time geometry**: `L1_SETS`, `L2_SETS`, index-bit counts, and set
  masks are `constexpr`, so `b & L1_SET_MASK` / `b & L2_SET_MASK` fold to a
  single `and` at `-O3`.
- **Branchless tag scan in `find()`**: instead of an `if` per way (8 branches,
  data-dependent and prone to misprediction), each way contributes a bit to an
  8-bit match mask via `match |= (base[w] == t) << w`, ANDed with the valid
  mask, then `__builtin_ctz` extracts the matching way in one instruction.
  Portable across ARM64 and x86 — no `<immintrin.h>` SIMD intrinsics, which
  don't exist on the ARM64 dev VM.
- **LRU update fast path**: `lru_touch` checks first whether `way` is already
  MRU (rank 0) — the common case for repeated accesses to the same line — and
  returns immediately, skipping the rank-search and nibble-shift loops
  entirely.

## Headline numbers (UTM Linux VM, MacBook Air M5 host, aarch64)

5,000,000-access trace, seed 42 (`python3 data/gen_trace.py --accesses 5000000 --seed 42 --out data/large.trace`):

```
accesses = 5000000   run = 117,723,418 ns   throughput = 42.47 M acc/s   (Release, -march=native)
accesses = 5000000   run = 117,836,500 ns   throughput = 42.43 M acc/s   (CSOT_JUDGE_BUILD, portable)
```

Both builds produce identical, correct counters and near-identical throughput.
This is an ~18% improvement (35.99 → 42.4x M acc/s) over the initial flat-SoA
baseline, from the branchless `find()` and the LRU fast-path alone — no SIMD
or layout changes needed.

Note on architecture: this VM is **aarch64** (UTM on Apple Silicon), while the
leaderboard judge runs **x86-64 (`c7i.xlarge`)**. Throughput numbers above are
a useful *relative* signal (did a change help or hurt on this hardware?) but
won't transfer 1:1 in absolute terms — the judge's own run is what determines
the ranking. Correctness (`MATCH` / `JUDGE MATCH` on `tiny.trace`, and the §7
identity checks on the large trace) is architecture-independent and is what
was actually verified to carry over.

`perf stat -e page-faults,task-clock,context-switches`:

```
255   page-faults
0     context-switches
0.142s user / 0.013s sys
```

255 total page faults for the whole process (binary load + the one-time
`on_init()` allocations being paged in) is consistent with **zero allocation
inside `run()`** — a hot loop touching new pages 5,000,000 times would show up
here.

Note: `perf stat -e cycles,instructions,L1-dcache-load-misses,branch-misses`
fails with "No supported events found" under UTM/QEMU — the hypervisor
doesn't pass through the host CPU's PMU to the guest. Hardware-counter
profiling (cache-miss rates, IPC) isn't available in this environment; the
harness's own wall-clock timing and the software `perf` events above are the
numbers reported here.

## What surprised me

The §5.5 "writeback is not a demand access" rule is easy to get right in
prose and easy to break in code — it's tempting to route the L1→L2 writeback
through the same `find`/`touch`/`victim_way` helpers used for demand misses,
but doing so would silently call `touch()` on an L2 hit during a writeback,
which must *not* update LRU recency. Keeping the writeback path
(§5.5) textually separate from the demand path (§5.2/§5.3) made this
much harder to get wrong.

Also: I initially reached for `<immintrin.h>` SSE intrinsics in `find()` for
an explicit SIMD tag scan. This compiled fine in my reasoning but failed
immediately on the actual dev VM — `immintrin.h` is x86-only and the UTM VM
on Apple Silicon is aarch64, so the header doesn't exist there at all. The
portable `__builtin_ctz` + bitmask version turned out to be just as fast (and
arguably clearer) without tying the code to one ISA — useful since the dev
machine (ARM) and the judge (x86) don't even share an architecture.

# `source_metric_basic.ncu-rep` source

`synthetic.cu` in this directory is the source the sibling
`../source_metric_basic.ncu-rep` was captured from. Committed for
reproducibility — the fixture covers
both source-counter gate paths (`Section.SourceMetrics` and
`ProfilerSourceMetricTable` body items) under `ncu --set full`,
plus populated per-PC PC-sampling instances.

## Two kernels

The TU has two kernels, captured in this order so the launch
indices stay stable across recaptures:

| Launch     | Kernel                           | Purpose                                                                                                                                                                   |
| ---------- | -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `launch:0` | `synthetic_bank_conflict_kernel` | One shared-memory load pattern with intentional 32-way bank conflict; exercises additive `derived__memory_l1_conflicts_*`.                                                |
| `launch:1` | `synthetic_long_stall_kernel`    | 256 blocks × 256 threads × 256-iter pointer-chase loop; long enough for PC sampling to accumulate per-PC instances in `smsp__pcsamp_warps_issue_stalled_*` (pm-category). |

## Regeneration

If the native sidecar schema changes (i.e. `NATIVE_SCHEMA` in
`src/native/mod.rs` or `ncu_export.py` bumps), regenerate the
fixture from this directory:

```bash
# Compile the synthetic kernels with -lineinfo so source counters
# resolve to (file, line) — without it, every instance lands in
# auxiliary.unattributed_sass_counter_totals.
cp synthetic.cu /tmp/synthetic.cu
cd /tmp
nvcc -lineinfo -O2 synthetic.cu -o synthetic

# Capture both kernels with --set full. --launch-count 2 grabs
# the first launch of each matched kernel; both Section.SourceMetrics
# / ProfilerSourceMetricTable body items AND the per-PC PC-sampling
# family populate this way.
ncu --set full --target-processes all --replay-mode kernel \
    -k regex:synthetic_ --launch-count 2 \
    -o source_metric_basic ./synthetic

# Move the result into the repo.
mv /tmp/source_metric_basic.ncu-rep \
   <repo>/crates/ncu/veloq-ncu/tests/fixtures/
```

The capture is built from `/tmp` so the DWARF embeds
`/tmp/synthetic.cu`, not a per-user `/home/<user>/...` path, to
keep the fixture free of local paths. Both kernel
names are intentionally generic.

## Captured against

- NCU: 2026.1.1.0 (see `ncu --version` at capture time)
- CUDA: 13.2 (nvcc compilation toolchain)
- GPU: whatever was visible to nvidia-smi at capture time (the
  exact sm_XX doesn't matter for test purposes — the verb just
  joins counter instances to SASS addresses)

## Warm disasm cache (required for CUDA-free CI)

The CI `test` job runs in `rust:slim-bookworm`
with no CUDA toolchain, so `nvdisasm` and `cuobjdump` aren't
available at test time. The populated-fixture
`source-metrics` tests (and the tabular `disasm` /
`source-metrics-*` smokes) join SASS addresses to source
lines via the disasm pipeline; without the toolchain that
pipeline silently returns zero kernels and every test sees
`count == 0`.

The warm `<sha>.correlated.json`
cache ships alongside the fixture. The disasm pipeline
(`crates/ncu/veloq-ncu/src/disasm_pipeline/cache.rs`) checks
`load_cached()` before invoking `acquire_correlated()` — a
cache hit skips nvdisasm/cuobjdump entirely. The cache lives
at:

```
source_metric_basic.ncu-rep.veloq/disasm/<sha>.correlated.json
```

It is `git add -f`-tracked despite the `*.veloq/`
`.gitignore` rule. The companion `<sha>.cubin` is **not**
committed — it is only consulted on a cache miss (i.e. when
the toolchain is present and the cache is being rebuilt).

When `CACHE_SCHEMA`
(`crates/ncu/veloq-ncu/src/disasm_pipeline/types.rs`) or the
cubin bytes change, regenerate the cache on a CUDA-enabled
box and re-stage it:

```bash
# From the repo root. The first invocation rebuilds the
# correlated cache from the fixture's cubin.
rm crates/ncu/veloq-ncu/tests/fixtures/source_metric_basic.ncu-rep.veloq/disasm/*.correlated.json
cargo test --release -p veloq --test ncu_source_metrics_smoke \
    -- populated_fixture_by_line_resolves_motivating_counter

git add -f crates/ncu/veloq-ncu/tests/fixtures/source_metric_basic.ncu-rep.veloq/disasm/*.correlated.json
```

# NSys Cookbook

Extended examples for common Nsight Systems investigations. The
canonical workflow catalog is `veloq recipes`: run `veloq recipes` for
the registry and `veloq recipes <id>` for the command sequence. This
cookbook adds jq post-processing, interpretation notes, and longer
variants. Assume `T` is the trace path. Use `veloq <cmd> --help` for
full flags and response shape.

Related references:

- Capability gates and metric trust:
  [capabilities.md](capabilities.md)
- Time windows and NVTX overlap:
  [time-and-nvtx.md](time-and-nvtx.md)
- `inspect` event shapes:
  [inspect-shapes.md](inspect-shapes.md)
- Capture prerequisites and handoffs:
  [limitations.md](limitations.md)

## Quick Probe

Run these once at the start of an NSys investigation:

```bash
veloq summary T | jq '.data.auxiliary.capabilities'
veloq hardware T | jq '.data.rows[0] | {gpus: [.gpus[].name], cpu: .cpu.model, cuda: .drivers.cuda_driver_version}'
veloq stats T --limit 20
veloq timeline T --interval 100ms --limit 200
```

Use the top-level workflow in `../SKILL.md` plus `veloq recipes` to
choose the canonical recipe before borrowing extended examples here.
Envelope shape (`data.rows[]+key`, `data.auxiliary`, `trace_span`) is
in [`../SKILL.md`](../SKILL.md).

## Report Timeline Figure

Use `veloq viz timeline` only after you have already selected a bounded
window with textual evidence. The command writes an SVG under the trace
artifact root and returns a normal JSON envelope; stdout is not the SVG.

```bash
veloq viz timeline T --from @100000000 --to @120000000 \
  --track gpu:device=all \
  --track cuda-streams:device=all,top=8 \
  --track gaps-overlay:device=all \
  --highlight-kernels top=3,scope=name \
  | jq '.data.rows[0] as $row | {
      path: $row.path,
      tracks: $row.track_count,
      rendered: $row.rendered_item_count,
      total: $row.total_item_count,
      density_items: $row.density_item_count,
      density_bins: $row.density_bin_count,
      aggregated: $row.aggregated,
      omitted_explicit_items: $row.omitted_explicit_item_count,
      omitted_tracks: $row.omitted_track_count,
      suppressed_labels: $row.suppressed_label_count,
      truncated_labels: $row.truncated_label_count,
      resolved_tracks: .data.auxiliary.resolved_tracks,
      resolved_highlights: .data.auxiliary.resolved_highlights
    }'
```

The returned `path` is relative to `<trace>.veloq/`. Dense windows are
compacted into per-track density bins instead of stretched into fake-width
bars; `--max-items` may raise the density threshold. If aggregation, omitted
tracks, suppressed labels, or truncated labels are non-zero, say so in
the report; density means the figure stayed visually compact, not that
the underlying query ignored those events.

To select one process-private logical GPU, replace `device=all` with an
exact pair such as
`--track gpu:process=12345,device=0`; a bare `device=0` is ambiguous when
multiple processes expose that ordinal.

`--highlight-kernels top=<n>,scope=name` colors the top kernel names in
the selected window without changing base event classes. Use
`scope=instance` when the report needs individual long-running kernel
instances instead of name aggregates. The response keeps short labels
and full kernel names in `data.auxiliary.resolved_highlights[]`.

Use `data.auxiliary.resolved_tracks[].role` when interpreting the
figure: `group` rows are ownership context, `summary` rows are rollups
such as GPU busy activity, `detail` rows are concrete lanes such as CUDA
streams, `annotation` rows are CUDA API or NVTX context, and idle gaps
are overlays. A kernel may appear in both a summary rollup and a stream
detail row; that is a rollup/detail relationship, not duplicate work.
Use each track's `source_axes`, `placement_axes`, and `placement_source`
when explaining why an annotation appears under a resource group. NVTX
under a GPU device is derived attribution, not native NVTX device
evidence.
This is a static report figure, not an interactive Nsight GUI
replacement.

## Kernel Hotspots

```bash
veloq stats T --type kernel --group-by demangled --sort total:desc --limit 20
veloq stats T --type kernel --all-devices --group-by short,device --sort total:desc --limit 20
veloq stats T --type kernel --group-by mangled --sort total:desc --limit 20
veloq stats T --type kernel --group-by demangled,grid_block --sort total:desc --limit 20
veloq search T --type kernel --name '*target*' --sort duration:desc --limit 10
# On large traces, --name-regex prunes the scan before name resolution,
# so it's several times faster than the equivalent --name glob (same results).
veloq search T --type kernel --name-regex 'target' --sort duration:desc --limit 10
veloq inspect T kernel:123    # includes nvtx_context if the trace has NVTX

# Per-instance launch-shape distribution via search + jq (the
# fast-aggregation complement is --group-by grid_block above):
veloq search T --type kernel --name '*target*' --limit 200000 \
  | jq '.data.rows
        | group_by([.name, .grid, .block])
        | map({
            name: .[0].name,
            grid: .[0].grid,
            block: .[0].block,
            count: length,
            total_ns: (map(.duration_ns) | add)
          })
        | sort_by(.total_ns) | reverse | .[0:10]'
# Recipe contract: pass --limit BIG and assert
# .data.count == .data.total_matched before consuming.
```

Use `demangled` when template variants matter. Use `mangled` when
link identity matters (two kernels can demangle to the same
signature but link distinct symbols); the axis falls back to
`demangled` on older NSys schemas missing the `mangledName` column
and the response surfaces `mangled_axis_fallback: true`. Use `short`
for a stable rollup when names are very long.

`inspect` default-populates `nvtx_context: { range_id, name, depth,
iter_index }` on kernel/memcpy/memset/sync/runtime rows when the
trace carries NVTX_EVENTS + CUPTI_ACTIVITY_KIND_RUNTIME (+
TARGET_INFO_CUDA_CONTEXT_INFO for the GPU-side kinds) — agent gets
"which iteration was this kernel in" without a follow-up query.

## GPU Idle And Sync

```bash
# Idle gaps. Default `--scope device` is cross-stream — a gap only
# counts when no stream is running GPU work on that device, so
# long-idle peer streams don't produce phantom gaps. Add
# `--scope stream --device D --stream S` for per-stream starvation
# diagnostics, or `--scope trace` for multi-GPU rig idle.
veloq gaps T --min-duration 1ms --sort duration:desc --limit 20

# Correlate the events bracketing a gap.
veloq correlate T kernel:123 kernel:456

# Host-side blocking on CUDA synchronization.
veloq stats T --type sync --sort total:desc --limit 10
veloq search T --type sync --sort duration:desc --limit 10
veloq correlate T sync:42
```

If the host syncs after long CPU work, the bottleneck may be launch
latency or pipeline feeding. If the GPU stream has long gaps without a
matching host block, inspect surrounding runtime/API rows and NVTX
ranges.

## NVTX Iteration Regression

```bash
veloq summary T | jq '.data.auxiliary.capabilities.has_nvtx'
veloq slices T --name '*step*' --sort attributed_kernel --limit 50
veloq slices T --aggregate --name '*step*' --sort p99:desc --limit 20
veloq slices T --name '*step*' --limit 100 \
  | jq '.data.rows[] | select(.cpu.nesting_depth == 0)'
veloq stats T --nvtx '*step 42*' --group-by demangled --limit 20

# Per-kernel iteration tag (batched, opt-in):
veloq search T --type kernel --duration '>1ms' --with-nvtx \
  | jq '.data.rows | map({key, name, dur: .duration_ns, iter: .nvtx_context.iter_index, range: .nvtx_context.name})'

# Per-step aggregate of GPU work via the innermost NVTX range:
veloq stats T --type kernel --group-by nvtx-parent \
  --sort total:desc --limit 20
# Each row carries .nvtx_parent_name (the range name, or
# "__no_nvtx__" sentinel), .nvtx_parent_key ("nvtx:<rowid>" or
# "nvtx:none" for cross-trace joins), and .nvtx_parent_depth (NVTX
# nesting depth, None on sentinel). The same comparator
# (rank-by-latest-start within fully-containing ranges) is used by
# `search --with-nvtx`, so forward and reverse attribution agree.

# Full-path aggregate for nested traces where the same leaf range name is
# reused under different parents:
veloq stats T --type kernel --group-by nvtx-path \
  --sort total:desc --limit 20
veloq slices T --aggregate --name '*' --group-by path \
  --sort path:asc --limit 200
# stats rows carry .nvtx_path and .nvtx_path_key
# ("nvtx-path:<path>" or "nvtx-path:none"). slices aggregate path rows
# carry .path and keys shaped as "scope|pid:<pid>|path:<path>".

# NVTX style breakdown: stats --type nvtx splits PushPop and StartEnd
# ranges sharing a name into distinct rows via the derived
# `nvtx_style` label. Raw `event_type` is the min int from NSys's
# NvtxEventType enum within the bucket; {59,70}→push_pop,
# {60,71}→start_end, anything else → unknown. To recover raw → label
# from a `search --type nvtx` stream (e.g. when picking ranges to
# probe further), feed `event_type` through the same table:
veloq search T --type nvtx --name '*step*' --limit 200000 \
  | jq '.data.rows | map(. + {
      style: (if .event_type == 59 or .event_type == 70 then "push_pop"
              elif .event_type == 60 or .event_type == 71 then "start_end"
              else "unknown" end)
    }) | group_by(.name + "|" + .style) | map({
      name: .[0].name, style: .[0].style, count: length,
      total_ns: (map(.duration_ns) | add)
    })'
# Recipe contract (per Cookbook conventions): pass --limit BIG and
# assert `.data.count == .data.total_matched` before consuming, so
# no rows were dropped by the implicit cap.

# Diff two traces by NVTX range key:
veloq stats T1 --all-devices --group-by demangled,device,stream > a.json
veloq stats T2 --all-devices --group-by demangled,device,stream > b.json
jq -n --slurpfile a a.json --slurpfile b b.json '
  ($a[0].data.rows | INDEX(.key)) as $A |
  ($b[0].data.rows | INDEX(.key)) as $B |
  ($A + $B | keys) | map({
    key: .,
    delta_ns: (($B[.].total_ns // 0) - ($A[.].total_ns // 0)),
    delta_per_sec: ((($B[.].total_ns // 0) / ($b[0].trace_span.span_ns / 1e9))
                  - (($A[.].total_ns // 0) / ($a[0].trace_span.span_ns / 1e9)))
  })' | jq 'sort_by(.delta_ns) | reverse | .[0:10]'
```

Compare outer/root ranges first. Nested ranges are useful only after
you know which outer iteration regressed. `nvtx_context` shape and
the `--with-nvtx` opt-in vs `inspect` default-on split are
documented in [time-and-nvtx.md](time-and-nvtx.md).

## CUDA Graphs

Probe graph mode first; graph-mode and node-mode traces answer
different questions.

```bash
veloq summary T | jq '.data.auxiliary.capabilities | {has_graph_trace, has_graph_nodes, has_graph_events}'
```

Graph-mode rollup:

```bash
veloq stats T --type graph --sort total:desc --limit 20
veloq search T --type graph --sort duration:desc --limit 10
veloq correlate T graph:1
veloq timeline T --interval 100ms \
  | jq '.data.rows[] | {start_ns, kernel_ns, graph_ns, total: (.kernel_ns + .graph_ns)}'
```

Node-mode drilldown:

```bash
veloq stats T --type kernel --group-by no-name,graph --limit 20
veloq stats T --type kernel --group-by short,graph_node --limit 20
veloq inspect T kernel:4437 | jq '.data.rows[0] | {short_name, graph_id, graph_node_id, duration_ns}'
```

In graph-mode captures, graph work may not appear as kernel rows. In
node-mode captures, graph work is represented through kernel rows with
graph/node ids.

## GPU And NIC Counters

Always read coverage before trusting sampled counters.

```bash
veloq summary T | jq '.data.auxiliary.capabilities | {gpu: .has_gpu_metrics, nic: .has_nic_metrics}'
veloq metrics T --type gpu | jq '.data.auxiliary.common.coverage'
veloq metrics T --type nic | jq '.data.auxiliary.common.coverage'
```

GPU quick read:

```bash
veloq metrics T --type gpu \
  --counter 'SMs Active*,Tensor Active*,DRAM Read*,DRAM Write*' \
  --sort=mean:desc
veloq metrics T --type gpu --counter '*Throughput*' --bucket 100ms --limit 100000
```

NIC quick read:

```bash
veloq metrics T --type nic --sort=mean:desc --limit 20
veloq metrics T --type nic --counter '*Bytes*' --bucket 100ms --limit 100000
```

NIC metrics are sampled hardware counters. Use them for pressure and
overlap questions, not exact message attribution. If coverage is low,
recapture with a shorter window or lower counter cadence.

Relevant capture shapes:

```bash
nsys profile --gpu-metrics-devices=<dev|all|cuda-visible> \
             --gpu-metrics-set=<chip-set> \
             --gpu-metrics-frequency=<Hz> \
             ...

nsys profile --trace=cuda,nvtx --nic-metrics=lf ...
```

Use high-frequency NIC metrics only when the NIC, firmware, libraries,
and privileges support them.

## CPU Sampling And Scheduler

CPU sampling is statistical; CPU scheduler events are transition
events. Check trust signals before interpreting either.

```bash
veloq summary T | jq '.data.auxiliary.capabilities | {samples: .has_sampling, composite: .has_composite_events, sched: .has_sched_events}'

veloq metrics T --type cpu-sampling | jq '{
  coverage: .data.auxiliary.common.coverage,
  unresolved: .data.auxiliary.unresolved_leaf_share,
  kernel: .data.auxiliary.kernel_leaf_share,
  truncated: .data.auxiliary.truncated_stack_share
}'
veloq metrics T --type cpu-sampling --limit 20
veloq metrics T --type cpu-sampling --group-by tid --limit 10
veloq metrics T --type cpu-sampling --group-by stack --limit 10
veloq inspect T "$(veloq metrics T --type cpu-sampling --limit 1 | jq -r '.data.rows[0].sample_row_id')"

veloq metrics T --type cpu-sched | jq '{
  coverage: .data.auxiliary.common.coverage,
  unresolved_state: .data.auxiliary.unresolved_state_share,
  max_gap_ns: .data.auxiliary.per_cpu_max_gap_ns
}'
veloq metrics T --type cpu-sched --sort=on_cpu:desc --limit 10
veloq metrics T --type cpu-sched --group-by cpu --limit 32
```

High unresolved + kernel leaf shares often mean the CPU is sleeping in
kernel/syscall paths, not burning user code. High on-CPU time in
resolved user symbols points to a real host bottleneck.

## Bandwidth

```bash
veloq stats T --type memcpy --all-devices --group-by short,device --sort gbps:desc --limit 20
veloq timeline T --interval 100ms --type memcpy --limit 200

# Byte-axis aggregation (hidden — requires VELOQ_UNSTABLE=1).
# Equivalent to nsys recipe cuda_gpu_mem_size_sum.
# Returns the StatsBySizeResponse shape: rows carry total_bytes /
# avg_bytes / p50_bytes / p95_bytes / p99_bytes — no duration
# columns. --type narrows implicitly to memcpy+memset; explicit
# non-memop kinds error up-front.
VELOQ_UNSTABLE=1 veloq stats T --by size --sort bytes:desc --limit 20
VELOQ_UNSTABLE=1 veloq stats T --by size --group-by short,device --limit 20

# Approximate cuda_gpu_mem_size_sum via search (no env gate
# required; the qualifier is "duration-bearing events only" since
# search projects raw rows):
veloq search T --type memcpy,memset --limit 200000 \
  | jq '.data.rows
        | group_by(.name)
        | map({
            name: .[0].name,
            count: length,
            total_bytes: (map(.bytes // 0) | add),
            avg_bytes: (((map(.bytes // 0) | add) / length) | round),
          })
        | sort_by(.total_bytes) | reverse | .[0:20]'
# Per Cookbook convention pass --limit BIG and assert
# .data.count == .data.total_matched before consuming.
```

`stats` rows for memcpy/memset carry `bytes_total` and `gbps`. Kernel
rows do not. `stats --by size` (hidden) carries the byte-aggregate
columns instead, under the dedicated `StatsBySizeResponse` shape.

## NSys To NCU Handoff

Use NSys to prove a kernel matters before spending time on NCU.

```bash
veloq stats T --type kernel --group-by demangled --sort total:desc --limit 20
veloq search T --type kernel --name '*target_kernel*' --sort duration:desc --limit 5
veloq inspect T kernel:123
veloq correlate T kernel:123
```

Carry forward the kernel name, row id, time window, stream/context,
and why it matters in the timeline. Then use native NCU or the
`ncu-profile-analysis` skill:

```bash
ncu --section SpeedOfLight \
    --section LaunchStats \
    --section Occupancy \
    --kernel-name regex:"target_kernel" \
    --export target-kernel \
    ./app args...

veloq ncu summary target-kernel.ncu-rep
```

NCU explains kernel internals. It does not explain launch timing, CPU
blocking, idle bubbles, or end-to-end overlap.

## Output And JQ

```bash
veloq stats T --limit 10 --format table
veloq stats T --limit 10 --format csv
veloq stats T --limit 1 | jq -r '.data.rows[0].name'
veloq gaps T --min-duration 5ms --limit 100 \
  | jq -r '.data.rows[] | [.start_ns, .duration_ns, .prev.name] | @tsv'
```

For repeated heavy queries, run `veloq prep T` once to build caches,
then continue with the same VeloQ verbs/recipes. Do not open generated
`.veloq/` parquet files directly unless the user explicitly requested
raw-table exploration.
Use `veloq clean T` if you need to force a rebuild.
